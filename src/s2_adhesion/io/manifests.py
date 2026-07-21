"""Artifact manifests: schema, validation, and canonical serialisation.

A manifest is the metadata that rides alongside a zarr or TIFF artifact so a
label volume produced on one machine can be *trusted* -- or rejected -- on
another with no ML stack installed at all. Every manifest is validated on
read: a missing or malformed field is :class:`~s2_adhesion.errors.ArtifactError`;
a label manifest that does not describe the image it claims to is
:class:`~s2_adhesion.errors.ArtifactBindingError`, never a bare
``ValueError`` and never the in-memory :class:`~s2_adhesion.errors.ContractViolation`
that :meth:`~s2_adhesion.contracts.LabelVolume.validate_against` raises --
that one guards in-memory pairing, this module guards artifacts on disk.

JSON on disk is always written with :func:`canonical_json`: sorted keys, a
fixed separator, ascii-only. Two writers given the same manifest content
therefore produce byte-identical files, so hashing or diffing a manifest.json
is meaningful.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, TypedDict

import numpy as np

from ..contracts import (
    SCHEMA_VERSION_IMAGE,
    SCHEMA_VERSION_LABELS,
    ChannelBinding,
    ChannelRole,
    FieldIdentity,
    ImageVolume,
    LabelVolume,
    SegmentationProvenance,
)
from ..errors import ArtifactBindingError, ArtifactError

__all__ = [
    "ChannelManifestEntry",
    "ImageArtifactManifest",
    "LabelEntryManifest",
    "SegmentationProvenanceManifest",
    "LabelArtifactManifest",
    "canonical_json",
    "sha256_of_array",
    "build_image_manifest",
    "validate_image_manifest",
    "image_identity_from_manifest",
    "image_channels_from_manifest",
    "build_label_manifest",
    "validate_label_manifest",
    "label_identity_from_manifest",
    "provenance_from_manifest",
    "check_label_image_binding",
]


# ─── manifest shapes ───────────────────────────────────────────────────────


class ChannelManifestEntry(TypedDict, total=False):
    channel_id: str
    source_index: int
    roles: list[str]
    source_name: str | None
    color: str | None
    population: str | None
    protein: str | None


class ImageArtifactManifest(TypedDict):
    schema_version: str
    artifact_id: str
    dataset_id: str
    field_id: str
    source_uri: str
    source_field_index: int
    image_content_sha256: str
    axes: str
    shape: list[int]
    dtype: str
    spacing_um_zyx: list[float]
    origin_um_zyx: list[float]
    channels: list[ChannelManifestEntry]
    array_sha256: str
    created_utc: str


class LabelEntryManifest(TypedDict):
    name: str
    present: bool
    array_sha256: str | None


class SegmentationProvenanceManifest(TypedDict):
    run_id: str
    backend_id: str
    strategy: str
    config_sha256: str
    input_image_sha256: str
    device: str
    host_platform: str
    package_name: str | None
    package_version: str | None
    model_name: str | None
    model_sha256: str | None


class LabelArtifactManifest(TypedDict):
    schema_version: str
    artifact_id: str
    dataset_id: str
    field_id: str
    source_uri: str
    source_field_index: int
    image_content_sha256: str
    axes: str
    shape: list[int]
    dtype: str
    spacing_um_zyx: list[float]
    origin_um_zyx: list[float]
    labels: list[LabelEntryManifest]
    array_sha256: str
    created_utc: str
    input_image_artifact_id: str | None
    input_image_sha256: str
    segmentation_provenance: SegmentationProvenanceManifest


_IMAGE_REQUIRED_KEYS = frozenset(ImageArtifactManifest.__annotations__)
_LABEL_REQUIRED_KEYS = frozenset(LabelArtifactManifest.__annotations__)


# ─── canonical serialisation / hashing ─────────────────────────────────────


def canonical_json(manifest: Mapping[str, Any]) -> str:
    """Stable JSON text for a manifest: sorted keys, fixed formatting."""
    return json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=True)


def sha256_of_array(arr: np.ndarray) -> str:
    """Content hash of an array's bytes, independent of memory layout."""
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _require_mapping(data: Any, where: str) -> None:
    if not isinstance(data, Mapping):
        raise ArtifactError(f"{where}: expected a JSON object, got {type(data).__name__}")


# ─── image manifests ───────────────────────────────────────────────────────


def build_image_manifest(
    image: ImageVolume,
    *,
    artifact_id: str,
    array_sha256: str,
    created_utc: str,
) -> ImageArtifactManifest:
    channels: list[ChannelManifestEntry] = [
        {
            "channel_id": c.channel_id,
            "source_index": c.source_index,
            "roles": sorted(r.value for r in c.roles),
            "source_name": c.source_name,
            "color": c.color,
            "population": c.population,
            "protein": c.protein,
        }
        for c in image.channels
    ]
    manifest: ImageArtifactManifest = {
        "schema_version": SCHEMA_VERSION_IMAGE,
        "artifact_id": artifact_id,
        "dataset_id": image.identity.dataset_id,
        "field_id": image.identity.field_id,
        "source_uri": image.identity.source_uri,
        "source_field_index": image.identity.source_field_index,
        "image_content_sha256": image.identity.image_content_sha256,
        "axes": image.axes,
        "shape": list(image.data.shape),
        "dtype": str(image.data.dtype),
        "spacing_um_zyx": list(image.geometry.spacing_um_zyx),
        "origin_um_zyx": list(image.geometry.origin_um_zyx),
        "channels": channels,
        "array_sha256": array_sha256,
        "created_utc": created_utc,
    }
    return manifest


def validate_image_manifest(data: Mapping[str, Any]) -> ImageArtifactManifest:
    """Structural + schema validation of a raw (json-loaded) image manifest."""
    _require_mapping(data, "image manifest")
    missing = _IMAGE_REQUIRED_KEYS - set(data)
    if missing:
        raise ArtifactError(f"image manifest missing key(s): {sorted(missing)}")
    if data["schema_version"] != SCHEMA_VERSION_IMAGE:
        raise ArtifactError(
            f"image manifest schema_version {data['schema_version']!r} is not "
            f"supported by this build (expected {SCHEMA_VERSION_IMAGE!r})"
        )
    if not isinstance(data["shape"], list) or len(data["shape"]) != 4:
        raise ArtifactError(f"image manifest shape must be 4-D CZYX, got {data['shape']!r}")
    if len(data["spacing_um_zyx"]) != 3 or len(data["origin_um_zyx"]) != 3:
        raise ArtifactError("image manifest spacing_um_zyx/origin_um_zyx must have 3 elements")
    if not isinstance(data["channels"], list):
        raise ArtifactError("image manifest 'channels' must be a list")
    return data  # type: ignore[return-value]


def image_identity_from_manifest(manifest: ImageArtifactManifest) -> FieldIdentity:
    return FieldIdentity(
        dataset_id=manifest["dataset_id"],
        field_id=manifest["field_id"],
        source_uri=manifest["source_uri"],
        source_field_index=manifest["source_field_index"],
        image_content_sha256=manifest["image_content_sha256"],
    )


def image_channels_from_manifest(manifest: ImageArtifactManifest) -> tuple[ChannelBinding, ...]:
    out: list[ChannelBinding] = []
    for c in manifest["channels"]:
        try:
            roles = frozenset(ChannelRole(r) for r in c["roles"])
        except ValueError as exc:
            raise ArtifactError(
                f"image manifest channel {c.get('channel_id')!r}: {exc}"
            ) from exc
        out.append(
            ChannelBinding(
                channel_id=c["channel_id"],
                source_index=c["source_index"],
                roles=roles,
                source_name=c.get("source_name"),
                color=c.get("color"),
                population=c.get("population"),
                protein=c.get("protein"),
            )
        )
    return tuple(out)


# ─── label manifests ────────────────────────────────────────────────────────


def build_label_manifest(
    label: LabelVolume,
    *,
    artifact_id: str,
    cells_sha256: str,
    nuclei_sha256: str | None,
    created_utc: str,
    input_image_artifact_id: str | None,
) -> LabelArtifactManifest:
    labels_entries: list[LabelEntryManifest] = [
        {"name": "cells", "present": True, "array_sha256": cells_sha256},
        {
            "name": "nuclei",
            "present": label.nuclei is not None,
            "array_sha256": nuclei_sha256,
        },
    ]
    p = label.provenance
    provenance_manifest: SegmentationProvenanceManifest = {
        "run_id": p.run_id,
        "backend_id": p.backend_id,
        "strategy": p.strategy,
        "config_sha256": p.config_sha256,
        "input_image_sha256": p.input_image_sha256,
        "device": p.device,
        "host_platform": p.host_platform,
        "package_name": p.package_name,
        "package_version": p.package_version,
        "model_name": p.model_name,
        "model_sha256": p.model_sha256,
    }
    manifest: LabelArtifactManifest = {
        "schema_version": SCHEMA_VERSION_LABELS,
        "artifact_id": artifact_id,
        "dataset_id": label.identity.dataset_id,
        "field_id": label.identity.field_id,
        "source_uri": label.identity.source_uri,
        "source_field_index": label.identity.source_field_index,
        "image_content_sha256": label.identity.image_content_sha256,
        "axes": label.axes,
        "shape": list(label.cells.shape),
        "dtype": str(label.cells.dtype),
        "spacing_um_zyx": list(label.geometry.spacing_um_zyx),
        "origin_um_zyx": list(label.geometry.origin_um_zyx),
        "labels": labels_entries,
        "array_sha256": cells_sha256,
        "created_utc": created_utc,
        "input_image_artifact_id": input_image_artifact_id,
        "input_image_sha256": p.input_image_sha256,
        "segmentation_provenance": provenance_manifest,
    }
    return manifest


def validate_label_manifest(data: Mapping[str, Any]) -> LabelArtifactManifest:
    """Structural + schema validation of a raw (json-loaded) label manifest."""
    _require_mapping(data, "label manifest")
    missing = _LABEL_REQUIRED_KEYS - set(data)
    if missing:
        raise ArtifactError(f"label manifest missing key(s): {sorted(missing)}")
    if data["schema_version"] != SCHEMA_VERSION_LABELS:
        raise ArtifactError(
            f"label manifest schema_version {data['schema_version']!r} is not "
            f"supported by this build (expected {SCHEMA_VERSION_LABELS!r})"
        )
    if data["dtype"] != "uint32":
        raise ArtifactError(f"label manifest dtype must be uint32, got {data['dtype']!r}")
    if not isinstance(data["shape"], list) or len(data["shape"]) != 3:
        raise ArtifactError(f"label manifest shape must be 3-D ZYX, got {data['shape']!r}")
    if not isinstance(data["labels"], list):
        raise ArtifactError("label manifest 'labels' must be a list")
    names = {e["name"] for e in data["labels"]}
    if "cells" not in names:
        raise ArtifactError("label manifest must list a 'cells' entry")
    cells_entry = next(e for e in data["labels"] if e["name"] == "cells")
    if not cells_entry.get("present", False):
        raise ArtifactError("label manifest 'cells' entry must have present=true")
    if "segmentation_provenance" not in data or not isinstance(
        data["segmentation_provenance"], Mapping
    ):
        raise ArtifactError("label manifest must carry a 'segmentation_provenance' object")
    return data  # type: ignore[return-value]


def label_identity_from_manifest(manifest: LabelArtifactManifest) -> FieldIdentity:
    return FieldIdentity(
        dataset_id=manifest["dataset_id"],
        field_id=manifest["field_id"],
        source_uri=manifest["source_uri"],
        source_field_index=manifest["source_field_index"],
        image_content_sha256=manifest["image_content_sha256"],
    )


def provenance_from_manifest(manifest: LabelArtifactManifest) -> SegmentationProvenance:
    p = manifest["segmentation_provenance"]
    return SegmentationProvenance(
        run_id=p["run_id"],
        backend_id=p["backend_id"],
        strategy=p["strategy"],
        config_sha256=p["config_sha256"],
        input_image_sha256=p["input_image_sha256"],
        device=p["device"],
        host_platform=p["host_platform"],
        package_name=p.get("package_name"),
        package_version=p.get("package_version"),
        model_name=p.get("model_name"),
        model_sha256=p.get("model_sha256"),
    )


def check_label_image_binding(label: LabelVolume, image: ImageVolume) -> None:
    """Guard before any artifact-boundary measurement.

    Raises :class:`~s2_adhesion.errors.ArtifactBindingError` -- distinct from
    the in-memory :class:`~s2_adhesion.errors.ContractViolation` that
    :meth:`LabelVolume.validate_against` raises -- so callers reading labels
    and an image back from disk can tell "wrong artifact pairing" apart from
    an in-process programming error. Checks shape, spacing, and image content
    hash independently so a caller can see exactly which one failed.
    """
    if label.shape_zyx != image.shape_zyx:
        raise ArtifactBindingError(
            f"label shape {label.shape_zyx} != image shape {image.shape_zyx}"
        )
    if not np.allclose(
        label.geometry.spacing_um_zyx, image.geometry.spacing_um_zyx, rtol=1e-6
    ):
        raise ArtifactBindingError(
            f"spacing mismatch: labels {label.geometry.spacing_um_zyx} vs "
            f"image {image.geometry.spacing_um_zyx}"
        )
    if label.provenance.input_image_sha256 != image.identity.image_content_sha256:
        raise ArtifactBindingError(
            "these labels were produced from a different image (labels claim "
            f"input image {label.provenance.input_image_sha256[:12]}..., image "
            f"artifact is {image.identity.image_content_sha256[:12]}...)"
        )
