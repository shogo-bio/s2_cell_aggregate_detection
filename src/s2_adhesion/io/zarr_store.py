"""zarr v3 artifact store -- the PRIMARY on-disk format for this pipeline.

Segmentation is far too slow to run on a laptop (cellpose 4 needed 638 s for
one 256x256 plane on the reference host; cellpose 3 is roughly 290x faster
but still slow), so a label volume is routinely written on a GPU machine and
read on a machine with no torch/cellpose installed at all. This module is
that boundary: writers never mutate a path in place, and readers reject
anything that is not provably complete or provably intact rather than
silently measuring a half-transferred or corrupted artifact.

Layout::

    image.ome.zarr/
        0                 array, CZYX, any dtype
        manifest.json     ImageArtifactManifest, canonical JSON
        _SUCCESS          empty marker file, written last

    labels.ome.zarr/
        labels/cells/0    array, ZYX, uint32
        labels/nuclei/0   optional, same shape, uint32
        manifest.json     LabelArtifactManifest, canonical JSON
        _SUCCESS          empty marker file, written last

Atomic write: every writer builds the full artifact (arrays, manifest,
``_SUCCESS``) in a temporary sibling directory and only then swaps it into
place. On POSIX this would be one atomic ``rename``; Windows cannot
atomically replace a non-empty directory the same way, so :func:`_replace_dir`
moves any existing artifact aside first and only deletes it once the new one
is confirmed in place. Either way, the final path is only ever fully absent,
holding the complete previous artifact, or holding the complete new one --
never a partial mix. Readers additionally refuse anything missing
``_SUCCESS``, so even a copy tool that does not preserve atomicity (e.g. an
interrupted network transfer) cannot produce a silently-measured artifact.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import zarr
from zarr.codecs import BloscCodec

from ..contracts import ImageVolume, LabelVolume, VoxelGeometry
from ..errors import ArtifactError
from .manifests import (
    build_image_manifest,
    build_label_manifest,
    canonical_json,
    check_label_image_binding,
    image_channels_from_manifest,
    image_identity_from_manifest,
    label_identity_from_manifest,
    provenance_from_manifest,
    sha256_of_array,
    validate_image_manifest,
    validate_label_manifest,
)

__all__ = [
    "write_image_volume",
    "read_image_volume",
    "write_label_volume",
    "read_label_volume",
]

_SUCCESS_NAME = "_SUCCESS"
_MANIFEST_NAME = "manifest.json"


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _compressor(compression_level: int) -> BloscCodec:
    return BloscCodec(cname="zstd", clevel=compression_level)


def _replace_dir(src: Path, dst: Path) -> None:
    """Swap ``src`` into ``dst``. See module docstring for the Windows caveat."""
    if not dst.exists():
        os.replace(src, dst)
        return
    backup = dst.parent / f".{dst.name}.bak-{uuid.uuid4().hex}"
    os.replace(dst, backup)
    try:
        os.replace(src, dst)
    except Exception:
        os.replace(backup, dst)
        raise
    shutil.rmtree(backup, ignore_errors=True)


def _write_atomic(final_path: Path, populate: Callable[[Path], None]) -> Path:
    """Build a complete artifact in a temp dir, then swap it into place.

    ``populate`` must write every array and ``manifest.json`` into the
    directory it is given; ``_SUCCESS`` is added here, last, after
    ``populate`` returns without error.
    """
    final_path = Path(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = final_path.parent / f".{final_path.name}.tmp-{uuid.uuid4().hex}"
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_path.mkdir(parents=True)
    try:
        populate(tmp_path)
        (tmp_path / _SUCCESS_NAME).write_text("", encoding="utf-8")
        _replace_dir(tmp_path, final_path)
    except Exception:
        shutil.rmtree(tmp_path, ignore_errors=True)
        raise
    return final_path


def _check_success(path: Path) -> None:
    if not path.exists():
        raise ArtifactError(f"no artifact at {path}")
    if not (path / _SUCCESS_NAME).exists():
        raise ArtifactError(
            f"{path} is missing {_SUCCESS_NAME!r}; this artifact was never "
            "completed (or was only partially transferred) and must not be "
            "measured"
        )


def _read_manifest(path: Path) -> dict:
    manifest_path = path / _MANIFEST_NAME
    if not manifest_path.exists():
        raise ArtifactError(f"{path} has no {_MANIFEST_NAME}")
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"{manifest_path}: invalid JSON: {exc}") from exc


# ─── image volumes ──────────────────────────────────────────────────────────


def write_image_volume(
    vol: ImageVolume,
    path: Path | str,
    chunks: tuple[int, int, int, int],
    compression_level: int = 3,
) -> Path:
    """Write ``vol`` as ``image.ome.zarr``-style artifact at ``path``."""
    final_path = Path(path)
    data = np.ascontiguousarray(vol.data)
    array_sha256 = sha256_of_array(data)

    def populate(tmp_path: Path) -> None:
        root = zarr.open_group(tmp_path, mode="w")
        root.create_array(
            "0",
            chunks=chunks,
            compressors=_compressor(compression_level),
            data=data,
        )
        manifest = build_image_manifest(
            vol,
            artifact_id=f"img-{array_sha256[:16]}",
            array_sha256=array_sha256,
            created_utc=_now_utc(),
        )
        (tmp_path / _MANIFEST_NAME).write_text(canonical_json(manifest), encoding="utf-8")

    return _write_atomic(final_path, populate)


def read_image_volume(path: Path | str, verify_hashes: bool = True) -> ImageVolume:
    """Read an ``image.ome.zarr``-style artifact back into an :class:`ImageVolume`.

    Rejects (``ArtifactError``) an artifact missing ``_SUCCESS``, a manifest
    that fails schema validation, or -- when ``verify_hashes`` -- an array
    whose content hash no longer matches the manifest.
    """
    path = Path(path)
    _check_success(path)
    manifest = validate_image_manifest(_read_manifest(path))

    root = zarr.open_group(path, mode="r")
    if "0" not in root:
        raise ArtifactError(f"{path}: no array '0' in image artifact")
    arr = np.asarray(root["0"][:])

    if list(arr.shape) != manifest["shape"]:
        raise ArtifactError(
            f"{path}: array shape {arr.shape} != manifest shape {manifest['shape']}"
        )
    if str(arr.dtype) != manifest["dtype"]:
        raise ArtifactError(
            f"{path}: array dtype {arr.dtype} != manifest dtype {manifest['dtype']}"
        )
    if verify_hashes:
        actual = sha256_of_array(arr)
        if actual != manifest["array_sha256"]:
            raise ArtifactError(
                f"{path}: content hash mismatch (manifest says "
                f"{manifest['array_sha256'][:12]}..., array is {actual[:12]}...) "
                "-- artifact is corrupted"
            )

    geometry = VoxelGeometry(
        spacing_um_zyx=tuple(manifest["spacing_um_zyx"]),
        origin_um_zyx=tuple(manifest["origin_um_zyx"]),
    )
    identity = image_identity_from_manifest(manifest)
    channels = image_channels_from_manifest(manifest)
    return ImageVolume(
        data=arr,
        geometry=geometry,
        channels=channels,
        identity=identity,
        axes=manifest["axes"],
    )


# ─── label volumes ──────────────────────────────────────────────────────────


def write_label_volume(
    lab: LabelVolume,
    path: Path | str,
    chunks: tuple[int, int, int],
    compression_level: int = 3,
    input_image_artifact_id: str | None = None,
) -> Path:
    """Write ``lab`` as a ``labels.ome.zarr``-style artifact at ``path``.

    ``lab.cells`` (and ``lab.nuclei`` if present) must be ``uint32`` --
    checked here even though :class:`~s2_adhesion.contracts.LabelVolume`
    already enforces it at construction, as defence in depth against a
    caller that mutated a frozen instance's array in place.
    """
    if lab.cells.dtype != np.uint32:
        raise ArtifactError(
            f"labels must be uint32 to write, got {lab.cells.dtype}. Downcasting "
            "would silently merge instances."
        )
    final_path = Path(path)
    cells = np.ascontiguousarray(lab.cells)
    cells_sha256 = sha256_of_array(cells)
    nuclei: np.ndarray | None = None
    nuclei_sha256: str | None = None
    if lab.nuclei is not None:
        if lab.nuclei.dtype != np.uint32:
            raise ArtifactError(
                f"nuclei labels must be uint32 to write, got {lab.nuclei.dtype}"
            )
        nuclei = np.ascontiguousarray(lab.nuclei)
        nuclei_sha256 = sha256_of_array(nuclei)

    def populate(tmp_path: Path) -> None:
        root = zarr.open_group(tmp_path, mode="w")
        labels_group = root.create_group("labels")
        cells_group = labels_group.create_group("cells")
        cells_group.create_array(
            "0", chunks=chunks, compressors=_compressor(compression_level), data=cells
        )
        if nuclei is not None:
            nuclei_group = labels_group.create_group("nuclei")
            nuclei_group.create_array(
                "0", chunks=chunks, compressors=_compressor(compression_level), data=nuclei
            )
        manifest = build_label_manifest(
            lab,
            artifact_id=f"lab-{cells_sha256[:16]}",
            cells_sha256=cells_sha256,
            nuclei_sha256=nuclei_sha256,
            created_utc=_now_utc(),
            input_image_artifact_id=input_image_artifact_id,
        )
        (tmp_path / _MANIFEST_NAME).write_text(canonical_json(manifest), encoding="utf-8")

    return _write_atomic(final_path, populate)


def read_label_volume(
    path: Path | str,
    verify_hashes: bool = True,
    image: ImageVolume | None = None,
) -> LabelVolume:
    """Read a ``labels.ome.zarr``-style artifact back into a :class:`LabelVolume`.

    Label ids are never renumbered: whatever was written comes back exactly.
    Pass ``image`` to additionally check the labels are bound to that exact
    image (shape, spacing, and image content hash); a mismatch on any of
    those raises :class:`~s2_adhesion.errors.ArtifactBindingError`.
    """
    path = Path(path)
    _check_success(path)
    manifest = validate_label_manifest(_read_manifest(path))

    root = zarr.open_group(path, mode="r")
    if (
        "labels" not in root
        or "cells" not in root["labels"]
        or "0" not in root["labels"]["cells"]
    ):
        raise ArtifactError(f"{path}: no labels/cells/0 array in label artifact")
    cells = np.asarray(root["labels"]["cells"]["0"][:])

    nuclei: np.ndarray | None = None
    nuclei_entry = next((e for e in manifest["labels"] if e["name"] == "nuclei"), None)
    has_nuclei_array = "nuclei" in root["labels"] and "0" in root["labels"]["nuclei"]
    if nuclei_entry is not None and nuclei_entry.get("present"):
        if not has_nuclei_array:
            raise ArtifactError(
                f"{path}: manifest declares nuclei present but the array is missing"
            )
        nuclei = np.asarray(root["labels"]["nuclei"]["0"][:])

    if cells.dtype != np.uint32:
        raise ArtifactError(f"{path}: cells array is {cells.dtype}, expected uint32")
    if list(cells.shape) != manifest["shape"]:
        raise ArtifactError(
            f"{path}: cells shape {cells.shape} != manifest shape {manifest['shape']}"
        )

    if verify_hashes:
        cells_entry = next(e for e in manifest["labels"] if e["name"] == "cells")
        actual = sha256_of_array(cells)
        if actual != cells_entry["array_sha256"]:
            raise ArtifactError(
                f"{path}: cells content hash mismatch (manifest says "
                f"{cells_entry['array_sha256'][:12]}..., array is {actual[:12]}...) "
                "-- artifact is corrupted"
            )
        if nuclei is not None and nuclei_entry is not None and nuclei_entry.get("array_sha256"):
            actual_n = sha256_of_array(nuclei)
            if actual_n != nuclei_entry["array_sha256"]:
                raise ArtifactError(
                    f"{path}: nuclei content hash mismatch -- artifact is corrupted"
                )

    geometry = VoxelGeometry(
        spacing_um_zyx=tuple(manifest["spacing_um_zyx"]),
        origin_um_zyx=tuple(manifest["origin_um_zyx"]),
    )
    identity = label_identity_from_manifest(manifest)
    provenance = provenance_from_manifest(manifest)
    lab = LabelVolume(
        cells=cells.astype(np.uint32, copy=False),
        geometry=geometry,
        identity=identity,
        provenance=provenance,
        nuclei=None if nuclei is None else nuclei.astype(np.uint32, copy=False),
        axes=manifest["axes"],
    )
    if image is not None:
        check_label_image_binding(lab, image)
    return lab
