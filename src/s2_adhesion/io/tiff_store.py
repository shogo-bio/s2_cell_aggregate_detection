"""OME-TIFF interchange format for label volumes.

TIFF is the interchange format for handing a label volume to tools outside
this pipeline (Fiji, napari, ad-hoc scripts). It is NOT sufficient on its
own: TIFF resolution tags cannot carry an anisotropic z spacing, an origin,
non-consecutive label identity, or a content hash, so spacing inferred from
TIFF tags alone is never trusted. Every OME-TIFF label artifact written here
carries a MANDATORY JSON sidecar, ``<stem>.labels.manifest.json``, that is
authoritative for every physical/identity field; a TIFF found without its
sidecar is rejected outright rather than falling back to whatever the TIFF
tags happen to say.

Layout, for a TIFF written to ``<field_id>.ome.tiff``::

    <field_id>.ome.tiff                    OME-TIFF, series "cells" (ZYX, uint32),
                                            plus optional series "nuclei"
    <field_id>.labels.manifest.json        LabelArtifactManifest, canonical JSON

The two files are swapped into place with the TIFF written first and the
sidecar written last, so an interrupted write is always observable as
"sidecar missing" -- the one rejection path every reader already takes --
rather than a TIFF paired with a stale or absent sidecar of unclear origin.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tifffile

from ..contracts import ImageVolume, LabelVolume, VoxelGeometry
from ..errors import ArtifactError
from .manifests import (
    LabelArtifactManifest,
    build_label_manifest,
    canonical_json,
    check_label_image_binding,
    label_identity_from_manifest,
    provenance_from_manifest,
    sha256_of_array,
    validate_label_manifest,
)

__all__ = ["sidecar_path_for", "write_label_tiff", "read_label_tiff"]

_SERIES_CELLS = "cells"
_SERIES_NUCLEI = "nuclei"
_TIFF_SUFFIXES = (".ome.tiff", ".ome.tif", ".tiff", ".tif")


def sidecar_path_for(tiff_path: Path | str) -> Path:
    """The mandatory sidecar path for a TIFF path.

    ``field00.ome.tiff`` -> ``field00.labels.manifest.json``. Name TIFF
    artifacts after their ``field_id`` so this matches the
    ``<field_id>.labels.manifest.json`` convention exactly.
    """
    tiff_path = Path(tiff_path)
    name = tiff_path.name
    stem = tiff_path.stem
    for suffix in _TIFF_SUFFIXES:
        if name.lower().endswith(suffix):
            stem = name[: -len(suffix)]
            break
    return tiff_path.with_name(f"{stem}.labels.manifest.json")


def write_label_tiff(
    lab: LabelVolume,
    path: Path | str,
    input_image_artifact_id: str | None = None,
) -> Path:
    """Write ``lab`` as an OME-TIFF ZYX uint32 volume plus its mandatory sidecar."""
    if lab.cells.dtype != np.uint32:
        raise ArtifactError(
            f"labels must be uint32 to write, got {lab.cells.dtype}. Downcasting "
            "would silently merge instances."
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
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

    sidecar = sidecar_path_for(path)
    tiff_tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    sidecar_tmp = sidecar.with_name(f".{sidecar.name}.tmp-{os.getpid()}")
    try:
        with tifffile.TiffWriter(tiff_tmp, ome=True) as tw:
            tw.write(
                cells,
                photometric="minisblack",
                metadata={"axes": "ZYX", "Name": _SERIES_CELLS},
            )
            if nuclei is not None:
                tw.write(
                    nuclei,
                    photometric="minisblack",
                    metadata={"axes": "ZYX", "Name": _SERIES_NUCLEI},
                )

        manifest = build_label_manifest(
            lab,
            artifact_id=f"lab-{cells_sha256[:16]}",
            cells_sha256=cells_sha256,
            nuclei_sha256=nuclei_sha256,
            created_utc=datetime.now(timezone.utc).isoformat(),
            input_image_artifact_id=input_image_artifact_id,
        )
        sidecar_tmp.write_text(canonical_json(manifest), encoding="utf-8")

        # TIFF first, sidecar last: see module docstring.
        os.replace(tiff_tmp, path)
        os.replace(sidecar_tmp, sidecar)
    except Exception:
        for tmp in (tiff_tmp, sidecar_tmp):
            if tmp.exists():
                tmp.unlink(missing_ok=True)
        raise
    return path


def read_label_tiff(
    path: Path | str,
    verify_hashes: bool = True,
    image: ImageVolume | None = None,
) -> LabelVolume:
    """Read an OME-TIFF label artifact back into a :class:`LabelVolume`.

    Rejects (``ArtifactError``) a TIFF with no sidecar, a sidecar that fails
    schema validation, or -- when ``verify_hashes`` -- a series whose content
    hash no longer matches the sidecar. Pass ``image`` to additionally check
    the labels are bound to that exact image; a mismatch raises
    :class:`~s2_adhesion.errors.ArtifactBindingError`.
    """
    path = Path(path)
    if not path.exists():
        raise ArtifactError(f"no TIFF at {path}")
    sidecar = sidecar_path_for(path)
    if not sidecar.exists():
        raise ArtifactError(
            f"{path} has no sidecar manifest at {sidecar}; spacing, origin, "
            "field identity and segmentation provenance cannot be inferred "
            "from TIFF tags alone, so this artifact is rejected"
        )
    try:
        raw_manifest = json.loads(sidecar.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"{sidecar}: invalid JSON: {exc}") from exc
    manifest: LabelArtifactManifest = validate_label_manifest(raw_manifest)

    with tifffile.TiffFile(path) as tf:
        series_by_name = {s.name: s for s in tf.series}
        if _SERIES_CELLS not in series_by_name:
            raise ArtifactError(f"{path}: no {_SERIES_CELLS!r} series found")
        cells = np.asarray(series_by_name[_SERIES_CELLS].asarray())
        nuclei = None
        if _SERIES_NUCLEI in series_by_name:
            nuclei = np.asarray(series_by_name[_SERIES_NUCLEI].asarray())

    if cells.dtype != np.uint32:
        raise ArtifactError(f"{path}: cells series is {cells.dtype}, expected uint32")
    if cells.ndim != 3:
        raise ArtifactError(f"{path}: cells series must be 3-D ZYX, got shape {cells.shape}")
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
        if nuclei is not None:
            nuclei_entry = next(
                (e for e in manifest["labels"] if e["name"] == "nuclei"), None
            )
            if nuclei_entry is not None and nuclei_entry.get("array_sha256"):
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
