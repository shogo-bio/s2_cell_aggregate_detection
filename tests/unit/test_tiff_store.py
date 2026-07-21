"""Tests for s2_adhesion.io.tiff_store: the OME-TIFF interchange format.

The TIFF format on its own cannot carry anisotropic spacing, an origin, or
provenance -- the mandatory JSON sidecar is what makes the artifact
trustworthy, and these tests are mostly about that boundary.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest
import tifffile

from s2_adhesion.contracts import (
    ChannelBinding,
    ChannelRole,
    ImageVolume,
    LabelVolume,
    VoxelGeometry,
)
from s2_adhesion.errors import ArtifactBindingError, ArtifactError
from s2_adhesion.io import tiff_store

from tests.conftest import ANISOTROPIC, make_identity, make_label_volume, make_provenance


# ─── round trip ─────────────────────────────────────────────────────────────


def test_label_tiff_round_trip_preserves_nonconsecutive_ids_and_spacing(tmp_path):
    cells = np.zeros((3, 5, 6), dtype=np.uint32)
    cells[0, 0, 0] = 1
    cells[1, 1, 1] = 5
    cells[2, 2, 2] = 900
    lab = make_label_volume(cells, spacing=ANISOTROPIC)

    out = tiff_store.write_label_tiff(lab, tmp_path / "field00.ome.tiff")
    back = tiff_store.read_label_tiff(out)

    assert np.array_equal(back.cells, cells)
    assert sorted(back.cell_ids().tolist()) == [1, 5, 900]
    assert back.cells.dtype == np.uint32
    assert back.geometry.spacing_um_zyx == ANISOTROPIC
    assert back.identity == lab.identity
    assert back.provenance == lab.provenance


def test_label_tiff_round_trip_with_nuclei(tmp_path):
    cells = np.zeros((2, 4, 4), dtype=np.uint32)
    cells[0, 0, 0] = 3
    nuclei = np.zeros((2, 4, 4), dtype=np.uint32)
    nuclei[0, 0, 0] = 3
    lab = make_label_volume(cells, spacing=ANISOTROPIC, nuclei=nuclei)

    out = tiff_store.write_label_tiff(lab, tmp_path / "field01.ome.tiff")
    back = tiff_store.read_label_tiff(out)

    assert back.nuclei is not None
    assert np.array_equal(back.nuclei, nuclei)
    assert back.nuclei.dtype == np.uint32


def test_label_tiff_writes_mandatory_sidecar_next_to_tiff(tmp_path):
    cells = np.zeros((2, 3, 3), dtype=np.uint32)
    lab = make_label_volume(cells, spacing=ANISOTROPIC)
    out = tiff_store.write_label_tiff(lab, tmp_path / "field02.ome.tiff")
    sidecar = tiff_store.sidecar_path_for(out)
    assert sidecar.name == "field02.labels.manifest.json"
    assert sidecar.exists()
    manifest = json.loads(sidecar.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "s2-label-volume/v1"
    assert manifest["dtype"] == "uint32"


# ─── dtype enforcement ──────────────────────────────────────────────────────


def test_write_label_tiff_rejects_non_uint32_defence_in_depth(tmp_path):
    cells = np.zeros((2, 2, 2), dtype=np.uint32)
    lab = make_label_volume(cells, spacing=ANISOTROPIC)
    object.__setattr__(lab, "cells", cells.astype(np.uint16))
    with pytest.raises(ArtifactError):
        tiff_store.write_label_tiff(lab, tmp_path / "field.ome.tiff")


def test_write_label_tiff_rejects_non_uint32_nuclei(tmp_path):
    cells = np.zeros((2, 2, 2), dtype=np.uint32)
    nuclei = np.zeros((2, 2, 2), dtype=np.uint32)
    lab = make_label_volume(cells, spacing=ANISOTROPIC, nuclei=nuclei)
    object.__setattr__(lab, "nuclei", nuclei.astype(np.int16))
    with pytest.raises(ArtifactError):
        tiff_store.write_label_tiff(lab, tmp_path / "field.ome.tiff")


# ─── mandatory sidecar ──────────────────────────────────────────────────────


def test_read_label_tiff_rejects_missing_sidecar(tmp_path):
    cells = np.zeros((2, 3, 3), dtype=np.uint32)
    lab = make_label_volume(cells, spacing=ANISOTROPIC)
    out = tiff_store.write_label_tiff(lab, tmp_path / "field03.ome.tiff")
    tiff_store.sidecar_path_for(out).unlink()

    with pytest.raises(ArtifactError, match="sidecar"):
        tiff_store.read_label_tiff(out)


def test_read_label_tiff_rejects_a_bare_tiff_with_no_sidecar_ever_written(tmp_path):
    """A TIFF written by something other than this module (no sidecar
    convention at all) must still be rejected, not silently measured using
    whatever TIFF tags happen to be present."""
    path = tmp_path / "external.ome.tiff"
    arr = np.zeros((2, 3, 3), dtype=np.uint32)
    tifffile.imwrite(path, arr, photometric="minisblack", metadata={"axes": "ZYX"}, ome=True)
    with pytest.raises(ArtifactError, match="sidecar"):
        tiff_store.read_label_tiff(path)


# ─── corruption detection ───────────────────────────────────────────────────


def test_read_label_tiff_rejects_corrupted_array_when_verifying(tmp_path):
    cells = np.zeros((3, 4, 4), dtype=np.uint32)
    cells[1, 1, 1] = 42
    lab = make_label_volume(cells, spacing=ANISOTROPIC)
    out = tiff_store.write_label_tiff(lab, tmp_path / "field04.ome.tiff")

    corrupted = cells.copy()
    corrupted[0, 0, 0] = 999999
    tifffile.imwrite(
        out, corrupted, photometric="minisblack", metadata={"axes": "ZYX", "Name": "cells"}, ome=True
    )

    with pytest.raises(ArtifactError, match="hash mismatch|corrupted"):
        tiff_store.read_label_tiff(out, verify_hashes=True)

    back = tiff_store.read_label_tiff(out, verify_hashes=False)
    assert back.cells[0, 0, 0] == 999999


# ─── binding checks ─────────────────────────────────────────────────────────


def _write_bound_tiff(tmp_path, shape_zyx=(3, 4, 4), spacing=ANISOTROPIC):
    image_identity = make_identity(field_id="field05", content=b"source-image-bytes")
    image = ImageVolume(
        data=np.zeros((1, *shape_zyx), dtype=np.uint16),
        geometry=VoxelGeometry(spacing_um_zyx=spacing),
        channels=(ChannelBinding(channel_id="membrane", source_index=0, roles=frozenset({ChannelRole.MEMBRANE})),),
        identity=image_identity,
    )
    cells = np.zeros(shape_zyx, dtype=np.uint32)
    cells[0, 0, 0] = 1
    provenance = make_provenance(image_identity.image_content_sha256)
    label = LabelVolume(
        cells=cells,
        geometry=VoxelGeometry(spacing_um_zyx=spacing),
        identity=make_identity(field_id="field05", content=cells.tobytes()),
        provenance=provenance,
    )
    out = tiff_store.write_label_tiff(label, tmp_path / "field05.ome.tiff")
    return out, image


def test_read_label_tiff_with_matching_image_does_not_raise(tmp_path):
    out, image = _write_bound_tiff(tmp_path)
    back = tiff_store.read_label_tiff(out, image=image)
    assert back is not None


def test_read_label_tiff_rejects_shape_mismatch_against_image(tmp_path):
    out, image = _write_bound_tiff(tmp_path, shape_zyx=(3, 4, 4))
    mismatched = replace(image, data=np.zeros((1, 3, 4, 9), dtype=np.uint16))
    with pytest.raises(ArtifactBindingError, match="shape"):
        tiff_store.read_label_tiff(out, image=mismatched)


def test_read_label_tiff_rejects_spacing_mismatch_against_image(tmp_path):
    out, image = _write_bound_tiff(tmp_path, spacing=ANISOTROPIC)
    mismatched = replace(image, geometry=VoxelGeometry(spacing_um_zyx=(0.1, 0.1, 0.1)))
    with pytest.raises(ArtifactBindingError, match="spacing"):
        tiff_store.read_label_tiff(out, image=mismatched)


def test_read_label_tiff_rejects_image_hash_mismatch(tmp_path):
    out, image = _write_bound_tiff(tmp_path)
    mismatched = replace(image, identity=replace(image.identity, image_content_sha256="0" * 64))
    with pytest.raises(ArtifactBindingError, match="different image"):
        tiff_store.read_label_tiff(out, image=mismatched)


# ─── no-ML import guarantee ──────────────────────────────────────────────────


def test_tiff_store_importable_without_torch_or_cellpose():
    script = (
        "import sys\n"
        "from s2_adhesion.io import tiff_store\n"
        "assert 'torch' not in sys.modules, sys.modules.keys()\n"
        "assert 'cellpose' not in sys.modules, sys.modules.keys()\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
