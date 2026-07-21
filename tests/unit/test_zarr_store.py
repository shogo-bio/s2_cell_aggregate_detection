"""Tests for s2_adhesion.io.zarr_store: the primary artifact format.

These exercise the actual acceptance boundary of the io layer -- a label
volume written on one machine must be exactly reconstructable, or cleanly
rejected, on another with no ML stack installed.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import zarr

from s2_adhesion.contracts import (
    ChannelBinding,
    ChannelRole,
    ContractViolation,
    ImageVolume,
    LabelVolume,
    VoxelGeometry,
)
from s2_adhesion.errors import ArtifactBindingError, ArtifactError
from s2_adhesion.io import zarr_store

from tests.conftest import ANISOTROPIC, make_identity, make_image_volume, make_label_volume, make_provenance


# ─── round trip ─────────────────────────────────────────────────────────────


def test_image_round_trip_preserves_anisotropic_spacing_and_dtype(tmp_path):
    rng = np.random.default_rng(0)
    data = (rng.random((3, 5, 6, 7)) * 1000).astype(np.uint16)
    vol = make_image_volume(data, spacing=ANISOTROPIC)

    out = zarr_store.write_image_volume(
        vol, tmp_path / "image.ome.zarr", chunks=(1, 2, 4, 4), compression_level=3
    )
    back = zarr_store.read_image_volume(out)

    assert np.array_equal(back.data, data)
    assert back.data.dtype == data.dtype
    assert back.geometry.spacing_um_zyx == ANISOTROPIC
    assert back.geometry.origin_um_zyx == (0.0, 0.0, 0.0)
    assert back.channels == vol.channels
    assert back.identity == vol.identity
    assert back.axes == "CZYX"


def test_image_round_trip_preserves_nondefault_origin(tmp_path):
    data = np.zeros((2, 3, 4, 4), dtype=np.uint8)
    from s2_adhesion.contracts import ChannelBinding, ChannelRole, FieldIdentity

    geometry = VoxelGeometry(spacing_um_zyx=ANISOTROPIC, origin_um_zyx=(10.0, -3.5, 2.25))
    vol = ImageVolume(
        data=data,
        geometry=geometry,
        channels=(
            ChannelBinding(channel_id="membrane", source_index=0, roles=frozenset({ChannelRole.MEMBRANE})),
            ChannelBinding(channel_id="signal", source_index=1, roles=frozenset({ChannelRole.SIGNAL})),
        ),
        identity=make_identity(content=data.tobytes()),
    )
    out = zarr_store.write_image_volume(vol, tmp_path / "image.ome.zarr", chunks=(1, 2, 4, 4))
    back = zarr_store.read_image_volume(out)
    assert back.geometry.origin_um_zyx == (10.0, -3.5, 2.25)


def test_label_round_trip_preserves_nonconsecutive_ids_unrenumbered(tmp_path):
    cells = np.zeros((3, 5, 6), dtype=np.uint32)
    cells[0, 0, 0] = 1
    cells[1, 1, 1] = 5
    cells[2, 2, 2] = 900
    lab = make_label_volume(cells, spacing=ANISOTROPIC)

    out = zarr_store.write_label_volume(
        lab, tmp_path / "labels.ome.zarr", chunks=(2, 4, 4), compression_level=3
    )
    back = zarr_store.read_label_volume(out)

    assert np.array_equal(back.cells, cells)
    assert sorted(back.cell_ids().tolist()) == [1, 5, 900]
    assert back.geometry.spacing_um_zyx == ANISOTROPIC
    assert back.identity == lab.identity


def test_label_round_trip_with_nuclei_and_bindings(tmp_path):
    cells = np.zeros((3, 5, 6), dtype=np.uint32)
    cells[0, 0, 0] = 1
    cells[1, 1, 1] = 5
    nuclei = np.zeros((3, 5, 6), dtype=np.uint32)
    nuclei[0, 0, 0] = 1
    lab = make_label_volume(cells, spacing=ANISOTROPIC, nuclei=nuclei)

    out = zarr_store.write_label_volume(lab, tmp_path / "labels.ome.zarr", chunks=(2, 4, 4))
    back = zarr_store.read_label_volume(out)

    assert back.nuclei is not None
    assert np.array_equal(back.nuclei, nuclei)
    assert back.nuclei.dtype == np.uint32


def test_label_without_nuclei_round_trips_with_nuclei_none(tmp_path):
    cells = np.zeros((2, 3, 3), dtype=np.uint32)
    cells[0, 0, 0] = 7
    lab = make_label_volume(cells, spacing=ANISOTROPIC)
    out = zarr_store.write_label_volume(lab, tmp_path / "labels.ome.zarr", chunks=(2, 3, 3))
    back = zarr_store.read_label_volume(out)
    assert back.nuclei is None


# ─── dtype enforcement ──────────────────────────────────────────────────────


def test_label_volume_rejects_non_uint32_at_construction():
    """The frozen contract itself refuses non-uint32 label arrays."""
    with pytest.raises(ContractViolation):
        LabelVolume(
            cells=np.zeros((2, 2, 2), dtype=np.uint16),
            geometry=VoxelGeometry(spacing_um_zyx=ANISOTROPIC),
            identity=make_identity(content=b"x"),
            provenance=make_provenance(make_identity(content=b"x").image_content_sha256),
        )


def test_write_label_volume_rejects_non_uint32_defence_in_depth(tmp_path):
    """Even if a caller mutates a frozen LabelVolume's array in place, the
    writer itself refuses to persist non-uint32 labels."""
    cells = np.zeros((2, 2, 2), dtype=np.uint32)
    lab = make_label_volume(cells, spacing=ANISOTROPIC)
    object.__setattr__(lab, "cells", cells.astype(np.uint16))
    with pytest.raises(ArtifactError):
        zarr_store.write_label_volume(lab, tmp_path / "labels.ome.zarr", chunks=(2, 2, 2))


def test_write_label_volume_rejects_non_uint32_nuclei(tmp_path):
    cells = np.zeros((2, 2, 2), dtype=np.uint32)
    nuclei = np.zeros((2, 2, 2), dtype=np.uint32)
    lab = make_label_volume(cells, spacing=ANISOTROPIC, nuclei=nuclei)
    object.__setattr__(lab, "nuclei", nuclei.astype(np.int32))
    with pytest.raises(ArtifactError):
        zarr_store.write_label_volume(lab, tmp_path / "labels.ome.zarr", chunks=(2, 2, 2))


# ─── _SUCCESS gate ──────────────────────────────────────────────────────────


def test_read_image_volume_rejects_missing_success(tmp_path):
    data = np.zeros((1, 2, 3, 3), dtype=np.uint8)
    vol = make_image_volume(
        data, spacing=ANISOTROPIC, channel_ids=("membrane",), roles=(ChannelRole.MEMBRANE,)
    )
    out = zarr_store.write_image_volume(vol, tmp_path / "image.ome.zarr", chunks=(1, 2, 3, 3))
    (out / "_SUCCESS").unlink()
    with pytest.raises(ArtifactError, match="_SUCCESS"):
        zarr_store.read_image_volume(out)


def test_read_label_volume_rejects_missing_success(tmp_path):
    cells = np.zeros((2, 2, 2), dtype=np.uint32)
    lab = make_label_volume(cells, spacing=ANISOTROPIC)
    out = zarr_store.write_label_volume(lab, tmp_path / "labels.ome.zarr", chunks=(2, 2, 2))
    (out / "_SUCCESS").unlink()
    with pytest.raises(ArtifactError, match="_SUCCESS"):
        zarr_store.read_label_volume(out)


def test_write_never_leaves_success_without_full_content(tmp_path):
    """Sanity check on the atomic-write contract: right after a successful
    write, _SUCCESS is present alongside the array and manifest."""
    cells = np.zeros((2, 2, 2), dtype=np.uint32)
    lab = make_label_volume(cells, spacing=ANISOTROPIC)
    out = zarr_store.write_label_volume(lab, tmp_path / "labels.ome.zarr", chunks=(2, 2, 2))
    assert (out / "_SUCCESS").exists()
    assert (out / "manifest.json").exists()
    assert (out / "labels" / "cells" / "0").exists()


# ─── corruption detection ───────────────────────────────────────────────────


def test_read_image_volume_rejects_corrupted_array_when_verifying(tmp_path):
    data = np.arange(2 * 3 * 4 * 4, dtype=np.uint16).reshape(2, 3, 4, 4)
    vol = make_image_volume(data, spacing=ANISOTROPIC)
    out = zarr_store.write_image_volume(vol, tmp_path / "image.ome.zarr", chunks=(1, 3, 4, 4))

    # corrupt the on-disk array directly, leaving the manifest hash stale
    g = zarr.open_group(out, mode="r+")
    g["0"][0, 0, 0, 0] = 65535

    with pytest.raises(ArtifactError, match="hash mismatch|corrupted"):
        zarr_store.read_image_volume(out, verify_hashes=True)

    # with verification off, corruption is not caught (fast path)
    back = zarr_store.read_image_volume(out, verify_hashes=False)
    assert back.data[0, 0, 0, 0] == 65535


def test_read_label_volume_rejects_corrupted_cells_when_verifying(tmp_path):
    cells = np.zeros((3, 4, 4), dtype=np.uint32)
    cells[1, 1, 1] = 5
    lab = make_label_volume(cells, spacing=ANISOTROPIC)
    out = zarr_store.write_label_volume(lab, tmp_path / "labels.ome.zarr", chunks=(3, 4, 4))

    g = zarr.open_group(out, mode="r+")
    g["labels"]["cells"]["0"][0, 0, 0] = 12345

    with pytest.raises(ArtifactError, match="hash mismatch|corrupted"):
        zarr_store.read_label_volume(out, verify_hashes=True)


# ─── binding checks ─────────────────────────────────────────────────────────


def _write_bound_pair(tmp_path, shape_zyx=(3, 4, 4), spacing=ANISOTROPIC):
    """Write an image artifact and a label artifact whose provenance matches it."""
    data = np.zeros((1, *shape_zyx), dtype=np.uint16)
    data.flat[: min(data.size, 5)] = np.arange(min(data.size, 5))

    image_identity = make_identity(field_id="field00", content=data.tobytes())
    image = ImageVolume(
        data=data,
        geometry=VoxelGeometry(spacing_um_zyx=spacing),
        channels=(ChannelBinding(channel_id="membrane", source_index=0, roles=frozenset({ChannelRole.MEMBRANE})),),
        identity=image_identity,
    )
    image_out = zarr_store.write_image_volume(image, tmp_path / "image.ome.zarr", chunks=(1, 2, 4, 4))
    image_back = zarr_store.read_image_volume(image_out)

    cells = np.zeros(shape_zyx, dtype=np.uint32)
    cells[0, 0, 0] = 1
    provenance = make_provenance(image_identity.image_content_sha256)
    label = LabelVolume(
        cells=cells,
        geometry=VoxelGeometry(spacing_um_zyx=spacing),
        identity=make_identity(field_id="field00", content=cells.tobytes()),
        provenance=provenance,
    )
    label_out = zarr_store.write_label_volume(label, tmp_path / "labels.ome.zarr", chunks=(2, 4, 4))
    return label_out, image_back


def test_read_label_volume_with_matching_image_does_not_raise(tmp_path):
    label_out, image_back = _write_bound_pair(tmp_path)
    back = zarr_store.read_label_volume(label_out, image=image_back)
    assert back is not None


def test_read_label_volume_rejects_shape_mismatch_against_image(tmp_path):
    label_out, image_back = _write_bound_pair(tmp_path, shape_zyx=(3, 4, 4))
    mismatched = replace(image_back, data=np.zeros((1, 3, 4, 9), dtype=np.uint16))
    with pytest.raises(ArtifactBindingError, match="shape"):
        zarr_store.read_label_volume(label_out, image=mismatched)


def test_read_label_volume_rejects_spacing_mismatch_against_image(tmp_path):
    label_out, image_back = _write_bound_pair(tmp_path, spacing=ANISOTROPIC)
    mismatched = replace(
        image_back, geometry=VoxelGeometry(spacing_um_zyx=(0.1, 0.1, 0.1))
    )
    with pytest.raises(ArtifactBindingError, match="spacing"):
        zarr_store.read_label_volume(label_out, image=mismatched)


def test_read_label_volume_rejects_image_hash_mismatch(tmp_path):
    label_out, image_back = _write_bound_pair(tmp_path)
    mismatched = replace(
        image_back,
        identity=replace(image_back.identity, image_content_sha256="0" * 64),
    )
    with pytest.raises(ArtifactBindingError, match="different image"):
        zarr_store.read_label_volume(label_out, image=mismatched)


# ─── chunking ────────────────────────────────────────────────────────────────


def test_write_image_volume_honours_requested_chunks(tmp_path):
    data = np.zeros((2, 8, 16, 16), dtype=np.uint16)
    vol = make_image_volume(
        data,
        spacing=ANISOTROPIC,
        channel_ids=("membrane", "signal"),
        roles=(ChannelRole.MEMBRANE, ChannelRole.SIGNAL),
    )
    requested = (1, 4, 8, 8)
    out = zarr_store.write_image_volume(vol, tmp_path / "image.ome.zarr", chunks=requested)
    g = zarr.open_group(out, mode="r")
    assert g["0"].chunks == requested


def test_write_label_volume_honours_requested_chunks(tmp_path):
    cells = np.zeros((6, 10, 10), dtype=np.uint32)
    lab = make_label_volume(cells, spacing=ANISOTROPIC)
    requested = (3, 5, 5)
    out = zarr_store.write_label_volume(lab, tmp_path / "labels.ome.zarr", chunks=requested)
    g = zarr.open_group(out, mode="r")
    assert g["labels"]["cells"]["0"].chunks == requested


# ─── no-ML import guarantee ──────────────────────────────────────────────────


def test_zarr_store_importable_without_torch_or_cellpose():
    """Fresh interpreter: importing zarr_store must not pull in torch/cellpose.

    Measurement must run on a machine with no ML stack installed at all, so
    this is checked in a subprocess with a clean sys.modules rather than the
    (already-polluted) test-runner process.
    """
    script = (
        "import sys\n"
        "from s2_adhesion.io import zarr_store\n"
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
