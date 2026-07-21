"""Tests for metrics.nuclei: nucleus-to-cell assignment by maximum overlap.

All fixtures are synthetic and built locally (per task rules, this file does
not import tests/synthetic_volumes.py -- that module is owned by a
concurrent agent). Sphere/box generators here are small and self-contained.
"""

from __future__ import annotations

import math
import subprocess
import sys

import numpy as np
import pytest

from s2_adhesion.contracts import VoxelGeometry as _Geometry
from s2_adhesion.metrics.nuclei import compute_nuclei
from tests.conftest import ANISOTROPIC, ISOTROPIC

# ─── local synthetic fixtures ──────────────────────────────────────────────


def _physical_centres(shape, spacing):
    dz, dy, dx = spacing
    Z, Y, X = shape
    z = (np.arange(Z, dtype=np.float64) + 0.5) * dz
    y = (np.arange(Y, dtype=np.float64) + 0.5) * dy
    x = (np.arange(X, dtype=np.float64) + 0.5) * dx
    return z[:, None, None], y[None, :, None], x[None, None, :]


def _sphere_mask(centre_um, radius_um, shape, spacing):
    zz, yy, xx = _physical_centres(shape, spacing)
    cz, cy, cx = centre_um
    d2 = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2
    return d2 <= radius_um**2


def _analytic_sphere_volume(radius_um):
    return (4.0 / 3.0) * math.pi * radius_um**3


def _box_label(shape, z_range, y_range, x_range, value=1, base=None):
    out = np.zeros(shape, dtype=np.uint32) if base is None else base.copy()
    out[z_range[0]:z_range[1], y_range[0]:y_range[1], x_range[0]:x_range[1]] = value
    return out


# ─── no nucleus channel at all ─────────────────────────────────────────────


def test_no_nucleus_labels_gives_all_none():
    shape = (10, 40, 40)
    cells = _sphere_mask((2.5, 2.0, 2.0), 1.5, shape, ANISOTROPIC).astype(np.uint32)
    geometry = _Geometry(ANISOTROPIC)

    out = compute_nuclei(cells, None, geometry)

    assert set(out.keys()) == {1}
    row = out[1]
    assert row["nucleus_count"] is None
    assert row["primary_nucleus_id"] is None
    assert row["nucleus_volume_um3"] is None
    assert row["nucleus_to_cell_volume_fraction"] is None
    assert row["nucleus_centroid_distance_um"] is None
    assert row["nucleus_offset_normalized"] is None
    assert row["nucleus_signal_fraction"] is None
    assert row["nuclear_enrichment"] is None
# ─── concentric / offset nucleus (shape metrics) ───────────────────────────


class TestShapeMetrics:
    SHAPE = (20, 60, 60)
    CELL_CENTRE = (5.0, 3.0, 3.0)
    CELL_RADIUS = 2.0

    def test_concentric_nucleus_volume_fraction_and_zero_offset(self):
        geometry = _Geometry(ANISOTROPIC)
        cells = _sphere_mask(
            self.CELL_CENTRE, self.CELL_RADIUS, self.SHAPE, ANISOTROPIC
        ).astype(np.uint32)
        nucleus_radius = 1.0
        nuclei = _sphere_mask(
            self.CELL_CENTRE, nucleus_radius, self.SHAPE, ANISOTROPIC
        ).astype(np.uint32)

        out = compute_nuclei(cells, nuclei, geometry)
        row = out[1]

        assert row["nucleus_count"] == 1
        assert row["primary_nucleus_id"] == 1

        cell_voxels = int(np.count_nonzero(cells == 1))
        nucleus_voxels = int(np.count_nonzero(nuclei == 1))
        voxel_volume = geometry.voxel_volume_um3
        expected_fraction = (nucleus_voxels * voxel_volume) / (cell_voxels * voxel_volume)

        assert row["nucleus_volume_um3"] == pytest.approx(
            nucleus_voxels * voxel_volume, rel=1e-9
        )
        assert row["nucleus_to_cell_volume_fraction"] == pytest.approx(
            expected_fraction, rel=1e-9
        )
        # concentric -> centroids coincide up to voxel discretisation noise
        assert row["nucleus_centroid_distance_um"] == pytest.approx(0.0, abs=0.15)
        assert row["nucleus_offset_normalized"] == pytest.approx(0.0, abs=0.15)

    def test_offset_nucleus_centroid_distance_and_normalized_offset(self):
        geometry = _Geometry(ISOTROPIC)
        shape = (60, 60, 60)
        cell_centre = (3.0, 3.0, 3.0)
        cell_radius = 2.0
        cells = _sphere_mask(cell_centre, cell_radius, shape, ISOTROPIC).astype(np.uint32)

        offset_x = 0.8
        nucleus_centre = (3.0, 3.0, 3.0 + offset_x)
        nucleus_radius = 0.5
        nuclei = _sphere_mask(nucleus_centre, nucleus_radius, shape, ISOTROPIC).astype(
            np.uint32
        )

        out = compute_nuclei(cells, nuclei, geometry)
        row = out[1]

        assert row["primary_nucleus_id"] == 1
        assert row["nucleus_centroid_distance_um"] == pytest.approx(offset_x, abs=0.1)

        cell_voxel_count = int(np.count_nonzero(cells == 1))
        cell_volume_um3 = cell_voxel_count * geometry.voxel_volume_um3
        expected_radius = (3.0 * cell_volume_um3 / (4.0 * math.pi)) ** (1.0 / 3.0)
        expected_normalized = offset_x / expected_radius
        assert row["nucleus_offset_normalized"] == pytest.approx(
            expected_normalized, rel=0.1
        )


# ─── ambiguous assignment => documented QC failure, never a coin flip ─────


class TestAmbiguousAssignment:
    def test_cell_overlapping_two_nuclei_is_ambiguous(self):
        shape = (6, 20, 20)
        geometry = _Geometry(ANISOTROPIC)
        # One cell, spanning the whole array.
        cells = np.ones(shape, dtype=np.uint32)
        # Two well-separated, fully-contained nucleus instances.
        nuclei = np.zeros(shape, dtype=np.uint32)
        nuclei[:, 2:6, 2:6] = 1
        nuclei[:, 12:16, 12:16] = 2

        out = compute_nuclei(cells, nuclei, geometry)
        row = out[1]

        assert row["nucleus_count"] == 2
        assert row["primary_nucleus_id"] is None
        assert row["nucleus_volume_um3"] is None
        assert row["nucleus_to_cell_volume_fraction"] is None
        assert row["nucleus_centroid_distance_um"] is None
        assert row["nucleus_offset_normalized"] is None

    def test_nucleus_straddling_two_cells_tied_overlap_is_ambiguous(self):
        shape = (4, 10, 20)
        geometry = _Geometry(ANISOTROPIC)
        cells = np.zeros(shape, dtype=np.uint32)
        cells[:, :, :10] = 1
        cells[:, :, 10:] = 2
        # Symmetric box straddling the boundary: 2 voxels into each cell
        # along x, identical z/y extent on both sides -> exact tie.
        nuclei = np.zeros(shape, dtype=np.uint32)
        nuclei[:, 3:7, 8:12] = 1

        out = compute_nuclei(cells, nuclei, geometry)

        for cid in (1, 2):
            row = out[cid]
            assert row["nucleus_count"] == 1, f"cell {cid} should see exactly one touching nucleus"
            assert row["primary_nucleus_id"] is None, (
                f"cell {cid}: tied overlap must not be arbitrarily assigned"
            )
            assert row["nucleus_volume_um3"] is None
            assert row["nucleus_centroid_distance_um"] is None

    def test_cell_with_no_touching_nucleus_is_a_valid_zero_not_a_failure(self):
        shape = (4, 10, 20)
        geometry = _Geometry(ANISOTROPIC)
        cells = np.zeros(shape, dtype=np.uint32)
        cells[:, :, :10] = 1
        cells[:, :, 10:] = 2
        nuclei = np.zeros(shape, dtype=np.uint32)
        nuclei[:, 3:7, 1:4] = 1  # only inside cell 1

        out = compute_nuclei(cells, nuclei, geometry)

        assert out[1]["nucleus_count"] == 1
        assert out[1]["primary_nucleus_id"] == 1

        assert out[2]["nucleus_count"] == 0
        assert out[2]["primary_nucleus_id"] is None
        assert out[2]["nucleus_volume_um3"] is None


# ─── signal fraction / nuclear enrichment ──────────────────────────────────


class TestSignalMetrics:
    SHAPE = (4, 10, 10)

    def _cell_and_nucleus(self):
        geometry = _Geometry(ANISOTROPIC)
        cells = np.ones(self.SHAPE, dtype=np.uint32)
        nuclei = np.zeros(self.SHAPE, dtype=np.uint32)
        nuclei[:, 3:7, 3:7] = 1  # box nucleus fully inside the box cell
        return geometry, cells, nuclei

    def test_signal_fraction_and_enrichment_from_known_intensities(self):
        geometry, cells, nuclei = self._cell_and_nucleus()
        signal = np.zeros(self.SHAPE, dtype=np.float64)
        nucleus_mask = nuclei == 1
        cytoplasm_mask = (cells == 1) & ~nucleus_mask
        signal[nucleus_mask] = 100.0
        signal[cytoplasm_mask] = 10.0

        out = compute_nuclei(cells, nuclei, geometry, signal=signal)
        row = out[1]

        n_nuc = int(np.count_nonzero(nucleus_mask))
        n_cyto = int(np.count_nonzero(cytoplasm_mask))
        expected_fraction = (100.0 * n_nuc) / (100.0 * n_nuc + 10.0 * n_cyto)
        assert row["nucleus_signal_fraction"] == pytest.approx(expected_fraction, rel=1e-9)
        assert row["nuclear_enrichment"] == pytest.approx(math.log2(100.0 / 10.0), rel=1e-9)

    def test_signal_none_leaves_only_signal_fields_none(self):
        geometry, cells, nuclei = self._cell_and_nucleus()
        out = compute_nuclei(cells, nuclei, geometry, signal=None)
        row = out[1]
        assert row["primary_nucleus_id"] == 1
        assert row["nucleus_volume_um3"] is not None
        assert row["nucleus_signal_fraction"] is None
        assert row["nuclear_enrichment"] is None

    def test_eps_damps_zero_denominator_instead_of_raising(self):
        geometry, cells, nuclei = self._cell_and_nucleus()
        signal = np.zeros(self.SHAPE, dtype=np.float64)
        signal[nuclei == 1] = 50.0  # cytoplasm signal is exactly zero

        out_no_eps = compute_nuclei(cells, nuclei, geometry, signal=signal, eps=0.0)
        assert out_no_eps[1]["nuclear_enrichment"] is None

        out_with_eps = compute_nuclei(cells, nuclei, geometry, signal=signal, eps=1.0)
        assert out_with_eps[1]["nuclear_enrichment"] is not None
        assert out_with_eps[1]["nuclear_enrichment"] > 0


# ─── no ML imports ──────────────────────────────────────────────────────────


def test_importing_nuclei_module_leaves_torch_and_cellpose_unimported():
    code = (
        "import sys\n"
        "import s2_adhesion.metrics.nuclei\n"
        "assert 'torch' not in sys.modules, sys.modules.keys()\n"
        "assert 'cellpose' not in sys.modules, sys.modules.keys()\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
