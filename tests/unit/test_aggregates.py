"""Tests for metrics.aggregates: connected components of the qualifying
contact graph, plus their 3D geometry.

Graph-topology fixtures use hand-built ``ContactRecord`` rows directly --
``ContactRecord`` is frozen in ``contracts.py`` so no dependency on
``metrics.contacts`` is needed. Geometry fixtures are analytic (voxelised
cuboids and spheres), matching the project convention of validating against
closed-form ground truth rather than real, unannotated data.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from s2_adhesion.contracts import ContactEstimator, ContactRecord, VoxelGeometry
from s2_adhesion.errors import MeasurementError
from s2_adhesion.metrics.aggregates import compute_aggregates

DATASET_ID = "synthetic"
FIELD_ID = "field00"
RUN_ID = "run0"


def make_contact(
    a: int,
    b: int,
    area: float = 1.0,
    qualifies: bool = True,
    valid: bool = True,
) -> ContactRecord:
    lo, hi = (a, b) if a < b else (b, a)
    return ContactRecord(
        dataset_id=DATASET_ID,
        field_id=FIELD_ID,
        segmentation_run_id=RUN_ID,
        cell_id_a=lo,
        cell_id_b=hi,
        contact_area_um2=area,
        estimator=ContactEstimator.MARCHING_CUBES,
        qualifies_as_contact=qualifies,
        valid_for_contact_metrics=valid,
    )


def labels_with_singleton_cells(
    ids: list[int], shape: tuple[int, int, int] = (1, 1, 32)
) -> np.ndarray:
    """One voxel per id, spaced out along the last axis. Geometry is not the
    point of the graph-topology tests -- only that each referenced id exists
    somewhere in the label array.
    """
    labels = np.zeros(shape, dtype=np.uint32)
    for i, cell_id in enumerate(ids):
        labels[0, 0, i] = cell_id
    return labels


def run(
    labels: np.ndarray,
    geometry: VoxelGeometry,
    contacts: list[ContactRecord],
    volumes: dict[int, float | None],
    truncated: set[int] = frozenset(),
):
    return compute_aggregates(
        labels,
        geometry,
        contacts,
        volumes,
        truncated,
        dataset_id=DATASET_ID,
        field_id=FIELD_ID,
        segmentation_run_id=RUN_ID,
    )


# ─── graph topology ─────────────────────────────────────────────────────────


class TestGraphTopology:
    def test_three_cell_chain(self, anisotropic_geometry):
        ids = [1, 2, 3]
        contacts = [make_contact(1, 2), make_contact(2, 3)]
        volumes = {i: 1.0 for i in ids}
        records = run(labels_with_singleton_cells(ids), anisotropic_geometry, contacts, volumes)

        assert len(records) == 1
        rec = records[0]
        assert rec.member_cell_ids == (1, 2, 3)
        assert rec.cell_count == 3
        assert rec.values["aggregate_cell_count"] == 3
        assert rec.values["aggregate_total_contact_area_um2"] == pytest.approx(2.0)
        assert rec.values["aggregate_mean_coordination"] == pytest.approx(2 * 2 / 3)
        assert rec.values["aggregate_max_coordination"] == 2  # cell 2 touches both ends
        assert rec.contains_truncated_cell is False

    def test_four_cell_clique(self, anisotropic_geometry):
        ids = [4, 5, 6, 7]
        pairs = [(a, b) for i, a in enumerate(ids) for b in ids[i + 1 :]]
        contacts = [make_contact(a, b) for a, b in pairs]
        volumes = {i: 1.0 for i in ids}
        records = run(labels_with_singleton_cells(ids), anisotropic_geometry, contacts, volumes)

        assert len(records) == 1
        rec = records[0]
        assert rec.member_cell_ids == (4, 5, 6, 7)
        assert rec.values["aggregate_total_contact_area_um2"] == pytest.approx(6.0)  # 6 edges
        assert rec.values["aggregate_mean_coordination"] == pytest.approx(2 * 6 / 4)
        assert rec.values["aggregate_max_coordination"] == 3

    def test_two_disjoint_pairs(self, anisotropic_geometry):
        ids = [8, 9, 10, 11]
        contacts = [make_contact(8, 9), make_contact(10, 11)]
        volumes = {i: 1.0 for i in ids}
        records = run(labels_with_singleton_cells(ids), anisotropic_geometry, contacts, volumes)

        assert len(records) == 2
        assert records[0].member_cell_ids == (8, 9)
        assert records[1].member_cell_ids == (10, 11)
        for rec in records:
            assert rec.cell_count == 2
            assert rec.values["aggregate_mean_coordination"] == pytest.approx(1.0)
            assert rec.values["aggregate_max_coordination"] == 1

    def test_isolated_cell_is_size_one_aggregate(self, anisotropic_geometry):
        ids = [12]
        volumes = {12: 1.0}
        records = run(labels_with_singleton_cells(ids), anisotropic_geometry, [], volumes)

        assert len(records) == 1
        rec = records[0]
        assert rec.member_cell_ids == (12,)
        assert rec.cell_count == 1
        assert rec.contains_truncated_cell is False
        assert rec.values["aggregate_mean_coordination"] == pytest.approx(0.0)
        assert rec.values["aggregate_max_coordination"] == 0
        assert rec.values["aggregate_total_contact_area_um2"] == pytest.approx(0.0)
        assert rec.values["aggregate_contact_area_per_cell_um2"] == pytest.approx(0.0)

    def test_non_qualifying_contact_does_not_merge(self, anisotropic_geometry):
        ids = [20, 21]
        contacts = [make_contact(20, 21, qualifies=False)]
        volumes = {i: 1.0 for i in ids}
        records = run(labels_with_singleton_cells(ids), anisotropic_geometry, contacts, volumes)

        assert len(records) == 2
        assert {r.member_cell_ids for r in records} == {(20,), (21,)}

    def test_aggregate_ids_are_sequential_by_min_member(self, anisotropic_geometry):
        ids = [8, 9, 10, 11, 12]
        contacts = [make_contact(10, 11)]
        volumes = {i: 1.0 for i in ids}
        records = run(labels_with_singleton_cells(ids), anisotropic_geometry, contacts, volumes)

        # three aggregates: {8}, {9}, {10,11}, {12} -> four aggregates actually
        assert [r.member_cell_ids for r in records] == [(8,), (9,), (10, 11), (12,)]
        assert [r.aggregate_id for r in records] == [1, 2, 3, 4]

    def test_deterministic_ordering_across_runs(self, anisotropic_geometry):
        ids = [1, 2, 3, 4, 5]
        contacts = [make_contact(1, 2), make_contact(2, 3), make_contact(4, 5)]
        volumes = {i: 1.0 for i in ids}
        labels = labels_with_singleton_cells(ids)

        run1 = run(labels, anisotropic_geometry, contacts, volumes)
        run2 = run(labels, anisotropic_geometry, list(reversed(contacts)), volumes)

        ids1 = [(r.aggregate_id, r.member_cell_ids) for r in run1]
        ids2 = [(r.aggregate_id, r.member_cell_ids) for r in run2]
        assert ids1 == ids2

    def test_contact_referencing_unknown_cell_raises(self, anisotropic_geometry):
        ids = [1, 2]
        contacts = [make_contact(1, 999)]
        volumes = {i: 1.0 for i in ids}
        with pytest.raises(MeasurementError):
            run(labels_with_singleton_cells(ids), anisotropic_geometry, contacts, volumes)

    def test_missing_volume_raises(self, anisotropic_geometry):
        ids = [1, 2]
        contacts = [make_contact(1, 2)]
        volumes = {1: 1.0}  # 2 missing
        with pytest.raises(MeasurementError):
            run(labels_with_singleton_cells(ids), anisotropic_geometry, contacts, volumes)


# ─── geometry: cuboid ───────────────────────────────────────────────────────


class TestCuboidPackingFraction:
    def test_filled_cuboid_packing_fraction_is_exactly_one(self, anisotropic_geometry):
        dz, dy, dx = anisotropic_geometry.spacing_um_zyx
        nz, ny, nx = 4, 10, 12
        labels = np.zeros((nz, ny, nx), dtype=np.uint32)
        labels[:, :, :] = 1

        # A convex hull built from voxel-CENTRE points spans from the first to
        # the last centre along each axis, i.e. (n-1)*spacing, not n*spacing.
        # Feed compute_aggregates the volume that is self-consistent with that
        # convention so packing_fraction is exactly the hull's own volume
        # ratio to itself.
        expected_hull_volume = (nz - 1) * dz * (ny - 1) * dy * (nx - 1) * dx
        volumes = {1: expected_hull_volume}

        records = run(labels, anisotropic_geometry, [], volumes)
        assert len(records) == 1
        rec = records[0]
        assert rec.values["aggregate_convex_hull_volume_um3"] == pytest.approx(
            expected_hull_volume, rel=1e-9
        )
        assert rec.values["packing_fraction"] == pytest.approx(1.0, rel=1e-9)
        assert rec.values["aggregate_valid_for_geometry"] is True
        assert rec.values["aggregate_qc_code"] == ""


# ─── geometry: sphere resolution ────────────────────────────────────────────


def _voxelised_sphere(radius_um: float, spacing: tuple[float, float, float]) -> np.ndarray:
    margin = radius_um + 3 * max(spacing)
    n = [int(2 * margin / s) + 2 for s in spacing]
    zz, yy, xx = np.meshgrid(*[np.arange(ni) for ni in n], indexing="ij")
    z_um = (zz + 0.5) * spacing[0]
    y_um = (yy + 0.5) * spacing[1]
    x_um = (xx + 0.5) * spacing[2]
    cz, cy, cx = n[0] / 2 * spacing[0], n[1] / 2 * spacing[1], n[2] / 2 * spacing[2]
    d2 = (z_um - cz) ** 2 + (y_um - cy) ** 2 + (x_um - cx) ** 2
    mask = d2 <= radius_um**2
    labels = np.zeros(mask.shape, dtype=np.uint32)
    labels[mask] = 1
    return labels


class TestSpherePackingFraction:
    def test_single_sphere_packing_fraction_approaches_one_with_finer_spacing(self):
        coarse = VoxelGeometry(spacing_um_zyx=(0.5, 0.1, 0.1))
        fine = VoxelGeometry(spacing_um_zyx=(0.1, 0.1, 0.1))

        pfs = {}
        for geometry in (coarse, fine):
            labels = _voxelised_sphere(2.0, geometry.spacing_um_zyx)
            voxel_count = int((labels == 1).sum())
            volume = voxel_count * geometry.voxel_volume_um3
            rec = run(labels, geometry, [], {1: volume})[0]
            pfs[geometry.spacing_um_zyx] = rec.values["packing_fraction"]

        pf_coarse = pfs[coarse.spacing_um_zyx]
        pf_fine = pfs[fine.spacing_um_zyx]
        assert pf_coarse is not None and pf_fine is not None
        assert abs(pf_fine - 1.0) < abs(pf_coarse - 1.0)

    def test_two_touching_spheres_packing_fraction_below_one_and_stable(self):
        def two_spheres(radius_um, sep_um, spacing):
            margin = radius_um + 3
            nz = int(2 * margin / spacing[0]) + 4
            ny = int(2 * margin / spacing[1]) + 4
            nx = int((2 * margin + sep_um) / spacing[2]) + 4
            z = (np.arange(nz) + 0.5) * spacing[0]
            y = (np.arange(ny) + 0.5) * spacing[1]
            x = (np.arange(nx) + 0.5) * spacing[2]
            zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
            cz = nz / 2 * spacing[0]
            cy = ny / 2 * spacing[1]
            cx = nx / 2 * spacing[2]
            cx1, cx2 = cx - sep_um / 2, cx + sep_um / 2
            d1 = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx1) ** 2
            d2 = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx2) ** 2
            labels = np.zeros((nz, ny, nx), dtype=np.uint32)
            labels[d1 <= radius_um**2] = 1
            labels[(d2 <= radius_um**2) & (labels == 0)] = 2
            return labels

        radius_um, sep_um = 3.0, 5.4  # sep < 2*radius -> touching
        pfs = {}
        for spacing in [(0.5, 0.1, 0.1), (0.1, 0.1, 0.1)]:
            geometry = VoxelGeometry(spacing_um_zyx=spacing)
            labels = two_spheres(radius_um, sep_um, spacing)
            volumes = {
                cid: int((labels == cid).sum()) * geometry.voxel_volume_um3 for cid in (1, 2)
            }
            contacts = [make_contact(1, 2, area=1.0)]
            rec = run(labels, geometry, contacts, volumes)[0]
            assert rec.member_cell_ids == (1, 2)
            pfs[spacing] = rec.values["packing_fraction"]

        pf_aniso = pfs[(0.5, 0.1, 0.1)]
        pf_iso = pfs[(0.1, 0.1, 0.1)]
        assert pf_aniso is not None and pf_iso is not None
        assert pf_aniso < 1.0
        assert pf_iso < 1.0
        rel_diff = abs(pf_aniso - pf_iso) / pf_aniso
        assert rel_diff < 0.05, f"packing fraction not stable across spacing: {pfs}"


# ─── nulling rules ──────────────────────────────────────────────────────────


class TestNullingRules:
    def test_truncated_member_nulls_hull_and_packing_fraction(self, anisotropic_geometry):
        labels = _voxelised_sphere(2.0, anisotropic_geometry.spacing_um_zyx)
        assert labels[0, 0, 0] == 0, "corner voxel must be free for the second cell"
        # a second, disjoint single-voxel cell so this is a genuine 2-member aggregate
        combined = labels.copy()
        combined[0, 0, 0] = 2

        volumes = {
            1: int((combined == 1).sum()) * anisotropic_geometry.voxel_volume_um3,
            2: int((combined == 2).sum()) * anisotropic_geometry.voxel_volume_um3,
        }
        contacts = [make_contact(1, 2)]
        rec = run(combined, anisotropic_geometry, contacts, volumes, truncated={2})[0]

        assert rec.member_cell_ids == (1, 2)
        assert rec.cell_count == 2
        assert rec.contains_truncated_cell is True
        assert rec.values["aggregate_convex_hull_volume_um3"] is None
        assert rec.values["packing_fraction"] is None
        assert rec.values["aggregate_valid_for_geometry"] is False
        assert "truncated_member" in rec.values["aggregate_qc_code"]
        # non-hull quantities are still reported
        assert rec.values["aggregate_volume_um3"] is not None
        assert rec.values["aggregate_cell_count"] == 2

    def test_degenerate_pair_yields_none_and_qc_code_without_raising(self, anisotropic_geometry):
        # Two single-voxel cells: only 2 points total, far short of the 4
        # non-coplanar points ConvexHull needs.
        ids = [1, 2]
        labels = labels_with_singleton_cells(ids)
        contacts = [make_contact(1, 2)]
        volumes = {1: 1.0, 2: 1.0}

        rec = run(labels, anisotropic_geometry, contacts, volumes)[0]

        assert rec.member_cell_ids == (1, 2)
        assert rec.values["aggregate_convex_hull_volume_um3"] is None
        assert rec.values["packing_fraction"] is None
        assert rec.values["aggregate_valid_for_geometry"] is False
        assert "hull_degenerate" in rec.values["aggregate_qc_code"]

    def test_missing_member_volume_nulls_dependent_fields_not_topology(
        self, anisotropic_geometry
    ):
        ids = [1, 2]
        labels = labels_with_singleton_cells(ids)
        contacts = [make_contact(1, 2)]
        volumes = {1: 1.0, 2: None}

        rec = run(labels, anisotropic_geometry, contacts, volumes)[0]
        assert rec.values["aggregate_volume_um3"] is None
        assert rec.values["packing_fraction"] is None
        assert rec.values["aggregate_equivalent_sphere_diameter_um"] is None
        assert rec.values["aggregate_sphericity"] is None
        assert rec.cell_count == 2  # topology is unaffected


# ─── no heavy/ML imports ────────────────────────────────────────────────────


def test_importing_module_does_not_import_torch_or_cellpose():
    code = (
        "import sys\n"
        "import s2_adhesion.metrics.aggregates\n"
        "assert 'torch' not in sys.modules, sys.modules.keys()\n"
        "assert 'cellpose' not in sys.modules, sys.modules.keys()\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout
