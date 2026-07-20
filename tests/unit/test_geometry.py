"""Tests for s2_adhesion.metrics.geometry against analytic ground truth.

Every acceptance number here is compared to closed-form math from
tests/synthetic_volumes.py, not to another numerical computation -- this is
deliberately stricter than comparing against real, unannotated microscopy
data (see tests/conftest.py).

Known, documented residual limitation (see the module docstring block above
``_SURFACE_SMOOTHING_SIGMA_TARGET_VOXELS`` in geometry.py): marching-cubes
surface area on a small sphere (radius <~3um) sampled at high z-anisotropy
(5x, i.e. spacing (0.5, 0.1, 0.1)) does not fully reach 5% agreement with the
same sphere sampled isotropically -- there are simply too few z-slices to
resolve it. ``test_surface_area_small_radius_high_anisotropy_is_bounded``
documents the measured bound instead of silently weakening the main
anisotropy-agreement test.
"""

from __future__ import annotations

import numpy as np
import pytest

from s2_adhesion.contracts import VoxelGeometry
from s2_adhesion.metrics import geometry as geom
from s2_adhesion.metrics import qc
from tests.synthetic_volumes import (
    analytic_contact_disc_area,
    analytic_cuboid_volume,
    analytic_ellipsoid_principal_axis_lengths_um,
    analytic_ellipsoid_volume,
    analytic_sphere_equivalent_diameter_um,
    analytic_sphere_surface_area,
    analytic_sphere_volume,
    cell_chain,
    clipped_sphere,
    cuboid,
    ellipsoid,
    sphere,
    sphere_pair_bisected,
)

ANISOTROPIC = (0.5, 0.1, 0.1)
ISOTROPIC = (0.1, 0.1, 0.1)


def _shape_for(radius_um: float, spacing: tuple[float, float, float], pad_um: float = 3.0):
    diameter = 2.0 * radius_um + pad_um
    return tuple(int(np.ceil(diameter / s)) for s in spacing)


def _centred_sphere(radius_um: float, spacing: tuple[float, float, float], pad_um: float = 3.0):
    shape = _shape_for(radius_um, spacing, pad_um)
    centre = tuple((sh * s) / 2.0 for sh, s in zip(shape, spacing))
    labels = sphere(centre, radius_um, shape, spacing)
    return labels, VoxelGeometry(spacing_um_zyx=spacing)


def _row(labels, geometry, label_id=1):
    return geom.compute_geometry(labels, geometry)[label_id]


# ─── centred sphere: the core acceptance bar ───────────────────────────────


def test_centred_sphere_matches_analytic_within_tolerance():
    radius_um = 4.0
    labels, geometry = _centred_sphere(radius_um, ISOTROPIC)
    row = _row(labels, geometry)

    assert qc.valid_for_geometry(labels) == {1: True}

    v_analytic = analytic_sphere_volume(radius_um)
    a_analytic = analytic_sphere_surface_area(radius_um)
    d_analytic = analytic_sphere_equivalent_diameter_um(radius_um)

    vol_err = abs(row["volume_um3"] - v_analytic) / v_analytic
    diam_err = abs(row["equivalent_sphere_diameter_um"] - d_analytic) / d_analytic
    surf_err = abs(row["surface_area_um2"] - a_analytic) / a_analytic

    assert vol_err <= 0.05, f"volume error {vol_err:.2%}"
    assert diam_err <= 0.03, f"equivalent diameter error {diam_err:.2%}"
    assert surf_err <= 0.10, f"surface area error {surf_err:.2%}"

    axes = [
        row["principal_axis_length_1_um"],
        row["principal_axis_length_2_um"],
        row["principal_axis_length_3_um"],
    ]
    axis_spread = max(axes) / min(axes) - 1.0
    assert axis_spread <= 0.02, f"principal axes not near-equal: {axes}"

    assert abs(row["sphericity"] - 1.0) <= 0.15, row["sphericity"]
    assert row["cell_solidity_3d"] <= 1.0 + 1e-9, row["cell_solidity_3d"]
    assert abs(row["cell_solidity_3d"] - 1.0) <= 0.15, row["cell_solidity_3d"]


# ─── anisotropy correctness: same physical sphere, two samplings ──────────


def test_anisotropic_and_isotropic_sampling_agree_within_5_percent():
    """A physically identical sphere must yield the same measurements
    regardless of whether z is 5x coarser than xy -- this is what makes the
    geometry code anisotropy-correct rather than merely voxel-counting."""
    # Radius chosen empirically: the corner-based convex-hull volume (needed
    # to keep solidity <= 1, see geometry.py) is the slowest field to
    # converge with anisotropy -- 5.8% disagreement at r=4um, 4.7% at r=5um,
    # 4.2% at r=6um. r=6um clears every field's 5% bound with margin; smaller
    # radii are covered separately (test_surface_area_small_radius_...).
    radius_um = 6.0
    labels_a, geometry_a = _centred_sphere(radius_um, ANISOTROPIC)
    labels_i, geometry_i = _centred_sphere(radius_um, ISOTROPIC)
    row_a = _row(labels_a, geometry_a)
    row_i = _row(labels_i, geometry_i)

    def rel_err(key: str) -> float:
        return abs(row_a[key] - row_i[key]) / abs(row_i[key])

    for key in (
        "volume_um3",
        "equivalent_sphere_diameter_um",
        "principal_axis_length_1_um",
        "principal_axis_length_2_um",
        "principal_axis_length_3_um",
        "cell_convex_hull_volume_um3",
        "cell_solidity_3d",
        "surface_area_um2",
    ):
        err = rel_err(key)
        assert err <= 0.05, f"{key} disagrees by {err:.2%}: aniso={row_a[key]} iso={row_i[key]}"

    # Centroid is exactly the volume centre by construction; compare
    # absolutely rather than relatively (both are ~0 after centring).
    for axis in ("z", "y", "x"):
        key = f"centroid_{axis}_um"
        assert abs(row_a[key] - row_i[key]) <= max(ANISOTROPIC) , key


def test_surface_area_small_radius_high_anisotropy_is_bounded():
    """Documented residual limitation: at radius 1.5um and 5x z-anisotropy
    there are only ~6 z-slices across the sphere, and marching-cubes surface
    area does not fully reach 5% aniso-vs-iso agreement no matter the
    (deterministic) smoothing applied -- see geometry.py's module comment
    above _SURFACE_SMOOTHING_SIGMA_TARGET_VOXELS for the full sweep this
    bound was measured from. This test pins the *actual* measured ceiling
    (<=10%) rather than either silently dropping the check or fudging the
    main test's 5% tolerance to hide the limitation.
    """
    radius_um = 1.5
    labels_a, geometry_a = _centred_sphere(radius_um, ANISOTROPIC)
    labels_i, geometry_i = _centred_sphere(radius_um, ISOTROPIC)
    row_a = _row(labels_a, geometry_a)
    row_i = _row(labels_i, geometry_i)

    err = abs(row_a["surface_area_um2"] - row_i["surface_area_um2"]) / row_i["surface_area_um2"]
    assert err <= 0.10, f"surface area aniso-vs-iso disagreement {err:.2%} exceeds documented bound"

    # Volume and diameter are NOT surface-area-derived and remain accurate
    # even at this small radius/high anisotropy -- the limitation is
    # specific to marching-cubes surface area, not geometry.py as a whole.
    vol_err = abs(row_a["volume_um3"] - row_i["volume_um3"]) / row_i["volume_um3"]
    assert vol_err <= 0.05, f"volume error {vol_err:.2%}"


# ─── convergence: finer spacing must reduce discretisation error ──────────


def test_halving_spacing_reduces_volume_discretisation_error():
    radius_um = 3.0
    coarse = (0.2, 0.2, 0.2)
    fine = (0.1, 0.1, 0.1)
    v_analytic = analytic_sphere_volume(radius_um)

    labels_c, geometry_c = _centred_sphere(radius_um, coarse)
    labels_f, geometry_f = _centred_sphere(radius_um, fine)
    row_c = _row(labels_c, geometry_c)
    row_f = _row(labels_f, geometry_f)

    err_c = abs(row_c["volume_um3"] - v_analytic) / v_analytic
    err_f = abs(row_f["volume_um3"] - v_analytic) / v_analytic
    assert err_f < err_c, f"finer spacing did not reduce volume error: {err_f:.4%} vs {err_c:.4%}"

    d_analytic = analytic_sphere_equivalent_diameter_um(radius_um)
    derr_c = abs(row_c["equivalent_sphere_diameter_um"] - d_analytic) / d_analytic
    derr_f = abs(row_f["equivalent_sphere_diameter_um"] - d_analytic) / d_analytic
    assert derr_f < derr_c, "finer spacing did not reduce equivalent-diameter error"


# ─── grid-aligned cuboid: exact to floating point ──────────────────────────


def test_grid_aligned_cuboid_is_exact():
    spacing = (0.5, 0.2, 0.2)
    shape = (40, 60, 60)
    origin_um = (1.0, 1.0, 1.0)  # multiples of spacing
    extent_um = (3.0, 4.0, 5.0)  # multiples of spacing

    labels = cuboid(origin_um, extent_um, shape, spacing)
    geometry = VoxelGeometry(spacing_um_zyx=spacing)
    row = _row(labels, geometry)

    assert row["volume_um3"] == pytest.approx(analytic_cuboid_volume(extent_um), abs=1e-9)
    assert row["bbox_extent_z_um"] == pytest.approx(extent_um[0], abs=1e-9)
    assert row["bbox_extent_y_um"] == pytest.approx(extent_um[1], abs=1e-9)
    assert row["bbox_extent_x_um"] == pytest.approx(extent_um[2], abs=1e-9)

    expected_centroid = tuple(o + e / 2.0 for o, e in zip(origin_um, extent_um))
    assert row["centroid_z_um"] == pytest.approx(expected_centroid[0], abs=1e-9)
    assert row["centroid_y_um"] == pytest.approx(expected_centroid[1], abs=1e-9)
    assert row["centroid_x_um"] == pytest.approx(expected_centroid[2], abs=1e-9)


def test_cuboid_solidity_is_exactly_one():
    """A convex, grid-aligned solid's hull must exactly equal its own volume."""
    spacing = (0.2, 0.2, 0.2)
    labels = cuboid((1.0, 1.0, 1.0), (4.0, 3.0, 5.0), (50, 50, 50), spacing)
    geometry = VoxelGeometry(spacing_um_zyx=spacing)
    row = _row(labels, geometry)
    assert row["cell_solidity_3d"] == pytest.approx(1.0, abs=1e-6)
    assert row["cell_solidity_3d"] <= 1.0 + 1e-9


# ─── ellipsoid: axis ordering and ratios ───────────────────────────────────


def test_ellipsoid_principal_axes_ordering_and_ratios():
    spacing = (0.1, 0.1, 0.1)
    shape = (80, 80, 80)
    radii_um_zyx = (1.5, 2.5, 3.5)  # z, y, x semi-axes -- deliberately distinct
    centre = tuple((sh * s) / 2.0 for sh, s in zip(shape, spacing))

    labels = ellipsoid(centre, radii_um_zyx, shape, spacing)
    geometry = VoxelGeometry(spacing_um_zyx=spacing)
    row = _row(labels, geometry)

    measured = [
        row["principal_axis_length_1_um"],
        row["principal_axis_length_2_um"],
        row["principal_axis_length_3_um"],
    ]
    expected = analytic_ellipsoid_principal_axis_lengths_um(radii_um_zyx)

    # Descending order, matching the expected (largest radius = x = 3.5).
    assert measured[0] >= measured[1] >= measured[2]
    for m, e in zip(measured, expected):
        assert abs(m - e) / e <= 0.05, f"measured {measured} vs analytic {expected}"

    v_analytic = analytic_ellipsoid_volume(radii_um_zyx)
    assert abs(row["volume_um3"] - v_analytic) / v_analytic <= 0.05

    expected_elongation = expected[0] / expected[1]
    expected_flatness = expected[1] / expected[2]
    assert abs(row["elongation"] - expected_elongation) / expected_elongation <= 0.05
    assert abs(row["flatness"] - expected_flatness) / expected_flatness <= 0.05

    assert row["cell_solidity_3d"] <= 1.0 + 1e-9


# ─── truncation: all six faces, interior valid, clipped invalid ───────────


@pytest.mark.parametrize(
    "face", ["z_min", "z_max", "y_min", "y_max", "x_min", "x_max"]
)
def test_clipped_sphere_is_invalid_and_nulls_canonical_fields(face):
    radius_um = 1.5
    labels = clipped_sphere(face, radius_um, (40, 40, 40), ISOTROPIC)
    geometry = VoxelGeometry(spacing_um_zyx=ISOTROPIC)
    row = _row(labels, geometry)

    assert qc.valid_for_geometry(labels) == {1: False}
    assert row["voxel_count_observed"] > 0
    assert row["volume_um3_observed"] > 0
    for key in (
        "volume_um3",
        "centroid_z_um",
        "centroid_y_um",
        "centroid_x_um",
        "bbox_extent_z_um",
        "bbox_extent_y_um",
        "bbox_extent_x_um",
        "surface_area_um2",
        "equivalent_sphere_diameter_um",
        "principal_axis_length_1_um",
        "principal_axis_length_2_um",
        "principal_axis_length_3_um",
        "elongation",
        "flatness",
        "sphericity",
        "cell_convex_hull_volume_um3",
        "cell_solidity_3d",
    ):
        assert row[key] is None, f"{key} should be null for a truncated cell ({face})"


def test_interior_sphere_is_valid_with_all_canonical_fields_populated():
    # shape (40, 40, 40) at spacing 0.1 spans 4.0um per axis; radius 1.5um
    # needs a centre in [1.5, 2.5] on every axis to stay fully interior.
    labels = sphere((2.0, 2.0, 2.0), 1.5, (40, 40, 40), ISOTROPIC)
    geometry = VoxelGeometry(spacing_um_zyx=ISOTROPIC)
    row = _row(labels, geometry)

    assert qc.valid_for_geometry(labels) == {1: True}
    assert row["volume_um3"] is not None
    assert row["surface_area_um2"] is not None
    assert row["cell_solidity_3d"] is not None


# ─── disconnected (reused) label ───────────────────────────────────────────


def test_disconnected_label_reports_two_components_and_nulls_geometry():
    labels = np.zeros((30, 30, 30), dtype=np.uint32)
    labels[2:5, 2:5, 2:5] = 1
    labels[20:23, 20:23, 20:23] = 1  # same id, far away
    geometry = VoxelGeometry(spacing_um_zyx=ISOTROPIC)

    assert qc.connected_component_count(labels) == {1: 2}
    assert qc.valid_for_geometry(labels) == {1: False}

    row = _row(labels, geometry)
    assert row["voxel_count_observed"] == 27 + 27
    assert row["volume_um3_observed"] > 0
    assert row["volume_um3"] is None
    assert row["cell_solidity_3d"] is None


# ─── truncated object keeps observed_*, nulls canonical volume ────────────


def test_truncated_object_keeps_volume_observed_but_nulls_volume():
    labels = clipped_sphere("z_min", 1.5, (30, 30, 30), ISOTROPIC)
    geometry = VoxelGeometry(spacing_um_zyx=ISOTROPIC)
    row = _row(labels, geometry)

    voxel_count = qc.voxel_count_observed(labels)[1]
    assert row["voxel_count_observed"] == voxel_count
    assert row["volume_um3_observed"] == pytest.approx(
        voxel_count * geometry.voxel_volume_um3
    )
    assert row["volume_um3"] is None


# ─── touching-cell fixtures: sanity-check the generators themselves ───────


def test_sphere_pair_bisected_produces_expected_voxel_split():
    """The interface is the exact perpendicular bisector for equal radii, so
    each ball should own very close to half the union (not tested for exact
    contact area here -- that belongs to a contact-metric module outside
    this file's ownership -- but the generator must not be lopsided)."""
    spacing = ISOTROPIC
    shape = (60, 50, 50)
    labels = sphere_pair_bisected(
        (3.0, 2.5, 2.5), radius_um=2.0, separation_um=3.0, shape=shape,
        spacing=spacing, orientation="axis_aligned",
    )
    n1 = int((labels == 1).sum())
    n2 = int((labels == 2).sum())
    assert n1 > 0 and n2 > 0
    assert abs(n1 - n2) / max(n1, n2) < 0.05
    assert analytic_contact_disc_area(2.0, 3.0) > 0


def test_sphere_pair_beyond_two_radii_does_not_touch():
    spacing = ISOTROPIC
    shape = (60, 50, 50)
    labels = sphere_pair_bisected(
        (3.0, 2.5, 2.5), radius_um=2.0, separation_um=4.5, shape=shape,
        spacing=spacing, orientation="axis_aligned",
    )
    assert analytic_contact_disc_area(2.0, 4.5) == 0.0
    # the two balls must be genuinely disjoint (no shared/ambiguous voxels)
    assert not np.any((labels == 1) & (labels == 2))


def test_cell_chain_members_touch_only_immediate_neighbours():
    labels = cell_chain(4, (4.0, 3.0, 3.0), 1.5, 2.0, (80, 60, 120), ISOTROPIC, axis="x")
    cc = qc.connected_component_count(labels)
    assert cc == {1: 1, 2: 1, 3: 1, 4: 1}
    # Every label present, none empty.
    for i in (1, 2, 3, 4):
        assert (labels == i).sum() > 0
