"""Free/contact surface partitioning and optical-resolvability flags."""

from __future__ import annotations

import numpy as np
import pytest

from s2_adhesion.config import OpticsConfig
from s2_adhesion.contracts import VoxelGeometry
from s2_adhesion.metrics.surface import (
    compute_surface_partition,
    flag_unresolved_contacts,
)

ANISO = (0.5, 0.1, 0.1)
R_UM = 4.0


def _spheres(shape, spacing, centres_um, r_um=R_UM):
    zz, yy, xx = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]].astype(np.float64)
    zz *= spacing[0]
    yy *= spacing[1]
    xx *= spacing[2]
    labels = np.zeros(shape, np.uint32)
    dists = []
    for cz, cy, cx in centres_um:
        dists.append((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2)
    stacked = np.stack(dists)
    inside = (stacked < r_um**2).any(axis=0)
    owner = np.argmin(stacked, axis=0) + 1
    labels[inside] = owner[inside]
    return labels


@pytest.fixture
def geom():
    return VoxelGeometry(spacing_um_zyx=ANISO)


@pytest.fixture
def optics():
    return OpticsConfig(axial_fwhm_um=0.7, lateral_fwhm_um=0.25)


class TestIsolatedCell:
    def test_a_lone_cell_is_all_free_surface(self, geom, optics):
        shape = (28, 120, 120)
        labels = _spheres(shape, ANISO, [(7.0, 6.0, 6.0)])
        rows = compute_surface_partition(labels, geom, optics)

        row = rows[1]
        assert row["free_surface_fraction"] == pytest.approx(1.0, abs=1e-9)
        assert row["surface_area_contact_um2"] == pytest.approx(0.0)
        assert row["mean_contact_resolution_um"] is None
        assert row["coverage_measurable"] is True

    def test_areas_sum_to_the_total(self, geom, optics):
        shape = (28, 120, 200)
        labels = _spheres(shape, ANISO, [(7.0, 6.0, 6.0), (7.0, 6.0, 13.0)])
        for row in compute_surface_partition(labels, geom, optics).values():
            parts = (
                row["surface_area_free_um2"]
                + row["surface_area_contact_um2"]
                + row["surface_area_guard_um2"]
            )
            assert parts == pytest.approx(row["surface_area_total_um2"], rel=1e-9)

    def test_fractions_sum_to_one(self, geom, optics):
        shape = (28, 120, 200)
        labels = _spheres(shape, ANISO, [(7.0, 6.0, 6.0), (7.0, 6.0, 13.0)])
        for row in compute_surface_partition(labels, geom, optics).values():
            total = (
                row["free_surface_fraction"]
                + row["contact_surface_fraction_meshed"]
                + row["guard_surface_fraction"]
            )
            assert total == pytest.approx(1.0, abs=1e-9)


class TestTouchingCells:
    def test_touching_cells_lose_free_surface_to_contact(self, geom, optics):
        shape = (28, 120, 200)
        apart = _spheres(shape, ANISO, [(7.0, 6.0, 4.5), (7.0, 6.0, 15.5)])
        touching = _spheres(shape, ANISO, [(7.0, 6.0, 6.0), (7.0, 6.0, 13.0)])

        free_apart = compute_surface_partition(apart, geom, optics)[1][
            "free_surface_fraction"
        ]
        free_touching = compute_surface_partition(touching, geom, optics)[1][
            "free_surface_fraction"
        ]
        assert free_apart > free_touching
        assert free_touching < 1.0

    def test_contact_surface_is_reported_as_unobservable(self, geom, optics):
        """Contact surface is missing data, not measurable coverage.

        Signal there belongs to both membranes and the boundary between them is
        arbitrary, so it must not enter a coverage denominator.
        """
        shape = (28, 120, 200)
        labels = _spheres(shape, ANISO, [(7.0, 6.0, 6.0), (7.0, 6.0, 13.0)])
        row = compute_surface_partition(labels, geom, optics)[1]
        assert row["observable_surface_fraction"] == row["free_surface_fraction"]
        assert row["observable_surface_fraction"] < 1.0

    def test_a_buried_cell_is_flagged_rather_than_given_a_number(self, geom, optics):
        """A cell with almost no free surface gets no confident coverage figure."""
        shape = (28, 140, 140)
        centre = (7.0, 7.0, 7.0)
        neighbours = [
            (7.0, 7.0, 7.0),
            (7.0, 1.0, 7.0),
            (7.0, 13.0, 7.0),
            (7.0, 7.0, 1.0),
            (7.0, 7.0, 13.0),
            (1.5, 7.0, 7.0),
            (12.5, 7.0, 7.0),
        ]
        labels = _spheres(shape, ANISO, neighbours, r_um=3.6)
        rows = compute_surface_partition(labels, geom, optics, min_observable_fraction=0.5)
        centre_row = rows[1]
        assert centre_row["free_surface_fraction"] < 0.5
        assert centre_row["coverage_measurable"] is False
        assert centre_row["surface_qc"] == "insufficient_free_surface"


class TestGuardBand:
    def test_guard_band_uses_physical_distance_not_voxels(self, geom, optics):
        """One voxel is 0.5 um along Z but 0.1 um along X.

        A guard band defined in voxels would be five times wider along the
        optical axis than in plane, which is precisely backwards -- the axial
        direction already has the worse resolution.
        """
        shape = (28, 120, 200)
        labels = _spheres(shape, ANISO, [(7.0, 6.0, 6.0), (7.0, 6.0, 13.5)])

        wide = compute_surface_partition(
            labels, geom, OpticsConfig(axial_fwhm_um=2.0, lateral_fwhm_um=2.0)
        )[1]["guard_surface_fraction"]
        narrow = compute_surface_partition(
            labels, geom, OpticsConfig(axial_fwhm_um=0.05, lateral_fwhm_um=0.05)
        )[1]["guard_surface_fraction"]
        assert wide > narrow


class TestUnresolvedContactFlag:
    def test_abutting_contacts_are_always_unresolved(self, optics):
        """Labels that touch have no measured cleft, so separation is zero."""
        flags = flag_unresolved_contacts({(1, 2): 0.0, (1, 3): 90.0}, optics)
        assert flags[(1, 2)]["optically_unresolved_contact"] is True
        assert flags[(1, 3)]["optically_unresolved_contact"] is True

    def test_axially_facing_contacts_have_the_worse_severity(self, optics):
        """An interface facing Z is judged against the axial PSF, not the lateral."""
        flags = flag_unresolved_contacts({(1, 2): 0.0, (1, 3): 90.0}, optics)
        facing_z = flags[(1, 2)]["optical_severity_um"]
        in_plane = flags[(1, 3)]["optical_severity_um"]
        assert facing_z == pytest.approx(0.7, abs=1e-6)
        assert in_plane == pytest.approx(0.25, abs=1e-6)
        assert facing_z > in_plane

    def test_a_well_separated_contact_is_resolved(self, optics):
        flags = flag_unresolved_contacts(
            {(1, 2): 90.0}, optics, separation_um={(1, 2): 1.0}
        )
        assert flags[(1, 2)]["optically_unresolved_contact"] is False

    def test_missing_orientation_fails_closed(self, optics):
        flags = flag_unresolved_contacts({(1, 2): None}, optics)
        assert flags[(1, 2)]["optically_unresolved_contact"] is True
        assert flags[(1, 2)]["contact_optics_qc"] == "no_orientation_evidence"


def test_no_ml_imports():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, s2_adhesion.metrics.surface as m; "
            "leaked=[k for k in sys.modules if k.split('.')[0] in ('torch','cellpose')]; "
            "print(leaked); sys.exit(1 if leaked else 0)",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout
