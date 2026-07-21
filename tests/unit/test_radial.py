"""Radial-profile metrics, validated on synthetic cells modelling the real biology.

The three cases that matter experimentally:

  * a transmembrane protein -> signal in a shell at the cell surface
  * the same protein expressed over only part of the surface -> a partial shell
    (an incomplete ring in any single plane)
  * a construct that failed to reach the membrane -> signal in the interior

A membrane-vs-interior call has to separate the first two from the third.
"""

from __future__ import annotations

import numpy as np
import pytest

from s2_adhesion.contracts import VoxelGeometry
from s2_adhesion.metrics.radial import (
    compute_radial_metrics,
    normalised_radius,
    radial_profile,
)

ANISO = (0.5, 0.1, 0.1)
ISO = (0.2, 0.2, 0.2)
R_UM = 4.0


def _grid(shape, spacing):
    zz, yy, xx = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]].astype(np.float64)
    zz *= spacing[0]
    yy *= spacing[1]
    xx *= spacing[2]
    centre = np.array(
        [shape[0] * spacing[0] / 2, shape[1] * spacing[1] / 2, shape[2] * spacing[2] / 2]
    )
    radius = np.sqrt(
        (zz - centre[0]) ** 2 + (yy - centre[1]) ** 2 + (xx - centre[2]) ** 2
    )
    return zz, yy, xx, centre, radius


def synthetic_cell(
    shape,
    spacing,
    kind: str,
    *,
    r_um: float = R_UM,
    shell_thickness_um: float = 0.8,
    coverage: float = 1.0,
    background: float = 10.0,
    amplitude: float = 500.0,
):
    """Return (labels, intensity) for one cell with the requested signal pattern.

    kind:
      "shell"     transmembrane protein, signal in a surface shell
      "interior"  mislocalised protein, signal filling the cell body
      "uniform"   signal everywhere in the cell, the ambiguous middle case
    ``coverage`` < 1 removes part of the shell, modelling a protein that is not
    expressed over the whole surface.
    """
    zz, yy, xx, centre, radius = _grid(shape, spacing)
    cell = radius < r_um
    labels = cell.astype(np.uint32)

    intensity = np.full(shape, background, dtype=np.float64)
    if kind == "shell":
        region = cell & (radius >= r_um - shell_thickness_um)
        if coverage < 1.0:
            # Keep only a polar cap covering the requested solid-angle fraction.
            cos_theta = (zz - centre[0]) / np.maximum(radius, 1e-9)
            cutoff = 1.0 - 2.0 * coverage  # coverage=1 -> -1 (all), 0.5 -> 0
            region &= cos_theta >= cutoff
        intensity[region] += amplitude
    elif kind == "interior":
        intensity[cell & (radius < r_um - shell_thickness_um)] += amplitude
    elif kind == "uniform":
        intensity[cell] += amplitude
    else:  # pragma: no cover - guards a typo in a test
        raise ValueError(kind)

    return labels, intensity


class TestNormalisedRadius:
    def test_runs_from_zero_at_the_centre_to_one_at_the_edge(self):
        shape = (24, 100, 100)
        labels, _ = synthetic_cell(shape, ANISO, "uniform")
        geom = VoxelGeometry(spacing_um_zyx=ANISO)
        r = normalised_radius(labels.astype(bool), geom)

        inside = labels.astype(bool)
        assert r[inside].min() == pytest.approx(0.0, abs=1e-9)
        assert r[inside].max() == pytest.approx(1.0, abs=0.05)
        assert (r[~inside] == 0.0).all()

    def test_uses_physical_spacing_not_voxel_counts(self):
        """One voxel along Z is 5x further than one along X at this sampling.

        With voxel-count distances the profile would be systematically skewed
        along the optical axis.
        """
        shape = (24, 100, 100)
        labels, _ = synthetic_cell(shape, ANISO, "uniform")
        geom_aniso = VoxelGeometry(spacing_um_zyx=ANISO)
        geom_wrong = VoxelGeometry(spacing_um_zyx=(0.1, 0.1, 0.1))

        r_right = normalised_radius(labels.astype(bool), geom_aniso)
        r_wrong = normalised_radius(labels.astype(bool), geom_wrong)
        assert not np.allclose(r_right, r_wrong), (
            "spacing was ignored -- the transform is not physical"
        )

    def test_empty_mask_is_handled(self):
        geom = VoxelGeometry(spacing_um_zyx=ANISO)
        r = normalised_radius(np.zeros((4, 8, 8), bool), geom)
        assert (r == 0.0).all()


class TestProfileShape:
    def test_shell_signal_peaks_at_the_periphery(self):
        shape = (24, 100, 100)
        labels, intensity = synthetic_cell(shape, ANISO, "shell")
        geom = VoxelGeometry(spacing_um_zyx=ANISO)
        profile = radial_profile(intensity - 10.0, labels.astype(bool), geom)

        means = np.asarray(profile["mean_intensity"])
        assert np.nanargmax(means) >= len(means) - 3, "shell did not peak at the edge"

    def test_interior_signal_peaks_at_the_centre(self):
        shape = (24, 100, 100)
        labels, intensity = synthetic_cell(shape, ANISO, "interior")
        geom = VoxelGeometry(spacing_um_zyx=ANISO)
        profile = radial_profile(intensity - 10.0, labels.astype(bool), geom)

        means = np.asarray(profile["mean_intensity"])
        assert np.nanargmax(means) <= 2, "interior signal did not peak at the centre"

    def test_every_cell_voxel_lands_in_exactly_one_bin(self):
        shape = (24, 100, 100)
        labels, intensity = synthetic_cell(shape, ANISO, "uniform")
        geom = VoxelGeometry(spacing_um_zyx=ANISO)
        profile = radial_profile(intensity, labels.astype(bool), geom)
        assert sum(profile["voxel_count"]) == int(labels.sum())


class TestMembraneVersusInterior:
    """The primary new readout."""

    def _score(self, kind, spacing=ANISO, shape=(24, 100, 100), **kw):
        labels, intensity = synthetic_cell(shape, spacing, kind, **kw)
        geom = VoxelGeometry(spacing_um_zyx=spacing)
        out = compute_radial_metrics(
            labels, {"signal": intensity}, geom,
            background_by_channel={"signal": 10.0},
            eps_by_channel={"signal": 1.0},
        )
        return out[1]

    def test_shell_scores_clearly_above_interior(self):
        shell = self._score("shell")["ch.signal.radial_shell_score"]
        interior = self._score("interior")["ch.signal.radial_shell_score"]
        assert shell > 1.0, f"shell score too low: {shell}"
        assert interior < -1.0, f"interior score too high: {interior}"
        assert shell - interior > 4.0, "the two cases are not separated"

    def test_partial_shell_is_still_called_shell_like(self):
        """A protein expressed over only half the surface is still membrane-localised.

        Radial position and surface COVERAGE are different questions; this metric
        answers the first. Coverage is measured separately.
        """
        full = self._score("shell", coverage=1.0)["ch.signal.radial_shell_score"]
        half = self._score("shell", coverage=0.5)["ch.signal.radial_shell_score"]
        assert half > 1.0, f"partial shell misread as non-membrane: {half}"
        assert full > 1.0

    def test_peripheral_fraction_is_volume_weighted_and_a_weak_discriminator(self):
        """Documents a trap rather than a capability.

        ``peripheral_signal_fraction`` integrates, so it inherits the fact that
        a sphere keeps most of its volume near its surface: the outer 30% of the
        radius already holds 1 - 0.7^3 = 66% of the volume. A perfectly UNIFORM
        signal therefore scores ~0.66, close to a real shell, and even a signal
        confined to the inner 80% of the radius still scores ~0.30.

        So this field must not be used alone to call membrane localisation. The
        mean-based ``radial_shell_score`` compares bin means and is not volume
        weighted, which is why it is the discriminator and this is context.
        """
        shell = self._score("shell")["ch.signal.peripheral_signal_fraction"]
        uniform = self._score("uniform")["ch.signal.peripheral_signal_fraction"]
        interior = self._score("interior")["ch.signal.peripheral_signal_fraction"]

        assert shell > 0.7
        assert 0.55 < uniform < 0.75, "uniform should sit near the volume fraction 0.66"
        assert interior < 0.45
        # The ordering holds, but shell and uniform are NOT well separated.
        assert shell > uniform > interior

    def test_shell_score_separates_uniform_from_shell_where_fraction_cannot(self):
        """The mean-based score is the one that actually discriminates."""
        shell = self._score("shell")["ch.signal.radial_shell_score"]
        uniform = self._score("uniform")["ch.signal.radial_shell_score"]
        assert uniform == pytest.approx(0.0, abs=0.4), "uniform signal is not flat"
        assert shell - uniform > 2.0

    def test_peak_position_separates_the_cases(self):
        assert self._score("shell")["ch.signal.radial_peak_position"] > 0.75
        assert self._score("interior")["ch.signal.radial_peak_position"] < 0.35

    def test_call_survives_a_change_of_sampling(self):
        """Same physical cell, two voxel sizes -- the conclusion must not flip.

        This is the property that makes the metric usable at 5x anisotropy: it
        is a volume integral, so it inherits the accuracy of volume rather than
        the orientation-dependence of surface reconstruction.
        """
        aniso = self._score("shell", spacing=ANISO, shape=(24, 100, 100))
        iso = self._score("shell", spacing=ISO, shape=(60, 60, 60))
        assert aniso["ch.signal.radial_shell_score"] > 1.0
        assert iso["ch.signal.radial_shell_score"] > 1.0
        assert aniso["ch.signal.peripheral_signal_fraction"] == pytest.approx(
            iso["ch.signal.peripheral_signal_fraction"], abs=0.15
        )


class TestContracts:
    def test_is_role_agnostic(self):
        """Nothing here may depend on what a channel is said to stain."""
        shape = (24, 100, 100)
        labels, intensity = synthetic_cell(shape, ANISO, "shell")
        geom = VoxelGeometry(spacing_um_zyx=ANISO)
        a = compute_radial_metrics(labels, {"alpha": intensity}, geom)
        b = compute_radial_metrics(labels, {"omega": intensity}, geom)
        assert (
            a[1]["ch.alpha.radial_shell_score"] == b[1]["ch.omega.radial_shell_score"]
        )

    def test_handles_multiple_cells_and_non_consecutive_ids(self):
        shape = (24, 100, 100)
        labels, intensity = synthetic_cell(shape, ANISO, "shell")
        labels = labels.copy()
        labels[labels == 1] = 7
        geom = VoxelGeometry(spacing_um_zyx=ANISO)
        out = compute_radial_metrics(labels, {"signal": intensity}, geom)
        assert list(out) == [7]

    def test_no_ml_imports(self):
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys, s2_adhesion.metrics.radial as m; "
                "leaked=[k for k in sys.modules if k.split('.')[0] in ('torch','cellpose')]; "
                "print(leaked); sys.exit(1 if leaked else 0)",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout
