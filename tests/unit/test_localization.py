"""Tests for metrics.localization: signed-distance profiles, shell enrichment
scores, per-cell colocalization, and the (default-disabled) categorical call.

All fixtures are synthetic and built locally (per task rules, this file does
not import tests/synthetic_volumes.py -- that module is owned by a
concurrent agent).
"""

from __future__ import annotations

import math
import subprocess
import sys

import numpy as np
import pytest

from s2_adhesion.config import (
    ColocalizationPairConfig,
    LocalizationConfig,
    LocalizationDecisionConfig,
    ThresholdSpec,
)
from s2_adhesion.contracts import VoxelGeometry as _Geometry
from s2_adhesion.errors import MeasurementError
from s2_adhesion.metrics.localization import (
    LocalizationDecisionInputs,
    apply_threshold_spec,
    classify_localization,
    compute_colocalization,
    compute_shell_enrichment_scores,
    compute_signed_distance_profiles,
)
from tests.conftest import ANISOTROPIC

# ─── local synthetic fixtures ──────────────────────────────────────────────


def _physical_centres(shape, spacing):
    dz, dy, dx = spacing
    Z, Y, X = shape
    z = (np.arange(Z, dtype=np.float64) + 0.5) * dz
    y = (np.arange(Y, dtype=np.float64) + 0.5) * dy
    x = (np.arange(X, dtype=np.float64) + 0.5) * dx
    return z[:, None, None], y[None, :, None], x[None, None, :]


def _radius_field(centre_um, shape, spacing):
    zz, yy, xx = _physical_centres(shape, spacing)
    cz, cy, cx = centre_um
    return np.sqrt((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2)


def _sphere_mask(centre_um, radius_um, shape, spacing):
    return _radius_field(centre_um, shape, spacing) <= radius_um


# ─── signed distance profiles ──────────────────────────────────────────────


class TestSignedDistanceProfiles:
    # A larger radius keeps discrete-EDT curvature error small relative to
    # the halo thickness being tested; a tight radius (a few voxels) makes
    # the digital sphere too staircase-y for a thin halo to stay contained
    # in one distance bin.
    SHAPE = (30, 80, 80)
    CENTRE = (7.5, 4.0, 4.0)
    RADIUS = 3.0

    def _cell(self):
        return _sphere_mask(self.CENTRE, self.RADIUS, self.SHAPE, ANISOTROPIC).astype(
            np.uint32
        )

    def test_halo_signal_lands_in_expected_negative_bins(self):
        geometry = _Geometry(ANISOTROPIC)
        cells = self._cell()
        r = _radius_field(self.CENTRE, self.SHAPE, ANISOTROPIC)
        halo = (r > self.RADIUS) & (r <= self.RADIUS + 0.4)
        signal = np.zeros(self.SHAPE, dtype=np.float64)
        signal[halo] = 7.0

        config = LocalizationConfig(
            signed_distance_bin_edges_um=(-1.0, -0.5, 0.0, 0.5, 1.0)
        )
        records = compute_signed_distance_profiles(
            cells, geometry, "ds", "field0", {"sig": signal}, config
        )
        by_bin = {(r.bin_start_um, r.bin_end_um): r for r in records if r.cell_id == 1}

        near_bin = by_bin[(-0.5, 0.0)]
        far_bin = by_bin[(-1.0, -0.5)]
        assert near_bin.mean_intensity == pytest.approx(7.0, rel=0.25)
        # The z spacing (0.5 um) is comparable to the bin width, so a sliver
        # of halo signal at the discretised edge can be assigned to the
        # neighbouring negative bin -- but it must stay a small minority of
        # the total halo signal, concentrated in the expected bin.
        near_total = near_bin.mean_intensity * near_bin.voxel_count
        far_total = far_bin.mean_intensity * far_bin.voxel_count
        assert near_total > 10 * far_total
        # Positive-distance (inside-cell) bins must be exactly zero: signal
        # was placed only in the outside halo, never inside the cell.
        assert by_bin[(0.0, 0.5)].mean_intensity == 0.0
        assert by_bin[(0.5, 1.0)].mean_intensity == 0.0
        assert near_bin.voxel_count > 0

    def test_two_adjacent_cells_no_double_counting_of_outside_voxels(self):
        shape = (4, 10, 20)
        geometry = _Geometry(ANISOTROPIC)
        cells = np.zeros(shape, dtype=np.uint32)
        cells[:, :, :10] = 1
        cells[:, :, 10:] = 2
        corrected_value = 3.0
        corrected = np.full(shape, corrected_value, dtype=np.float64)
        raw = np.zeros(shape, dtype=np.float64)

        # a single wide bin so every voxel in the (small) volume lands inside
        # it, isolating the no-double-counting property from binning cutoffs
        config = LocalizationConfig(signed_distance_bin_edges_um=(-1000.0, 1000.0))
        records = compute_signed_distance_profiles(
            cells,
            geometry,
            "ds",
            "field0",
            {"sig": raw},
            config,
            corrected_channels={"sig": corrected},
        )

        total_voxel_count = sum(r.voxel_count for r in records)
        assert total_voxel_count == shape[0] * shape[1] * shape[2]

        total_integrated = sum(r.integrated_corrected_intensity for r in records)
        expected = corrected_value * total_voxel_count * geometry.voxel_volume_um3
        assert total_integrated == pytest.approx(expected, rel=1e-9)

    def test_profiles_agree_between_anisotropic_and_isotropic_sampling(self):
        # Every shell (radius + outermost bin edge) must stay clear of the
        # array boundary in every axis, or the two resolutions would
        # discretise the boundary clip differently and disagree for a
        # reason unrelated to the sampling comparison being tested. A large
        # radius relative to the 1 um bin width also keeps voxel-scale
        # curvature discretisation (dominated by the coarse 0.5 um z
        # spacing) a small fraction of each shell's volume.
        half_extent = 12.0
        centre = (half_extent, half_extent, half_extent)
        radius = 8.0
        aniso_spacing = ANISOTROPIC
        aniso_shape = (
            int(2 * half_extent / aniso_spacing[0]),
            int(2 * half_extent / aniso_spacing[1]),
            int(2 * half_extent / aniso_spacing[2]),
        )
        iso_spacing = (0.1, 0.1, 0.1)
        iso_shape = tuple(int(2 * half_extent / s) for s in iso_spacing)
        edges = (-2.0, -1.0, 0.0, 1.0, 2.0)
        config = LocalizationConfig(signed_distance_bin_edges_um=edges)

        results = {}
        for name, spacing, shape in (
            ("aniso", aniso_spacing, aniso_shape),
            ("iso", iso_spacing, iso_shape),
        ):
            geometry = _Geometry(spacing)
            cells = _sphere_mask(centre, radius, shape, spacing).astype(np.uint32)
            xx = _physical_centres(shape, spacing)[2]
            ramp = np.broadcast_to(xx, shape).astype(np.float64)
            channels = {"dummy": np.zeros(shape, dtype=np.float64), "ramp": ramp}
            records = compute_signed_distance_profiles(
                cells, geometry, "ds", "field0", channels, config
            )
            results[name] = {
                (rec.channel_id, rec.bin_start_um, rec.bin_end_um): rec for rec in records
            }

        for bin_start, bin_end in zip(edges[:-1], edges[1:]):
            vol_a = results["aniso"][("dummy", bin_start, bin_end)].sampled_volume_um3
            vol_i = results["iso"][("dummy", bin_start, bin_end)].sampled_volume_um3
            if vol_a == 0.0 and vol_i == 0.0:
                continue
            assert vol_a == pytest.approx(vol_i, rel=0.05), (bin_start, bin_end)


# ─── shell enrichment scores ────────────────────────────────────────────────


class TestShellEnrichmentScores:
    SHAPE = (20, 60, 60)
    CENTRE = (5.0, 3.0, 3.0)

    def test_uniform_signal_gives_near_zero_enrichment_scores(self):
        geometry = _Geometry(ANISOTROPIC)
        cells = _sphere_mask(self.CENTRE, 2.0, self.SHAPE, ANISOTROPIC).astype(np.uint32)
        signal = np.full(self.SHAPE, 5.0, dtype=np.float64)
        config = LocalizationConfig()

        out = compute_shell_enrichment_scores(cells, geometry, {"sig": signal}, config)
        row = out[1]
        assert row["sig__extracellular_enrichment_score"] == pytest.approx(0.0, abs=1e-9)
        assert row["sig__intracellular_enrichment_score"] == pytest.approx(0.0, abs=1e-9)
        assert row["sig__membrane_enrichment_score"] == pytest.approx(0.0, abs=1e-9)

    def test_bright_core_gives_positive_intracellular_score(self):
        geometry = _Geometry(ANISOTROPIC)
        radius = 2.5
        cells = _sphere_mask(self.CENTRE, radius, self.SHAPE, ANISOTROPIC).astype(
            np.uint32
        )
        r = _radius_field(self.CENTRE, self.SHAPE, ANISOTROPIC)
        core_bright = r <= 1.0  # well inside the default 1.0 um inner shell boundary
        signal = np.full(self.SHAPE, 1.0, dtype=np.float64)
        signal[core_bright] = 50.0
        config = LocalizationConfig()  # inner_shell_width_um default 1.0

        out = compute_shell_enrichment_scores(cells, geometry, {"sig": signal}, config)
        score = out[1]["sig__intracellular_enrichment_score"]
        assert score is not None
        assert score > 1.0

    def test_halo_only_signal_gives_positive_extracellular_score(self):
        geometry = _Geometry(ANISOTROPIC)
        radius = 1.5
        cells = _sphere_mask(self.CENTRE, radius, self.SHAPE, ANISOTROPIC).astype(
            np.uint32
        )
        r = _radius_field(self.CENTRE, self.SHAPE, ANISOTROPIC)
        halo = (r > radius) & (r <= radius + 0.4)
        signal = np.zeros(self.SHAPE, dtype=np.float64)
        signal[halo] = 7.0
        config = LocalizationConfig()

        out = compute_shell_enrichment_scores(
            cells, geometry, {"sig": signal}, config, background_noise_std={"sig": 0.5}
        )
        score = out[1]["sig__extracellular_enrichment_score"]
        assert score is not None
        assert score > 0


# ─── colocalization ─────────────────────────────────────────────────────────


class TestColocalization:
    SHAPE = (4, 10, 10)

    def test_pearson_identical_inverted_and_zero_variance(self):
        geometry = _Geometry(ANISOTROPIC)
        cells = np.ones(self.SHAPE, dtype=np.uint32)
        ramp = np.broadcast_to(
            np.arange(self.SHAPE[2], dtype=np.float64), self.SHAPE
        ).copy()

        pair = ColocalizationPairConfig(
            signal_channel_id="a", reference_channel_id="b", reference_role="organelle_marker"
        )
        config = LocalizationConfig(colocalization_pairs=(pair,))

        out_identical = compute_colocalization(
            cells, geometry, {"a": ramp, "b": ramp.copy()}, config
        )
        assert out_identical[1]["a_vs_b__pearson_r"] == pytest.approx(1.0, abs=1e-9)

        out_inverted = compute_colocalization(cells, geometry, {"a": ramp, "b": -ramp}, config)
        assert out_inverted[1]["a_vs_b__pearson_r"] == pytest.approx(-1.0, abs=1e-9)

        out_const = compute_colocalization(
            cells, geometry, {"a": ramp, "b": np.zeros(self.SHAPE)}, config
        )
        assert out_const[1]["a_vs_b__pearson_r"] is None

    def test_manders_coincident_and_disjoint_binary_regions(self):
        geometry = _Geometry(ANISOTROPIC)
        cells = np.ones(self.SHAPE, dtype=np.uint32)
        region_a = np.zeros(self.SHAPE, dtype=np.float64)
        region_a[:, :5, :] = 1.0

        pair = ColocalizationPairConfig(
            signal_channel_id="a",
            reference_channel_id="b",
            reference_role="organelle_marker",
            signal_threshold=ThresholdSpec(mode="fixed", fixed_value=0.5),
            reference_threshold=ThresholdSpec(mode="fixed", fixed_value=0.5),
        )
        config = LocalizationConfig(colocalization_pairs=(pair,))

        out_coincident = compute_colocalization(
            cells, geometry, {"a": region_a, "b": region_a.copy()}, config
        )
        assert out_coincident[1]["a_vs_b__manders_signal_in_reference"] == pytest.approx(1.0)
        assert out_coincident[1]["a_vs_b__manders_reference_in_signal"] == pytest.approx(1.0)

        region_b_disjoint = np.zeros(self.SHAPE, dtype=np.float64)
        region_b_disjoint[:, 5:, :] = 1.0
        out_disjoint = compute_colocalization(
            cells, geometry, {"a": region_a, "b": region_b_disjoint}, config
        )
        assert out_disjoint[1]["a_vs_b__manders_signal_in_reference"] == pytest.approx(0.0)
        assert out_disjoint[1]["a_vs_b__manders_reference_in_signal"] == pytest.approx(0.0)

    def test_organelle_surface_ring_scores_higher_than_core_and_uniform(self):
        shape = (20, 60, 60)
        geometry = _Geometry(ANISOTROPIC)
        centre = (5.0, 3.0, 3.0)
        cell_radius = 2.5
        organelle_radius = 1.0
        band_width = 0.3

        cells = _sphere_mask(centre, cell_radius, shape, ANISOTROPIC).astype(np.uint32)
        r = _radius_field(centre, shape, ANISOTROPIC)
        organelle_mask = r <= organelle_radius
        marker = np.zeros(shape, dtype=np.float64)
        marker[organelle_mask] = 10.0

        ring_mask = np.abs(r - organelle_radius) <= band_width
        core_mask = r <= (organelle_radius - band_width - 0.2)

        pair = ColocalizationPairConfig(
            signal_channel_id="sig",
            reference_channel_id="marker",
            reference_role="organelle_marker",
            reference_threshold=ThresholdSpec(mode="fixed", fixed_value=5.0),
            organelle_surface_band_um=band_width,
        )
        config = LocalizationConfig(colocalization_pairs=(pair,))
        background_stats = {"sig": (0.0, 0.1)}

        def _score(signal_mask):
            signal = np.full(shape, 1.0, dtype=np.float64)
            signal[signal_mask & (cells == 1)] = 50.0
            out = compute_colocalization(
                cells, geometry, {"sig": signal, "marker": marker}, config, background_stats
            )
            return out[1]["sig_vs_marker__organelle_surface_enrichment_score"]

        ring_score = _score(ring_mask)
        core_score = _score(core_mask)

        uniform_signal = np.full(shape, 5.0, dtype=np.float64)
        out_uniform = compute_colocalization(
            cells,
            geometry,
            {"sig": uniform_signal, "marker": marker},
            config,
            background_stats,
        )
        uniform_score = out_uniform[1]["sig_vs_marker__organelle_surface_enrichment_score"]

        assert ring_score is not None
        assert core_score is not None
        assert uniform_score is not None
        assert ring_score > core_score
        assert ring_score > uniform_score


# ─── threshold application ──────────────────────────────────────────────────


class TestApplyThresholdSpec:
    def test_fixed_mode(self):
        values = np.array([0.0, 1.0, 2.0, 3.0])
        spec = ThresholdSpec(mode="fixed", fixed_value=1.5)
        mask = apply_threshold_spec(values, spec)
        assert mask.tolist() == [False, False, True, True]

    def test_fixed_mode_missing_value_raises(self):
        spec = ThresholdSpec(mode="fixed", fixed_value=None)
        with pytest.raises(MeasurementError):
            apply_threshold_spec(np.array([1.0]), spec)

    def test_background_mad_mode(self):
        spec = ThresholdSpec(mode="background_mad", mad_multiplier=2.0)
        values = np.array([0.0, 5.0, 10.0])
        mask = apply_threshold_spec(
            values, spec, background_median=0.0, background_mad=1.0
        )
        # threshold = 0 + 2 * 1.4826 * 1.0 ~= 2.965
        assert mask.tolist() == [False, True, True]

    def test_background_mad_missing_multiplier_raises(self):
        spec = ThresholdSpec(mode="background_mad", mad_multiplier=None)
        with pytest.raises(MeasurementError):
            apply_threshold_spec(np.array([1.0]), spec)

    def test_unknown_mode_raises(self):
        spec = object.__new__(ThresholdSpec)
        object.__setattr__(spec, "mode", "otsu")
        object.__setattr__(spec, "fixed_value", None)
        object.__setattr__(spec, "mad_multiplier", None)
        with pytest.raises(MeasurementError):
            apply_threshold_spec(np.array([1.0]), spec)


# ─── categorical classifier ─────────────────────────────────────────────────


class TestClassifyLocalization:
    def _inputs(self, **overrides):
        base = dict(
            total_corrected_signal=100.0,
            extracellular_enrichment_score=None,
            intracellular_enrichment_score=None,
            has_nucleus=False,
            nuclear_enrichment=None,
            organelle_surface_enrichment_score=None,
            organelle_manders_signal_in_reference=None,
            qc_passed=True,
        )
        base.update(overrides)
        return LocalizationDecisionInputs(**base)

    def test_disabled_by_default_is_indeterminate_regardless_of_signal_strength(self):
        decision = LocalizationDecisionConfig()
        assert decision.enabled is False
        inputs = self._inputs(
            extracellular_enrichment_score=100.0,
            intracellular_enrichment_score=100.0,
            has_nucleus=True,
            nuclear_enrichment=100.0,
        )
        assert classify_localization(inputs, decision) == "indeterminate"

    def test_single_passing_class_wins(self):
        decision = LocalizationDecisionConfig(enabled=True)
        inputs = self._inputs(extracellular_enrichment_score=5.0)
        assert classify_localization(inputs, decision) == "extracellular_enriched"

    def test_two_passing_without_margin_is_mixed(self):
        decision = LocalizationDecisionConfig(enabled=True, winning_margin=0.5)
        inputs = self._inputs(
            extracellular_enrichment_score=1.2, intracellular_enrichment_score=1.2
        )
        assert classify_localization(inputs, decision) == "mixed"

    def test_two_passing_with_margin_picks_winner(self):
        decision = LocalizationDecisionConfig(enabled=True, winning_margin=0.5)
        inputs = self._inputs(
            extracellular_enrichment_score=5.0, intracellular_enrichment_score=1.1
        )
        assert classify_localization(inputs, decision) == "extracellular_enriched"

    def test_nuclear_class_requires_has_nucleus(self):
        decision = LocalizationDecisionConfig(enabled=True)
        without_nucleus = self._inputs(nuclear_enrichment=10.0, has_nucleus=False)
        assert classify_localization(without_nucleus, decision) == "indeterminate"

        with_nucleus = self._inputs(nuclear_enrichment=10.0, has_nucleus=True)
        assert classify_localization(with_nucleus, decision) == "nuclear_enriched"

    def test_organelle_class_requires_both_surface_and_manders(self):
        decision = LocalizationDecisionConfig(enabled=True)
        surface_only = self._inputs(
            organelle_surface_enrichment_score=5.0,
            organelle_manders_signal_in_reference=0.1,
        )
        assert classify_localization(surface_only, decision) == "indeterminate"

        both = self._inputs(
            organelle_surface_enrichment_score=5.0,
            organelle_manders_signal_in_reference=0.9,
        )
        assert classify_localization(both, decision) == "organelle_surface_associated"

    def test_insufficient_signal_is_indeterminate(self):
        decision = LocalizationDecisionConfig(enabled=True, minimum_corrected_signal=50.0)
        too_weak = self._inputs(
            total_corrected_signal=10.0, extracellular_enrichment_score=10.0
        )
        assert classify_localization(too_weak, decision) == "indeterminate"

        missing = self._inputs(
            total_corrected_signal=None, extracellular_enrichment_score=10.0
        )
        assert classify_localization(missing, decision) == "indeterminate"

    def test_failed_qc_is_indeterminate(self):
        decision = LocalizationDecisionConfig(enabled=True)
        inputs = self._inputs(extracellular_enrichment_score=10.0, qc_passed=False)
        assert classify_localization(inputs, decision) == "indeterminate"

    def test_no_class_passing_is_indeterminate(self):
        decision = LocalizationDecisionConfig(enabled=True)
        inputs = self._inputs(extracellular_enrichment_score=0.1)
        assert classify_localization(inputs, decision) == "indeterminate"


# ─── no ML imports ──────────────────────────────────────────────────────────


def test_importing_both_modules_leaves_torch_and_cellpose_unimported():
    code = (
        "import sys\n"
        "import s2_adhesion.metrics.nuclei\n"
        "import s2_adhesion.metrics.localization\n"
        "assert 'torch' not in sys.modules, sys.modules.keys()\n"
        "assert 'cellpose' not in sys.modules, sys.modules.keys()\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
