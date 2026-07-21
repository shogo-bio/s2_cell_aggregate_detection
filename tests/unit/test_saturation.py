"""Detector-saturation detection and the reliability calls that key off it."""

from __future__ import annotations

import numpy as np
import pytest

from s2_adhesion.contracts import ChannelBinding, ChannelRole, VoxelGeometry
from s2_adhesion.metrics.saturation import (
    LOC_INDETERMINATE,
    LOC_LOWER_BOUND,
    LOC_QUANTITATIVE,
    MATERIAL,
    NONE,
    PATTERN_CORE,
    PATTERN_SHELL_ONLY,
    SEVERE,
    channel_dynamic_range,
    colocalization_trustworthy,
    compute_saturation,
    localization_reliability,
    resolve_detector_max,
)

ANISO = (0.5, 0.1, 0.1)
DMAX = 4095


def _sphere_cell(shape=(24, 100, 100), spacing=ANISO, r_um=4.0):
    zz, yy, xx = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]].astype(float)
    zz *= spacing[0]
    yy *= spacing[1]
    xx *= spacing[2]
    c = np.array([s * sp / 2 for s, sp in zip(shape, spacing)])
    radius = np.sqrt((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2)
    labels = (radius < r_um).astype(np.uint32)
    return labels, radius, r_um


def _bindings(*ids):
    return [ChannelBinding(i, n, frozenset({ChannelRole.SIGNAL})) for n, i in enumerate(ids)]


class TestDetectorMax:
    def test_config_value_is_used(self):
        data = np.array([100, 200, 4095], np.uint16)
        assert resolve_detector_max(data, 4095) == (4095, "config")

    def test_config_below_observed_max_is_an_error(self):
        data = np.array([100, 5000], np.uint16)
        with pytest.raises(ValueError, match="set wrong"):
            resolve_detector_max(data, 4095)

    def test_auto_detects_a_known_ceiling(self):
        data = np.array([0, 100, 4095], np.uint16)
        value, prov = resolve_detector_max(data, None)
        assert value == 4095
        assert prov == "auto_detected_at_known_ceiling"

    def test_unsaturated_data_falls_back_uncertainly(self):
        data = np.array([0, 100, 3000], np.uint16)
        value, prov = resolve_detector_max(data, None)
        assert value == 3000
        assert "uncertain" in prov


class TestSaturationDetection:
    def test_a_clean_cell_reads_none(self):
        labels, radius, r = _sphere_cell()
        chan = np.full(labels.shape, 500, np.uint16)  # nowhere near 4095
        out = compute_saturation(labels, {"c0": chan}, _bindings("c0"),
                                 VoxelGeometry(spacing_um_zyx=ANISO), DMAX)
        row = out[1]
        assert row["ch.c0.saturation_status"] == NONE
        assert row["ch.c0.saturated_fraction_cell"] == 0.0
        assert row["ch.c0.intensity_is_lower_bound"] is False
        assert localization_reliability(row, "c0") == LOC_QUANTITATIVE

    def test_modest_shell_saturation_is_shell_pattern_and_survives_localization(self):
        """A partly-clipped rim only biases the ring score downward -- still a lower
        bound, so a conservative 'shell-enriched' call stays allowed.

        A FULLY clipped rim is a different case (severe) and is covered separately:
        this uses a small polar patch so the cell fraction stays below the severe
        threshold.
        """
        labels, radius, r = _sphere_cell()
        zz = np.mgrid[0 : labels.shape[0], 0 : labels.shape[1], 0 : labels.shape[2]][0]
        cz = labels.shape[0] // 2
        chan = np.full(labels.shape, 300, np.uint16)
        # clip only a thin equatorial band of the rim -> small cell fraction
        rim = (labels == 1) & (radius >= r - 0.4) & (np.abs(zz - cz) <= 1)
        chan[rim] = DMAX
        out = compute_saturation(labels, {"c0": chan}, _bindings("c0"),
                                 VoxelGeometry(spacing_um_zyx=ANISO), DMAX)
        row = out[1]
        assert row["ch.c0.saturation_pattern"] == PATTERN_SHELL_ONLY
        assert row["ch.c0.saturated_fraction_shell"] > 0
        assert row["ch.c0.saturated_fraction_core"] == 0.0
        # Shell clipping (core clean) makes the score a valid lower bound, not
        # quantitative and not forbidden.
        assert localization_reliability(row, "c0") == LOC_LOWER_BOUND

    def test_a_saturated_core_forbids_localization(self):
        """A clipped core can masquerade as uniform signal -- must not be trusted."""
        labels, radius, r = _sphere_cell()
        chan = np.full(labels.shape, 300, np.uint16)
        chan[(labels == 1) & (radius < r - 1.5)] = DMAX  # clip the interior
        out = compute_saturation(labels, {"c0": chan}, _bindings("c0"),
                                 VoxelGeometry(spacing_um_zyx=ANISO), DMAX)
        row = out[1]
        assert row["ch.c0.saturation_pattern"] == PATTERN_CORE
        assert localization_reliability(row, "c0") == LOC_INDETERMINATE

    def test_severe_saturation_forbids_localization(self):
        labels, radius, r = _sphere_cell()
        chan = np.full(labels.shape, DMAX, np.uint16)  # whole cell clipped
        out = compute_saturation(labels, {"c0": chan}, _bindings("c0"),
                                 VoxelGeometry(spacing_um_zyx=ANISO), DMAX)
        row = out[1]
        assert row["ch.c0.saturation_status"] == SEVERE
        assert row["ch.c0.intensity_is_lower_bound"] is True
        assert localization_reliability(row, "c0") == LOC_INDETERMINATE

    def test_ignored_channels_are_skipped(self):
        labels, _, _ = _sphere_cell()
        chan = np.full(labels.shape, DMAX, np.uint16)
        bindings = [ChannelBinding("c0", 0, frozenset({ChannelRole.IGNORE}))]
        out = compute_saturation(labels, {"c0": chan}, bindings,
                                 VoxelGeometry(spacing_um_zyx=ANISO), DMAX)
        assert not any("c0" in k for k in out[1])

    def test_measured_on_raw_values_not_background_subtracted(self):
        """Only exact-ceiling voxels count; background subtraction would hide them."""
        labels, radius, r = _sphere_cell()
        chan = np.full(labels.shape, 300, np.uint16)
        chan[(labels == 1) & (radius >= r - 0.9)] = DMAX
        # 4094 (one below ceiling) must NOT count as saturated.
        chan2 = np.full(labels.shape, 300, np.uint16)
        chan2[(labels == 1) & (radius >= r - 0.9)] = DMAX - 1
        g = VoxelGeometry(spacing_um_zyx=ANISO)
        sat = compute_saturation(labels, {"c0": chan}, _bindings("c0"), g, DMAX)[1]
        clean = compute_saturation(labels, {"c0": chan2}, _bindings("c0"), g, DMAX)[1]
        assert sat["ch.c0.saturated_fraction_shell"] > 0
        assert clean["ch.c0.saturated_fraction_shell"] == 0.0


class TestColocalizationGate:
    def test_material_saturation_in_either_channel_nulls_the_pair(self):
        row = {
            "ch.a.saturation_status": MATERIAL,
            "ch.b.saturation_status": NONE,
        }
        assert colocalization_trustworthy(row, "a", "b") is False
        assert colocalization_trustworthy(row, "b", "b") is True


class TestDynamicRange:
    def test_under_exposed_channel_has_low_occupancy(self):
        labels, _, _ = _sphere_cell()
        # mimic mCherry: signal at ~900, detector max 4095 -> ~22%
        chan = np.full(labels.shape, 5, np.uint16)
        chan[labels == 1] = 900
        dr = channel_dynamic_range(chan, labels, DMAX)
        assert dr["range_occupancy"] < 0.25

    def test_well_exposed_channel_has_high_occupancy(self):
        labels, _, _ = _sphere_cell()
        chan = np.full(labels.shape, 5, np.uint16)
        chan[labels == 1] = 3500
        dr = channel_dynamic_range(chan, labels, DMAX)
        assert dr["range_occupancy"] > 0.5


def test_no_ml_imports():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, s2_adhesion.metrics.saturation as m; "
            "leaked=[k for k in sys.modules if k.split('.')[0] in ('torch','cellpose')]; "
            "print(leaked); sys.exit(1 if leaked else 0)",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout
