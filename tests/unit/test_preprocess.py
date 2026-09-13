"""Tests for the cellpose pre/post-processing grid logic.

No ML packages needed here: everything under test operates on plain
numpy arrays and the frozen contracts/config types. Resampling factors are
chosen to be exact integers (2x, 4x) so nearest-neighbour round trips can be
checked with plain array equality rather than fuzzy tolerances.
"""

from __future__ import annotations

import numpy as np
import pytest

from s2_adhesion.config import (
    CellposeModelConfig,
    ChannelNormalizationOverride,
    DirectInstanceConfig,
    NormalizationConfig,
)
from s2_adhesion.contracts import ChannelRole, ContractViolation
from s2_adhesion.segmentation.preprocess import (
    combine_channels,
    combined_channel_id,
    map_labels_to_original_grid,
    normalization_bounds,
    prepare_cellpose_input,
)
from s2_adhesion.segmentation.protocol import PreparedCellposeInput
from tests.conftest import ANISOTROPIC, make_image_volume

# spacing with dy != dx, so in-plane resampling is exercised: target=min(dy,dx)=0.1,
# scale_y = 0.2/0.1 = 2 (exact), scale_x = 0.1/0.1 = 1 (no-op).
NONSQUARE_INPLANE = (0.5, 0.2, 0.1)


class TestPrepareCellposeInput:
    def test_resamples_inplane_to_square_pixels(self):
        data = np.random.default_rng(0).random((1, 4, 5, 10)).astype(np.float32)
        image = make_image_volume(
            data, spacing=NONSQUARE_INPLANE, channel_ids=("membrane",),
            roles=(ChannelRole.MEMBRANE,),
        )
        config = DirectInstanceConfig(strategy="direct_cellpose", input_channel_ids=("membrane",))

        prepared = prepare_cellpose_input(image, config)

        # dy=0.2 -> scale 2, Y 5->10; dx=0.1 -> scale 1, X stays 10.
        assert prepared.data.shape == (1, 4, 10, 10)
        assert prepared.resampled_spacing_um_zyx[1] == pytest.approx(0.1)
        assert prepared.resampled_spacing_um_zyx[2] == pytest.approx(0.1)
        # square in-plane pixel: resampled dy == resampled dx.
        assert prepared.resampled_spacing_um_zyx[1] == prepared.resampled_spacing_um_zyx[2]
        # Z is never resampled.
        assert prepared.resampled_spacing_um_zyx[0] == pytest.approx(0.5)

    def test_leaves_isotropic_inplane_spacing_untouched(self):
        data = np.random.default_rng(1).random((1, 4, 6, 6)).astype(np.float32)
        image = make_image_volume(
            data, spacing=ANISOTROPIC, channel_ids=("membrane",), roles=(ChannelRole.MEMBRANE,)
        )
        config = DirectInstanceConfig(strategy="direct_cellpose", input_channel_ids=("membrane",))

        prepared = prepare_cellpose_input(image, config)

        # ANISOTROPIC = (0.5, 0.1, 0.1): dy == dx already, no in-plane resampling.
        assert prepared.data.shape == (1, 4, 6, 6)

    def test_anisotropy_equals_dz_over_min_dy_dx(self):
        data = np.random.default_rng(2).random((1, 3, 5, 10)).astype(np.float32)
        image = make_image_volume(
            data, spacing=NONSQUARE_INPLANE, channel_ids=("membrane",), roles=(ChannelRole.MEMBRANE,)
        )
        config = DirectInstanceConfig(strategy="direct_cellpose", input_channel_ids=("membrane",))

        prepared = prepare_cellpose_input(image, config)

        dz, dy, dx = NONSQUARE_INPLANE
        assert prepared.anisotropy == pytest.approx(dz / min(dy, dx))
        assert prepared.anisotropy == pytest.approx(image.geometry.anisotropy_z_to_xy)

    def test_selects_configured_channels_in_order_not_source_order(self):
        rng = np.random.default_rng(3)
        # source order: membrane(0), signal(1), nucleus(2); request nucleus then membrane.
        data = rng.random((3, 2, 4, 4)).astype(np.float32)
        image = make_image_volume(data, spacing=ANISOTROPIC)
        config = DirectInstanceConfig(
            strategy="direct_cellpose", input_channel_ids=("nucleus", "membrane")
        )

        prepared = prepare_cellpose_input(image, config)

        assert prepared.channel_ids == ("nucleus", "membrane")
        assert prepared.data.shape[0] == 2

    def test_diameter_um_converted_to_resampled_pixels(self):
        data = np.random.default_rng(4).random((1, 3, 5, 10)).astype(np.float32)
        image = make_image_volume(
            data, spacing=NONSQUARE_INPLANE, channel_ids=("membrane",), roles=(ChannelRole.MEMBRANE,)
        )
        config = DirectInstanceConfig(
            strategy="direct_cellpose",
            input_channel_ids=("membrane",),
            model=CellposeModelConfig(diameter_um=2.0),
        )

        prepared = prepare_cellpose_input(image, config)

        # target in-plane spacing is min(dy, dx) = 0.1 um/px.
        assert prepared.diameter_px == pytest.approx(2.0 / 0.1)

    def test_diameter_none_stays_none(self):
        data = np.random.default_rng(5).random((1, 3, 5, 10)).astype(np.float32)
        image = make_image_volume(
            data, spacing=NONSQUARE_INPLANE, channel_ids=("membrane",), roles=(ChannelRole.MEMBRANE,)
        )
        config = DirectInstanceConfig(
            strategy="direct_cellpose",
            input_channel_ids=("membrane",),
            model=CellposeModelConfig(diameter_um=None),
        )

        prepared = prepare_cellpose_input(image, config)
        assert prepared.diameter_px is None

    def test_normalizes_each_channel_independently_to_unit_range(self):
        rng = np.random.default_rng(6)
        # very different intensity scales per channel
        membrane = rng.uniform(0, 100, size=(2, 4, 4)).astype(np.float32)
        signal = rng.uniform(1000, 5000, size=(2, 4, 4)).astype(np.float32)
        data = np.stack([membrane, signal], axis=0)
        image = make_image_volume(data, spacing=ANISOTROPIC)
        config = DirectInstanceConfig(
            strategy="direct_cellpose",
            input_channel_ids=("membrane", "signal"),
            model=CellposeModelConfig(normalization=NormalizationConfig(clip=True)),
        )

        prepared = prepare_cellpose_input(image, config)

        assert prepared.data.min() >= 0.0
        assert prepared.data.max() <= 1.0 + 1e-6

    def test_normalization_of_flat_channel_is_all_zero_not_nan(self):
        data = np.full((1, 2, 4, 4), 7.0, dtype=np.float32)
        image = make_image_volume(
            data, spacing=ANISOTROPIC, channel_ids=("membrane",), roles=(ChannelRole.MEMBRANE,)
        )
        config = DirectInstanceConfig(strategy="direct_cellpose", input_channel_ids=("membrane",))

        prepared = prepare_cellpose_input(image, config)

        assert not np.isnan(prepared.data).any()
        assert np.all(prepared.data == 0.0)


def _two_population_image(rng):
    """green-only blob at Y 1..3, red-only blob at Y 6..8, on a Z=3, 10x10 grid.

    Red is ten times dimmer in raw counts than green, so a merge that did not
    normalise each channel first would nearly erase the red cell.
    """
    green = rng.uniform(0, 20, size=(3, 10, 10)).astype(np.float32)
    red = rng.uniform(0, 2, size=(3, 10, 10)).astype(np.float32)
    green[:, 1:4, 1:4] = 1000.0
    red[:, 6:9, 6:9] = 100.0
    data = np.stack([green, red], axis=0)
    return make_image_volume(
        data, spacing=ANISOTROPIC, channel_ids=("green", "red"),
        roles=(ChannelRole.SIGNAL, ChannelRole.SIGNAL),
    )


class TestChannelCombination:
    def test_default_is_stack_and_keeps_two_channels(self):
        image = _two_population_image(np.random.default_rng(10))
        config = DirectInstanceConfig(
            strategy="direct_cellpose", input_channel_ids=("green", "red")
        )
        assert config.channel_combination == "stack"

        prepared = prepare_cellpose_input(image, config)

        assert prepared.data.shape == (2, 3, 10, 10)
        assert prepared.channel_ids == ("green", "red")

    def test_max_yields_single_channel_with_synthetic_id(self):
        image = _two_population_image(np.random.default_rng(11))
        config = DirectInstanceConfig(
            strategy="direct_cellpose",
            input_channel_ids=("green", "red"),
            channel_combination="max",
        )

        prepared = prepare_cellpose_input(image, config)

        assert prepared.data.shape == (1, 3, 10, 10)
        assert prepared.data.dtype == np.float32
        assert prepared.channel_ids == ("max(green,red)",)

    def test_max_keeps_inplane_resampling(self):
        rng = np.random.default_rng(12)
        data = rng.random((2, 4, 5, 10)).astype(np.float32)
        image = make_image_volume(
            data, spacing=NONSQUARE_INPLANE, channel_ids=("green", "red"),
            roles=(ChannelRole.SIGNAL, ChannelRole.SIGNAL),
        )
        config = DirectInstanceConfig(
            strategy="direct_cellpose",
            input_channel_ids=("green", "red"),
            channel_combination="max",
        )

        prepared = prepare_cellpose_input(image, config)

        # Y 5 -> 10 (scale 2), X stays 10, one merged channel.
        assert prepared.data.shape == (1, 4, 10, 10)

    def test_red_only_cell_appears_in_max_image_at_full_intensity(self):
        """The bug this exists for: a cell bright only in the dim channel
        must reach cellpose as bright as a cell in the bright channel."""
        image = _two_population_image(np.random.default_rng(13))
        config = DirectInstanceConfig(
            strategy="direct_cellpose",
            input_channel_ids=("green", "red"),
            channel_combination="max",
            model=CellposeModelConfig(normalization=NormalizationConfig(clip=True)),
        )

        merged = prepare_cellpose_input(image, config).data[0]

        green_blob = merged[:, 1:4, 1:4]
        red_blob = merged[:, 6:9, 6:9]
        background = merged[:, 4:6, :]
        assert green_blob.min() == pytest.approx(1.0)
        assert red_blob.min() == pytest.approx(1.0)
        assert background.max() < 0.2

    def test_max_normalises_each_channel_independently(self):
        """With a shared (not per-channel) normalisation the dim red channel
        would top out near 0.1; independently normalised it reaches 1."""
        image = _two_population_image(np.random.default_rng(14))
        config = DirectInstanceConfig(
            strategy="direct_cellpose",
            input_channel_ids=("green", "red"),
            channel_combination="max",
        )
        merged = prepare_cellpose_input(image, config).data[0]
        # red raw max is 100, green raw max is 1000; only per-channel
        # normalisation makes the red blob as bright as the green one.
        assert merged[:, 6:9, 6:9].min() == pytest.approx(merged[:, 1:4, 1:4].min())

    def test_sum_is_renormalised_to_unit_range(self):
        image = _two_population_image(np.random.default_rng(15))
        config = DirectInstanceConfig(
            strategy="direct_cellpose",
            input_channel_ids=("green", "red"),
            channel_combination="sum",
        )

        prepared = prepare_cellpose_input(image, config)

        assert prepared.data.shape == (1, 3, 10, 10)
        assert prepared.channel_ids == ("sum(green,red)",)
        assert prepared.data.min() >= 0.0
        assert prepared.data.max() <= 1.0 + 1e-6
        # both blobs survive the merge
        merged = prepared.data[0]
        assert merged[:, 1:4, 1:4].min() > 0.5
        assert merged[:, 6:9, 6:9].min() > 0.5

    def test_combine_channels_max_is_voxelwise_maximum(self):
        a = np.zeros((2, 3, 4), dtype=np.float32)
        b = np.zeros((2, 3, 4), dtype=np.float32)
        a[0] = 0.7
        b[1] = 0.4
        merged = combine_channels(np.stack([a, b]), "max", NormalizationConfig())
        assert merged.shape == (1, 2, 3, 4)
        np.testing.assert_allclose(merged[0, 0], 0.7)
        np.testing.assert_allclose(merged[0, 1], 0.4)

    def test_combine_channels_rejects_stack_and_bad_shape(self):
        stacked = np.zeros((2, 2, 3, 4), dtype=np.float32)
        with pytest.raises(ContractViolation):
            combine_channels(stacked, "stack", NormalizationConfig())
        with pytest.raises(ContractViolation):
            combine_channels(stacked[0], "max", NormalizationConfig())

    def test_combined_channel_id_names_mode_and_inputs(self):
        assert combined_channel_id("max", ("green", "red")) == "max(green,red)"
        assert combined_channel_id("sum", ("a", "b", "c")) == "sum(a,b,c)"


class TestMapLabelsToOriginalGrid:
    def _prepared(self, original_shape_zyx, resampled_spacing=(0.5, 0.1, 0.1)):
        z, y, x = original_shape_zyx
        return PreparedCellposeInput(
            data=np.zeros((1, z, y * 2, x), dtype=np.float32),
            channel_ids=("membrane",),
            anisotropy=5.0,
            original_shape_zyx=original_shape_zyx,
            resampled_spacing_um_zyx=resampled_spacing,
            diameter_px=None,
        )

    def test_maps_back_to_exact_original_shape(self):
        original_shape = (3, 5, 10)
        prepared = self._prepared(original_shape)
        model_labels = np.zeros((3, 10, 10), dtype=np.uint32)

        mapped = map_labels_to_original_grid(model_labels, prepared)

        assert mapped.shape == original_shape
        assert mapped.dtype == np.uint32

    def test_round_trip_preserves_label_ids_exactly(self):
        # Original grid: Z=3, Y=5, X=10, with a distinct label per Y row (1..5).
        original = np.zeros((3, 5, 10), dtype=np.uint32)
        for y in range(5):
            original[:, y, :] = y + 1

        # Simulate what a model would see after 2x nearest-neighbour upsampling
        # of Y (matching NONSQUARE_INPLANE's scale_y=2): each row duplicated.
        upsampled = np.repeat(original, 2, axis=1)
        assert upsampled.shape == (3, 10, 10)

        prepared = self._prepared((3, 5, 10))
        mapped = map_labels_to_original_grid(upsampled, prepared)

        assert mapped.shape == original.shape
        np.testing.assert_array_equal(mapped, original)
        assert set(np.unique(mapped).tolist()) == set(np.unique(original).tolist())

    def test_raises_on_z_mismatch(self):
        prepared = self._prepared((3, 5, 10))
        wrong_z_labels = np.zeros((4, 10, 10), dtype=np.uint32)

        with pytest.raises(ContractViolation):
            map_labels_to_original_grid(wrong_z_labels, prepared)

    def test_noop_when_already_square_inplane(self):
        original_shape = (2, 6, 6)
        prepared = PreparedCellposeInput(
            data=np.zeros((1, 2, 6, 6), dtype=np.float32),
            channel_ids=("membrane",),
            anisotropy=5.0,
            original_shape_zyx=original_shape,
            resampled_spacing_um_zyx=(0.5, 0.1, 0.1),
            diameter_px=None,
        )
        model_labels = np.arange(2 * 6 * 6, dtype=np.uint32).reshape(2, 6, 6) % 5

        mapped = map_labels_to_original_grid(model_labels, prepared)

        np.testing.assert_array_equal(mapped, model_labels)


class TestPerChannelNormalization:
    """``normalization_by_channel``: one channel may depart from the common
    percentile rule -- here, an absolute upper bound for a sparse population
    whose whole-stack percentiles measure cell count rather than brightness."""

    @staticmethod
    def _config(overrides=(), combination="max", common=None):
        model = CellposeModelConfig(
            normalization=common or NormalizationConfig(),
            normalization_by_channel=tuple(overrides),
        )
        return DirectInstanceConfig(
            strategy="direct_cellpose",
            input_channel_ids=("green", "red"),
            channel_combination=combination,
            model=model,
        )

    def test_no_overrides_is_byte_identical_to_the_common_rule(self):
        image = _two_population_image(np.random.default_rng(20))
        plain = prepare_cellpose_input(image, self._config()).data
        empty = prepare_cellpose_input(image, self._config(overrides=())).data
        np.testing.assert_array_equal(plain, empty)

    def test_absolute_upper_value_applies_only_to_the_named_channel(self):
        """Red blob is 100 counts. With upper_value=200 it lands at ~0.5,
        while the green blob (percentile rule, untouched) still reaches 1."""
        image = _two_population_image(np.random.default_rng(21))
        config = self._config(
            overrides=[ChannelNormalizationOverride(channel_id="red", upper_value=200.0)],
            combination="stack",
        )
        prepared = prepare_cellpose_input(image, config).data
        green, red = prepared[0], prepared[1]
        assert green[:, 1:4, 1:4].min() == pytest.approx(1.0)
        # (100 - p1) / (200 - p1) with p1 of the red channel ~ 0..2
        assert 0.48 <= red[:, 6:9, 6:9].mean() <= 0.51
        assert red[:, 0:1, :].max() < 0.05

    def test_override_leaves_the_other_channel_array_unchanged(self):
        image = _two_population_image(np.random.default_rng(22))
        plain = prepare_cellpose_input(image, self._config(combination="stack")).data
        with_red = prepare_cellpose_input(
            image,
            self._config(
                overrides=[ChannelNormalizationOverride(channel_id="red", upper_value=50.0)],
                combination="stack",
            ),
        ).data
        np.testing.assert_array_equal(plain[0], with_red[0])
        assert not np.array_equal(plain[1], with_red[1])

    def test_override_is_matched_by_id_not_by_position(self):
        image = _two_population_image(np.random.default_rng(23))
        override = ChannelNormalizationOverride(channel_id="red", upper_value=200.0)
        model = CellposeModelConfig(normalization_by_channel=(override,))
        swapped = DirectInstanceConfig(
            strategy="direct_cellpose",
            input_channel_ids=("red", "green"),
            channel_combination="stack",
            model=model,
        )
        prepared = prepare_cellpose_input(image, swapped).data
        red, green = prepared[0], prepared[1]
        assert 0.48 <= red[:, 6:9, 6:9].mean() <= 0.51
        assert green[:, 1:4, 1:4].min() == pytest.approx(1.0)

    def test_values_above_the_absolute_upper_bound_are_clipped(self):
        image = _two_population_image(np.random.default_rng(24))
        config = self._config(
            overrides=[ChannelNormalizationOverride(channel_id="red", upper_value=50.0)],
            combination="stack",
        )
        red = prepare_cellpose_input(image, config).data[1]
        assert red.max() == pytest.approx(1.0)
        assert red[:, 6:9, 6:9].min() == pytest.approx(1.0)

    def test_normalization_bounds_reports_the_absolute_hi(self):
        data = np.arange(1000, dtype=np.float32).reshape(1, 10, 100)
        lo, hi = normalization_bounds(data, NormalizationConfig(upper_value=1000.0))
        assert hi == 1000.0
        assert lo == pytest.approx(np.percentile(data, 1.0))
        lo2, hi2 = normalization_bounds(data, NormalizationConfig())
        assert hi2 == pytest.approx(np.percentile(data, 99.0))
        assert lo2 == lo

    def test_absolute_bound_below_background_yields_zeros_not_nan(self):
        data = np.full((1, 1, 4, 4), 500.0, dtype=np.float32)
        image = make_image_volume(
            data, spacing=ANISOTROPIC, channel_ids=("membrane",),
            roles=(ChannelRole.MEMBRANE,),
        )
        config = DirectInstanceConfig(
            strategy="direct_cellpose",
            input_channel_ids=("membrane",),
            model=CellposeModelConfig(
                normalization_by_channel=(
                    ChannelNormalizationOverride(channel_id="membrane", upper_value=10.0),
                )
            ),
        )
        prepared = prepare_cellpose_input(image, config)
        assert np.all(prepared.data == 0.0)
        assert not np.isnan(prepared.data).any()


class TestChannelNormalizationOverrideResolve:
    def test_unset_fields_inherit_the_common_config(self):
        base = NormalizationConfig(lower_percentile=2.0, upper_percentile=98.0, clip=False)
        resolved = ChannelNormalizationOverride(channel_id="red", upper_value=1000.0).resolve(base)
        assert resolved == NormalizationConfig(
            lower_percentile=2.0, upper_percentile=98.0, clip=False, upper_value=1000.0
        )

    def test_explicit_percentile_replaces_an_inherited_absolute_value(self):
        base = NormalizationConfig(upper_value=1000.0)
        resolved = ChannelNormalizationOverride(channel_id="red", upper_percentile=99.9).resolve(base)
        assert resolved.upper_value is None
        assert resolved.upper_percentile == 99.9

    def test_lower_and_clip_override_independently(self):
        base = NormalizationConfig()
        resolved = ChannelNormalizationOverride(
            channel_id="red", lower_percentile=5.0, clip=False
        ).resolve(base)
        assert resolved == NormalizationConfig(lower_percentile=5.0, upper_percentile=99.0, clip=False)

    def test_normalization_for_falls_back_to_the_common_config(self):
        model = CellposeModelConfig(
            normalization_by_channel=(
                ChannelNormalizationOverride(channel_id="red", upper_value=1000.0),
            )
        )
        assert model.normalization_for("green") == model.normalization
        assert model.normalization_for("red").upper_value == 1000.0


def test_importing_preprocess_does_not_import_torch_or_cellpose():
    import subprocess
    import sys
    from pathlib import Path

    code = (
        "import sys\n"
        "import s2_adhesion.segmentation.preprocess\n"
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
