"""Tests for the cellpose pre/post-processing grid logic.

No ML packages needed here: everything under test operates on plain
numpy arrays and the frozen contracts/config types. Resampling factors are
chosen to be exact integers (2x, 4x) so nearest-neighbour round trips can be
checked with plain array equality rather than fuzzy tolerances.
"""

from __future__ import annotations

import numpy as np
import pytest

from s2_adhesion.config import CellposeModelConfig, DirectInstanceConfig, NormalizationConfig
from s2_adhesion.contracts import ChannelRole, ContractViolation
from s2_adhesion.segmentation.preprocess import (
    map_labels_to_original_grid,
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
