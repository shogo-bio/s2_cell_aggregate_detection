"""Tests for the frozen canonical contracts.

These guard the conventions every other module assumes. If one of these fails,
the failure is in the foundation, not in the module that surfaced it.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from s2_adhesion.contracts import (
    ChannelBinding,
    ChannelRole,
    ContactEstimator,
    ContractViolation,
    ImageVolume,
    LabelVolume,
    VoxelGeometry,
)
from tests.conftest import ANISOTROPIC, make_identity, make_provenance


class TestVoxelGeometry:
    def test_voxel_volume_is_product_of_spacing(self):
        g = VoxelGeometry(spacing_um_zyx=(0.5, 0.1, 0.2))
        assert g.voxel_volume_um3 == pytest.approx(0.5 * 0.1 * 0.2)

    def test_face_areas_are_perpendicular_to_each_axis(self):
        dz, dy, dx = 0.5, 0.1, 0.2
        g = VoxelGeometry(spacing_um_zyx=(dz, dy, dx))
        az, ay, ax = g.face_area_um2_zyx
        assert az == pytest.approx(dy * dx)
        assert ay == pytest.approx(dz * dx)
        assert ax == pytest.approx(dz * dy)

    def test_anisotropy_uses_the_finer_in_plane_spacing(self):
        g = VoxelGeometry(spacing_um_zyx=(0.5, 0.1, 0.2))
        assert g.anisotropy_z_to_xy == pytest.approx(5.0)

    def test_isotropic_detection(self):
        assert VoxelGeometry(spacing_um_zyx=(0.1, 0.1, 0.1)).is_isotropic
        assert not VoxelGeometry(spacing_um_zyx=ANISOTROPIC).is_isotropic

    @pytest.mark.parametrize("bad", [(0.0, 0.1, 0.1), (-0.5, 0.1, 0.1), (0.5, 0.0, 0.1)])
    def test_non_positive_spacing_is_rejected(self, bad):
        with pytest.raises(ContractViolation, match="strictly positive"):
            VoxelGeometry(spacing_um_zyx=bad)

    def test_index_to_um_uses_voxel_centres(self):
        g = VoxelGeometry(spacing_um_zyx=(0.5, 0.1, 0.1))
        # index 0 sits at half a voxel, not at the origin
        got = g.index_to_um(np.array([0.0, 0.0, 0.0]))
        assert got == pytest.approx([0.25, 0.05, 0.05])


class TestImageVolume:
    def _vol(self, n_c=3, shape=(4, 8, 8)):
        data = np.zeros((n_c, *shape), np.uint16)
        channels = tuple(
            ChannelBinding(f"ch{i}", i, frozenset({ChannelRole.SIGNAL}))
            for i in range(n_c)
        )
        return ImageVolume(
            data=data,
            geometry=VoxelGeometry(spacing_um_zyx=ANISOTROPIC),
            channels=channels,
            identity=make_identity(),
        )

    def test_rejects_non_4d_data(self):
        with pytest.raises(ContractViolation, match="4-D CZYX"):
            ImageVolume(
                data=np.zeros((4, 8, 8), np.uint16),
                geometry=VoxelGeometry(spacing_um_zyx=ANISOTROPIC),
                channels=(ChannelBinding("a", 0, frozenset()),),
                identity=make_identity(),
            )

    def test_rejects_duplicate_channel_id(self):
        with pytest.raises(ContractViolation, match="duplicate channel_id"):
            ImageVolume(
                data=np.zeros((2, 4, 8, 8), np.uint16),
                geometry=VoxelGeometry(spacing_um_zyx=ANISOTROPIC),
                channels=(
                    ChannelBinding("same", 0, frozenset()),
                    ChannelBinding("same", 1, frozenset()),
                ),
                identity=make_identity(),
            )

    def test_rejects_channel_index_past_end_of_array(self):
        with pytest.raises(ContractViolation, match="only 2 channels"):
            ImageVolume(
                data=np.zeros((2, 4, 8, 8), np.uint16),
                geometry=VoxelGeometry(spacing_um_zyx=ANISOTROPIC),
                channels=(ChannelBinding("a", 5, frozenset()),),
                identity=make_identity(),
            )

    def test_channel_lookup_is_by_logical_id_not_index(self):
        vol = self._vol()
        vol.data[1] = 7
        assert vol.channel("ch1").max() == 7

    def test_unknown_channel_error_lists_the_known_ones(self):
        with pytest.raises(ContractViolation, match=r"ch0.*ch1.*ch2"):
            self._vol().channel("nucleus")

    def test_channels_with_role(self):
        data = np.zeros((2, 4, 8, 8), np.uint16)
        vol = ImageVolume(
            data=data,
            geometry=VoxelGeometry(spacing_um_zyx=ANISOTROPIC),
            channels=(
                ChannelBinding("n", 0, frozenset({ChannelRole.NUCLEUS})),
                ChannelBinding("s", 1, frozenset({ChannelRole.SIGNAL})),
            ),
            identity=make_identity(),
        )
        assert [c.channel_id for c in vol.channels_with_role(ChannelRole.NUCLEUS)] == ["n"]


class TestLabelVolume:
    def _labels(self, arr=None, spacing=ANISOTROPIC):
        arr = np.zeros((4, 8, 8), np.uint32) if arr is None else arr
        ident = make_identity(content=b"image-bytes")
        return LabelVolume(
            cells=arr,
            geometry=VoxelGeometry(spacing_um_zyx=spacing),
            identity=ident,
            provenance=make_provenance(ident.image_content_sha256),
        )

    def test_rejects_non_uint32_labels(self):
        with pytest.raises(ContractViolation, match="silently merge"):
            self._labels(np.zeros((4, 8, 8), np.uint16))

    def test_rejects_non_3d_labels(self):
        with pytest.raises(ContractViolation, match="3-D ZYX"):
            self._labels(np.zeros((2, 4, 8, 8), np.uint32))

    def test_label_ids_are_not_renumbered(self):
        arr = np.zeros((4, 8, 8), np.uint32)
        arr[0, 0, 0] = 1
        arr[0, 0, 1] = 900
        arr[0, 0, 2] = 5
        assert list(self._labels(arr).cell_ids()) == [1, 5, 900]

    def test_validate_against_none_is_allowed_for_geometry_only_measurement(self):
        self._labels().validate_against(None)  # must not raise

    def test_shape_mismatch_is_rejected(self):
        lab = self._labels()
        img = ImageVolume(
            data=np.zeros((1, 4, 8, 16), np.uint16),
            geometry=VoxelGeometry(spacing_um_zyx=ANISOTROPIC),
            channels=(ChannelBinding("a", 0, frozenset()),),
            identity=make_identity(content=b"image-bytes"),
        )
        with pytest.raises(ContractViolation, match="label shape"):
            lab.validate_against(img)

    def test_spacing_mismatch_is_rejected(self):
        lab = self._labels(spacing=(0.5, 0.1, 0.1))
        img = ImageVolume(
            data=np.zeros((1, 4, 8, 8), np.uint16),
            geometry=VoxelGeometry(spacing_um_zyx=(0.25, 0.1, 0.1)),
            channels=(ChannelBinding("a", 0, frozenset()),),
            identity=make_identity(content=b"image-bytes"),
        )
        with pytest.raises(ContractViolation, match="spacing mismatch"):
            lab.validate_against(img)

    def test_labels_from_a_different_image_are_rejected(self):
        """The guard that stops labels being measured against the wrong image."""
        lab = self._labels()
        img = ImageVolume(
            data=np.zeros((1, 4, 8, 8), np.uint16),
            geometry=VoxelGeometry(spacing_um_zyx=ANISOTROPIC),
            channels=(ChannelBinding("a", 0, frozenset()),),
            identity=make_identity(content=b"a-DIFFERENT-image"),
        )
        with pytest.raises(ContractViolation, match="different image"):
            lab.validate_against(img)


class TestContactEstimator:
    def test_marching_cubes_is_the_default_choice(self):
        """face_count is exact only for axis-aligned interfaces; see the docstring."""
        from s2_adhesion.config import ContactConfig

        assert ContactConfig().estimator is ContactEstimator.MARCHING_CUBES

    def test_docstring_records_the_measured_orientation_bias(self):
        """The docstring is the only place a reader learns the metric's limits.

        It must carry both the isotropic comparison that justifies choosing
        marching cubes AND the anisotropic numbers for the real acquisition
        condition -- quoting only the flattering isotropic figures would let
        someone report a contact area as if it were trustworthy at 5x anisotropy.
        """
        doc = ContactEstimator.__doc__ or ""
        assert "73.3pp" in doc, "isotropic face-count spread missing"
        assert "7.9pp" in doc, "isotropic marching-cubes spread missing"
        assert "45.9%" in doc, "anisotropic worst-case error missing"
        assert "39.6pp" in doc, "anisotropic orientation spread missing"
        assert "does NOT cancel" in doc, "must state the bias is not cancellable"


def test_importing_contracts_does_not_pull_in_ml_packages():
    """The measurement layer must run where torch and cellpose are not installed."""
    assert "torch" not in sys.modules
    assert "cellpose" not in sys.modules
