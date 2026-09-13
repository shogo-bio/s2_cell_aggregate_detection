"""``direct_cellpose`` must honour its post-processing config.

Until 2026-09-13 the strategy mapped cellpose's labels back to the acquired
grid and stopped: ``min_cell_volume_um3``, ``fill_internal_holes`` and
``split_disconnected_labels`` were accepted by the config and silently
ignored. On the 25-field real-data run that left 142 objects below the
configured 50 um^3. These tests drive the backend with a fake engine so the
whole post-processing chain is exercised with no ML package installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from s2_adhesion.config import CellposeModelConfig, DirectInstanceConfig
from s2_adhesion.contracts import ChannelRole
from s2_adhesion.segmentation.direct_cellpose import DirectCellposeBackend
from s2_adhesion.segmentation.protocol import (
    CellposeRawResult,
    PreparedCellposeInput,
    SegmentationRequest,
)
from tests.conftest import make_image_volume

SPACING = (2.0, 1.0, 1.0)  # dz, dy, dx in um -> voxel = 2 um^3, square in-plane


@dataclass
class FakeEngine:
    """Returns a fixed label volume whatever it is given."""

    labels: np.ndarray
    model_name: str = "fake"
    package_version: str = "0"
    device: str = "cpu"
    model_checksum: str | None = None
    package_major: int = 3
    calls: list = field(default_factory=list)

    def evaluate(self, prepared: PreparedCellposeInput, *, config: CellposeModelConfig):
        self.calls.append(prepared)
        return CellposeRawResult(labels=self.labels.astype(np.uint32), diameters_px=None, extra={})


def _image(shape=(6, 20, 20)):
    data = np.zeros((1, *shape), np.float32)
    data[0, :, 2:18, 2:18] = 100.0
    return make_image_volume(
        data, spacing=SPACING, channel_ids=("membrane",), roles=(ChannelRole.MEMBRANE,)
    )


def _labels_with_debris():
    """One real cell (6 planes, 6x6 = 432 um^3), one 1-plane fleck away from the
    Z border (1x3x3 = 18 um^3), one 2-plane fragment inside the volume
    (2x6x6 = 144 um^3, above the volume floor but too thin), and one 1-plane
    sliver ON the Z border (1x6x6 = 72 um^3)."""
    lab = np.zeros((6, 20, 20), np.uint32)
    lab[:, 2:8, 2:8] = 1          # real cell
    lab[3, 12:15, 12:15] = 2      # fleck, 18 um^3, plane 3
    lab[2:4, 12:18, 2:8] = 3      # thin fragment, 144 um^3, planes 2-3
    lab[0, 12:18, 12:18] = 4      # sliver on the first plane
    return lab


def _run(config: DirectInstanceConfig, labels: np.ndarray):
    engine = FakeEngine(labels=labels)
    backend = DirectCellposeBackend(config=config, engine=engine)
    result = backend.segment(SegmentationRequest(image=_image(), run_id="t"))
    return result


class TestVolumeFilterIsApplied:
    def test_objects_below_min_volume_are_dropped(self):
        config = DirectInstanceConfig(
            strategy="direct_cellpose", input_channel_ids=("membrane",),
            min_cell_volume_um3=50.0,
        )
        result = _run(config, _labels_with_debris())
        ids = set(np.unique(result.labels.cells)) - {0}
        assert 2 not in ids            # 18 um^3 fleck removed
        assert {1, 3, 4} <= ids        # everything >= 50 um^3 kept
        assert any("small_objects_removed: 1" in w for w in result.diagnostics.warnings)
        assert result.diagnostics.extra["n_instances_raw"] == 4
        assert result.diagnostics.extra["n_instances_after_volume_filter"] == 3

    def test_min_volume_zero_keeps_everything(self):
        config = DirectInstanceConfig(
            strategy="direct_cellpose", input_channel_ids=("membrane",),
            min_cell_volume_um3=0.0,
        )
        result = _run(config, _labels_with_debris())
        assert set(np.unique(result.labels.cells)) - {0} == {1, 2, 3, 4}
        assert result.diagnostics.warnings == ()


class TestZExtentFilter:
    def test_thin_objects_away_from_the_border_are_dropped_but_border_slivers_kept(self):
        config = DirectInstanceConfig(
            strategy="direct_cellpose", input_channel_ids=("membrane",),
            min_cell_volume_um3=50.0, min_z_extent_planes=3,
        )
        result = _run(config, _labels_with_debris())
        ids = set(np.unique(result.labels.cells)) - {0}
        assert ids == {1, 4}, ids      # thin fragment 3 dropped; border sliver 4 kept
        assert any("thin_objects_removed: 1" in w for w in result.diagnostics.warnings)
        assert result.diagnostics.extra["n_instances_final"] == 2

    def test_default_none_applies_no_z_filter(self):
        config = DirectInstanceConfig(
            strategy="direct_cellpose", input_channel_ids=("membrane",),
            min_cell_volume_um3=50.0,
        )
        assert config.min_z_extent_planes is None
        result = _run(config, _labels_with_debris())
        assert 3 in set(np.unique(result.labels.cells))


class TestSplitAndFill:
    def test_disconnected_id_is_split_and_hole_is_filled(self):
        lab = np.zeros((6, 20, 20), np.uint32)
        lab[:, 2:8, 2:8] = 1
        lab[:, 12:18, 12:18] = 1          # same id, far away -> should split
        lab[2:4, 4:6, 4:6] = 0            # enclosed 3D hole in the first cell -> should fill
        config = DirectInstanceConfig(
            strategy="direct_cellpose", input_channel_ids=("membrane",),
            min_cell_volume_um3=50.0,
        )
        result = _run(config, lab)
        cells = result.labels.cells
        assert len(set(np.unique(cells)) - {0}) == 2
        assert cells[2:4, 4:6, 4:6].all()

    def test_split_and_fill_can_be_switched_off(self):
        lab = np.zeros((6, 20, 20), np.uint32)
        lab[:, 2:8, 2:8] = 1
        lab[:, 12:18, 12:18] = 1
        lab[2:4, 4:6, 4:6] = 0
        config = DirectInstanceConfig(
            strategy="direct_cellpose", input_channel_ids=("membrane",),
            min_cell_volume_um3=50.0, split_disconnected_labels=False,
            fill_internal_holes=False,
        )
        cells = _run(config, lab).labels.cells
        assert set(np.unique(cells)) - {0} == {1}
        assert not cells[2:4, 4:6, 4:6].any()


def test_min_z_extent_planes_must_be_positive(tmp_path):
    import textwrap

    from s2_adhesion.config import load_config
    from s2_adhesion.errors import ConfigError

    cfg = tmp_path / "c.yaml"
    cfg.write_text(textwrap.dedent("""
        schema_version: s2-pipeline-config/v1
        channels:
          - {channel_id: mem, source_index: 0, roles: [membrane]}
        segmentation:
          strategy: direct_cellpose
          input_channel_ids: [mem]
          min_z_extent_planes: 0
    """), encoding="utf-8")
    with pytest.raises(ConfigError, match="min_z_extent_planes must be >= 1"):
        load_config(cfg)
