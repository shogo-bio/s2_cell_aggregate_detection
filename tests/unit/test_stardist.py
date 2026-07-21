"""StarDist backend, exercised against a fake engine with no ML installed.

The real library is TensorFlow-based and lives in a separate environment
(dev/envs/stardist-x64), so every test here injects a fake engine. The one test
that touches the real thing is marked ``needs_ml`` and excluded by default.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass

import numpy as np
import pytest

from s2_adhesion.config import (
    ConfigError,
    DirectStarDistConfig,
    StarDistModelConfig,
    load_config,
)
from s2_adhesion.contracts import (
    ChannelBinding,
    ChannelRole,
    FieldIdentity,
    ImageVolume,
    VoxelGeometry,
)
from s2_adhesion.segmentation.direct_stardist import DirectStarDistBackend
from s2_adhesion.segmentation.protocol import (
    PreparedStarDistInput,
    SegmentationRequest,
    StarDistRawResult,
)
from s2_adhesion.segmentation.stardist_preprocess import (
    map_labels_to_original_grid,
    prepare_stardist_input,
)

ANISO = (0.5, 0.1, 0.1)


@dataclass
class FakeStarDistEngine:
    """Records what it was handed and returns canned labels."""

    labels: np.ndarray
    model_name: str = "fake"
    package_version: str = "0.0.0"
    model_anisotropy: tuple[float, float, float] | None = None
    is_pretrained: bool = False
    seen: PreparedStarDistInput | None = None

    def predict(self, prepared: PreparedStarDistInput) -> StarDistRawResult:
        self.seen = prepared
        return StarDistRawResult(
            labels=self.labels, n_instances=int(self.labels.max())
        )


def make_image(shape=(20, 60, 100), spacing=ANISO) -> ImageVolume:
    zz, yy, xx = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]].astype(float)
    zz *= spacing[0]
    yy *= spacing[1]
    xx *= spacing[2]
    vol = np.zeros(shape, np.float32)
    for cx in (3.0, 7.0):
        r = np.sqrt((zz - 5.0) ** 2 + (yy - 3.0) ** 2 + (xx - cx) ** 2)
        vol[(r < 2.0) & (r > 1.4)] = 1.0
    return ImageVolume(
        data=vol[None],
        geometry=VoxelGeometry(spacing_um_zyx=spacing),
        channels=(ChannelBinding("membrane", 0, frozenset({ChannelRole.MEMBRANE})),),
        identity=FieldIdentity("synthetic", "f0", "memory://", 0, "0" * 64),
    )


def make_config(**kw) -> DirectStarDistConfig:
    model_kw = kw.pop("model", {})
    return DirectStarDistConfig(
        strategy="direct_stardist",
        input_channel_ids=("membrane",),
        model=StarDistModelConfig(pretrained_name="3D_demo", **model_kw),
        min_cell_volume_um3=kw.pop("min_cell_volume_um3", 0.1),
        **kw,
    )


class TestPreprocessing:
    def test_resamples_to_isotropic_and_records_the_model_grid(self):
        image = make_image()
        prepared = prepare_stardist_input(image, make_config())
        assert prepared.was_resampled is True
        assert prepared.spacing_um_zyx == (0.1, 0.1, 0.1)
        # Z was 0.5 um, now 0.1 um -> five times as many planes.
        assert prepared.data.shape[0] == pytest.approx(image.shape_zyx[0] * 5, abs=1)
        assert prepared.original_shape_zyx == image.shape_zyx

    def test_resampling_can_be_turned_off(self):
        image = make_image()
        prepared = prepare_stardist_input(
            image, make_config(model={"resample_isotropic": False})
        )
        assert prepared.was_resampled is False
        assert prepared.spacing_um_zyx == ANISO
        assert prepared.data.shape == image.shape_zyx

    def test_isotropic_input_is_left_alone(self):
        image = make_image(shape=(40, 40, 60), spacing=(0.2, 0.2, 0.2))
        prepared = prepare_stardist_input(image, make_config())
        assert prepared.was_resampled is False

    def test_labels_come_back_on_the_original_grid_without_interpolation(self):
        image = make_image()
        prepared = prepare_stardist_input(image, make_config())

        model_labels = np.zeros(prepared.data.shape, np.uint32)
        model_labels[:, :, :50] = 1
        model_labels[:, :, 50:] = 7  # non-consecutive on purpose

        mapped = map_labels_to_original_grid(model_labels, prepared)
        assert mapped.shape == image.shape_zyx
        assert mapped.dtype == np.uint32
        # Nearest-neighbour only: no id may be invented between 1 and 7.
        assert set(np.unique(mapped)) <= {1, 7}

    def test_more_than_one_input_channel_is_rejected(self):
        from s2_adhesion.errors import SegmentationError

        image = make_image()
        config = DirectStarDistConfig(
            strategy="direct_stardist",
            input_channel_ids=("membrane", "membrane2"),
            model=StarDistModelConfig(pretrained_name="3D_demo"),
        )
        with pytest.raises(SegmentationError, match="single intensity volume"):
            prepare_stardist_input(image, config)


class TestBackend:
    def _run(self, engine_labels=None, config=None, image=None):
        image = image or make_image()
        config = config or make_config()
        prepared = prepare_stardist_input(image, config)
        if engine_labels is None:
            engine_labels = np.zeros(prepared.data.shape, np.uint32)
            engine_labels[:, :, : prepared.data.shape[2] // 2] = 1
            engine_labels[:, :, prepared.data.shape[2] // 2 :] = 2
        engine = FakeStarDistEngine(labels=engine_labels)
        backend = DirectStarDistBackend(config=config, engine=engine)
        return backend.segment(SegmentationRequest(image=image, run_id="run0")), engine

    def test_produces_labels_on_the_acquired_grid(self):
        result, _ = self._run()
        assert result.labels.cells.shape == make_image().shape_zyx
        assert result.labels.cells.dtype == np.uint32

    def test_records_stardist_provenance(self):
        result, _ = self._run()
        prov = result.labels.provenance
        assert prov.package_name == "stardist"
        assert prov.strategy == "direct_stardist"
        assert prov.backend_id == "direct_stardist"
        assert prov.config_sha256

    def test_config_hash_changes_with_the_config(self):
        a, _ = self._run(config=make_config(min_cell_volume_um3=1.0))
        b, _ = self._run(config=make_config(min_cell_volume_um3=2.0))
        assert a.labels.provenance.config_sha256 != b.labels.provenance.config_sha256

    def test_warns_when_a_published_model_is_used(self):
        image = make_image()
        config = make_config()
        prepared = prepare_stardist_input(image, config)
        engine = FakeStarDistEngine(
            labels=np.ones(prepared.data.shape, np.uint32), is_pretrained=True
        )
        result = DirectStarDistBackend(config=config, engine=engine).segment(
            SegmentationRequest(image=image, run_id="r")
        )
        assert any("pretrained_model" in w for w in result.diagnostics.warnings)

    def test_warns_when_the_model_anisotropy_does_not_match_the_input(self):
        """StarDist has no predict-time anisotropy knob, so this is silent otherwise.

        The one published 3D model is trained at (2, 1, 1), so feeding it
        isotropic data -- which resample_isotropic=True does -- is a real
        mismatch that would otherwise pass unnoticed.
        """
        image = make_image()
        config = make_config()
        prepared = prepare_stardist_input(image, config)
        engine = FakeStarDistEngine(
            labels=np.ones(prepared.data.shape, np.uint32),
            model_anisotropy=(2.0, 1.0, 1.0),
        )
        result = DirectStarDistBackend(config=config, engine=engine).segment(
            SegmentationRequest(image=image, run_id="r")
        )
        assert any("anisotropy_mismatch" in w for w in result.diagnostics.warnings)

    def test_matching_anisotropy_does_not_warn(self):
        image = make_image()
        config = make_config(model={"resample_isotropic": False})
        prepared = prepare_stardist_input(image, config)
        engine = FakeStarDistEngine(
            labels=np.ones(prepared.data.shape, np.uint32),
            model_anisotropy=(5.0, 1.0, 1.0),  # matches 0.5/0.1/0.1
        )
        result = DirectStarDistBackend(config=config, engine=engine).segment(
            SegmentationRequest(image=image, run_id="r")
        )
        assert not any("anisotropy_mismatch" in w for w in result.diagnostics.warnings)

    def test_warns_when_nothing_was_found(self):
        image = make_image()
        config = make_config()
        prepared = prepare_stardist_input(image, config)
        result, _ = self._run(
            engine_labels=np.zeros(prepared.data.shape, np.uint32), config=config
        )
        assert any("no_instances" in w for w in result.diagnostics.warnings)

    def test_a_raising_engine_leaves_nothing_behind(self):
        class Boom:
            model_name = "boom"
            package_version = "0"
            model_anisotropy = None
            is_pretrained = False

            def predict(self, prepared):
                raise RuntimeError("model exploded")

        image = make_image()
        backend = DirectStarDistBackend(config=make_config(), engine=Boom())
        with pytest.raises(RuntimeError, match="model exploded"):
            backend.segment(SegmentationRequest(image=image, run_id="r"))


class TestConfigValidation:
    BASE = """
    schema_version: s2-pipeline-config/v1
    channels:
      - {channel_id: membrane, source_index: 0, roles: [membrane]}
    segmentation:
      strategy: direct_stardist
      input_channel_ids: [membrane]
      model:
    """

    def _write(self, tmp_path, model_block):
        import textwrap

        text = textwrap.dedent(self.BASE) + textwrap.indent(
            textwrap.dedent(model_block), "    "
        )
        path = tmp_path / "cfg.yaml"
        path.write_text(text, encoding="utf-8")
        return path

    def test_a_valid_pretrained_config_loads(self, tmp_path):
        cfg = load_config(self._write(tmp_path, "pretrained_name: 3D_demo\n"))
        assert cfg.segmentation.strategy == "direct_stardist"

    def test_naming_both_a_pretrained_and_a_custom_model_is_rejected(self, tmp_path):
        with pytest.raises(ConfigError, match="exactly one"):
            load_config(
                self._write(
                    tmp_path,
                    "pretrained_name: 3D_demo\ncustom_model_dir: ./m\n"
                    "custom_model_name: mine\n",
                )
            )

    def test_naming_neither_is_rejected(self, tmp_path):
        """No silent fallback: a nuclei model on membrane data must be deliberate."""
        with pytest.raises(ConfigError, match="exactly one"):
            load_config(self._write(tmp_path, "prob_threshold: 0.5\n"))

    def test_custom_model_dir_needs_a_name(self, tmp_path):
        with pytest.raises(ConfigError, match="custom_model_name"):
            load_config(self._write(tmp_path, "custom_model_dir: ./models\n"))


def test_importing_the_backend_does_not_import_tensorflow_or_stardist():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            "import s2_adhesion.segmentation.direct_stardist; "
            "import s2_adhesion.segmentation.stardist_preprocess; "
            "import s2_adhesion.segmentation.factory; "
            "leaked=[m for m in sys.modules "
            "if m.split('.')[0] in ('tensorflow','stardist','torch','cellpose')]; "
            "print(leaked); sys.exit(1 if leaked else 0)",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout


@pytest.mark.slow
@pytest.mark.needs_ml
def test_real_stardist_model_loads_and_predicts():
    """Runs the actual library. Needs the stardist environment.

    Deliberately asserts only that the plumbing works, NOT that the instances
    are good: the only published 3D model is a nuclei demo, and on
    membrane-shell data it over-segments badly. Judging quality needs a model
    trained on this data.
    """
    pytest.importorskip("stardist")
    from s2_adhesion.segmentation.factory import create_stardist_engine

    engine = create_stardist_engine(StarDistModelConfig(pretrained_name="3D_demo"))
    assert engine.is_pretrained is True
    assert engine.model_anisotropy == (2.0, 1.0, 1.0)

    image = make_image()
    config = make_config()
    result = DirectStarDistBackend(config=config, engine=engine).segment(
        SegmentationRequest(image=image, run_id="smoke")
    )
    assert result.labels.cells.shape == image.shape_zyx
    assert result.labels.cells.dtype == np.uint32
    assert result.labels.provenance.package_name == "stardist"
