"""Tests for the segmentation factory, DirectCellposeBackend, and the two
version-specific cellpose adapters.

Two families of test live here, deliberately kept separate:

* Tests against ``FakeCellposeEngine`` / ``RaisingCellposeEngine`` -- pure
  Python doubles satisfying the ``CellposeEngine`` protocol structurally,
  used to test ``DirectCellposeBackend`` and preprocessing wiring with zero
  ML packages involved.
* Tests against the REAL ``CellposeV3Engine`` / ``CellposeV4Engine`` classes
  with a FAKE ``_model`` injected -- these import ``cellpose_v3``/
  ``cellpose_v4`` (real modules, allowed to import cellpose/torch), but never
  construct a real cellpose model and never run real inference. Each such
  check runs in a fresh subprocess (see ``_run_json_snippet``), never as an
  in-process import: importing those adapter modules in-process would pull
  torch into this pytest session's ``sys.modules`` for good, breaking the
  "no ML imports" checks other test files in this suite rely on being
  in-process (e.g. ``tests/test_contracts.py``,
  ``tests/unit/test_manifests.py``) -- those files run whenever the *whole*
  suite runs, in a session-wide shared ``sys.modules``, and have no way to
  know this file ran first.

This machine has cellpose 4.2.1.1 installed and nothing else -- there is no
real cellpose 3 to test cellpose_v3.py against. That is fine: the adapter's
own logic (which kwargs it builds and forwards) is fully exercised via a
fake ``_model``, independent of which major is actually installed, and the
"configured major 3, installed major 4" factory test uses the REAL installed
version deliberately.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import pytest

from s2_adhesion.config import CellposeModelConfig, DirectInstanceConfig
from s2_adhesion.errors import MLDependencyError
from s2_adhesion.segmentation.direct_cellpose import DirectCellposeBackend
from s2_adhesion.segmentation.factory import create_cellpose_engine
from s2_adhesion.segmentation.protocol import (
    CellposeRawResult,
    PreparedCellposeInput,
    SegmentationRequest,
)
from s2_adhesion.contracts import ChannelRole
from tests.conftest import ANISOTROPIC, make_image_volume

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_snippet(code: str) -> subprocess.CompletedProcess:
    """Run `code` in a brand-new interpreter so sys.modules starts empty."""
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO_ROOT
    )


# ─── fakes satisfying the CellposeEngine protocol structurally ────────────────


@dataclass
class FakeCellposeEngine:
    package_major: Literal[3, 4] = 4
    model_name: str = "fake-model"
    package_version: str = "0.0.0-fake"
    device: str = "cpu"
    model_checksum: str | None = "deadbeef"
    labels_factory: Callable[[PreparedCellposeInput], np.ndarray] | None = None
    calls: list[tuple[PreparedCellposeInput, CellposeModelConfig]] = field(
        default_factory=list
    )

    def evaluate(
        self, prepared: PreparedCellposeInput, *, config: CellposeModelConfig
    ) -> CellposeRawResult:
        self.calls.append((prepared, config))
        if self.labels_factory is not None:
            labels = self.labels_factory(prepared)
        else:
            labels = np.zeros(prepared.data.shape[1:], dtype=np.uint32)
        return CellposeRawResult(labels=labels, diameters_px=None, extra={})


class RaisingCellposeEngine:
    package_major: Literal[3, 4] = 4
    model_name = "fake-raising"
    package_version = "0.0.0-fake"
    device = "cpu"
    model_checksum: str | None = None

    def evaluate(
        self, prepared: PreparedCellposeInput, *, config: CellposeModelConfig
    ) -> CellposeRawResult:
        raise RuntimeError("engine exploded")


def _direct_config(**overrides: Any) -> DirectInstanceConfig:
    kwargs: dict[str, Any] = dict(
        strategy="direct_cellpose", input_channel_ids=("membrane",)
    )
    kwargs.update(overrides)
    return DirectInstanceConfig(**kwargs)


# ─── DirectCellposeBackend (dependency-injected engine, no ML needed) ─────────


class TestDirectCellposeBackend:
    def test_anisotropy_passed_to_engine_equals_dz_over_min_dy_dx(self):
        data = np.random.default_rng(0).random((1, 4, 6, 6)).astype(np.float32)
        image = make_image_volume(
            data,
            spacing=ANISOTROPIC,
            channel_ids=("membrane",),
            roles=(ChannelRole.MEMBRANE,),
        )
        engine = FakeCellposeEngine()
        backend = DirectCellposeBackend(config=_direct_config(), engine=engine)

        backend.segment(SegmentationRequest(image=image, run_id="run0"))

        assert len(engine.calls) == 1
        prepared, _ = engine.calls[0]
        dz, dy, dx = ANISOTROPIC
        assert prepared.anisotropy == pytest.approx(dz / min(dy, dx))

    def test_result_label_volume_matches_original_grid_and_dtype(self):
        data = np.random.default_rng(1).random((1, 3, 5, 10)).astype(np.float32)
        image = make_image_volume(
            data,
            spacing=(0.5, 0.2, 0.1),
            channel_ids=("membrane",),
            roles=(ChannelRole.MEMBRANE,),
        )
        engine = FakeCellposeEngine()
        backend = DirectCellposeBackend(config=_direct_config(), engine=engine)

        result = backend.segment(SegmentationRequest(image=image, run_id="run0"))

        assert result.labels.cells.shape == image.shape_zyx
        assert result.labels.cells.dtype == np.uint32

    def test_provenance_is_populated_from_engine_attributes(self):
        data = np.random.default_rng(2).random((1, 3, 4, 4)).astype(np.float32)
        image = make_image_volume(
            data,
            spacing=ANISOTROPIC,
            channel_ids=("membrane",),
            roles=(ChannelRole.MEMBRANE,),
        )
        engine = FakeCellposeEngine(
            model_name="cyto3", package_version="3.1.1.3", device="cpu",
            model_checksum="abc123",
        )
        backend = DirectCellposeBackend(config=_direct_config(), engine=engine)

        result = backend.segment(SegmentationRequest(image=image, run_id="run-xyz"))
        prov = result.labels.provenance

        assert prov.run_id == "run-xyz"
        assert prov.strategy == "direct_cellpose"
        assert prov.backend_id == "direct_cellpose"
        assert prov.device == "cpu"
        assert prov.package_version == "3.1.1.3"
        assert prov.model_name == "cyto3"
        assert prov.model_sha256 == "abc123"
        assert prov.input_image_sha256 == image.identity.image_content_sha256

    def test_diagnostics_report_preprocess_and_evaluate_timing(self):
        data = np.random.default_rng(3).random((1, 3, 4, 4)).astype(np.float32)
        image = make_image_volume(
            data,
            spacing=ANISOTROPIC,
            channel_ids=("membrane",),
            roles=(ChannelRole.MEMBRANE,),
        )
        engine = FakeCellposeEngine()
        backend = DirectCellposeBackend(config=_direct_config(), engine=engine)

        result = backend.segment(SegmentationRequest(image=image, run_id="run0"))

        assert set(result.diagnostics.timing_seconds) == {"preprocess", "evaluate"}
        assert all(v >= 0.0 for v in result.diagnostics.timing_seconds.values())

    def test_raising_engine_leaves_no_partial_result(self):
        data = np.random.default_rng(4).random((1, 3, 4, 4)).astype(np.float32)
        image = make_image_volume(
            data,
            spacing=ANISOTROPIC,
            channel_ids=("membrane",),
            roles=(ChannelRole.MEMBRANE,),
        )
        backend = DirectCellposeBackend(config=_direct_config(), engine=RaisingCellposeEngine())
        request = SegmentationRequest(image=image, run_id="run0")

        with pytest.raises(RuntimeError, match="engine exploded"):
            backend.segment(request)

        # the backend itself is stateless (frozen dataclass); a second call
        # with a working engine must succeed cleanly, proving the failed call
        # left nothing behind on the backend, config, or request.
        good_engine = FakeCellposeEngine()
        good_backend = DirectCellposeBackend(config=backend.config, engine=good_engine)
        result = good_backend.segment(request)
        assert result.labels.cells.shape == image.shape_zyx

    def test_label_ids_from_engine_survive_to_final_result(self):
        # Model grid is 2x upsampled in Y (matches spacing below); give back
        # two distinct label blocks and confirm both ids reach the final grid.
        data = np.random.default_rng(5).random((1, 2, 4, 4)).astype(np.float32)
        image = make_image_volume(
            data,
            spacing=(0.5, 0.2, 0.1),
            channel_ids=("membrane",),
            roles=(ChannelRole.MEMBRANE,),
        )

        def labels_factory(prepared: PreparedCellposeInput) -> np.ndarray:
            _, z, y, x = prepared.data.shape
            out = np.zeros((z, y, x), dtype=np.uint32)
            out[:, : y // 2, :] = 1
            out[:, y // 2 :, :] = 2
            return out

        engine = FakeCellposeEngine(labels_factory=labels_factory)
        backend = DirectCellposeBackend(config=_direct_config(), engine=engine)

        result = backend.segment(SegmentationRequest(image=image, run_id="run0"))

        # the fake fills the whole resampled grid with labels 1 and 2 (no
        # background), so both ids -- and only those -- must survive mapping
        # back onto the original grid.
        assert set(np.unique(result.labels.cells).tolist()) == {1, 2}


# ─── factory: version gating, before any model construction ───────────────────


class TestFactoryVersionGating:
    def test_configured_major_3_with_installed_major_4_raises_before_import(self):
        """Uses the REAL installed cellpose (4.2.1.1 on this host)."""
        code = (
            "import sys\n"
            "from s2_adhesion.config import CellposeModelConfig\n"
            "from s2_adhesion.errors import MLDependencyError\n"
            "from s2_adhesion.segmentation.factory import create_cellpose_engine\n"
            "config = CellposeModelConfig(package_major=3, model_name='cyto3')\n"
            "try:\n"
            "    create_cellpose_engine(config)\n"
            "    print('NO_RAISE')\n"
            "except MLDependencyError as exc:\n"
            "    assert 's2_adhesion.segmentation.cellpose_v3' not in sys.modules\n"
            "    assert 's2_adhesion.segmentation.cellpose_v4' not in sys.modules\n"
            "    assert 'cellpose' not in sys.modules\n"
            "    assert 'torch' not in sys.modules\n"
            "    print('RAISED:' + str(exc))\n"
        )
        result = _run_snippet(code)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "RAISED:" in result.stdout, result.stdout + result.stderr
        message = result.stdout
        assert "3" in message and "4" in message

    def test_cellpose_not_installed_raises(self, monkeypatch):
        import importlib.metadata as md

        def fake_version(name: str) -> str:
            raise md.PackageNotFoundError(name)

        monkeypatch.setattr(md, "version", fake_version)
        config = CellposeModelConfig(package_major=4, model_name="cpsam")

        with pytest.raises(MLDependencyError, match="not installed"):
            create_cellpose_engine(config)

    def test_checksum_mismatch_raises_before_import(self, tmp_path):
        model_file = tmp_path / "model.pt"
        model_file.write_bytes(b"not a real model checkpoint")
        config = CellposeModelConfig(
            package_major=4,
            model_name="cpsam",
            pretrained_model_path=model_file,
            expected_model_sha256="0" * 64,
        )

        with pytest.raises(MLDependencyError, match="checksum"):
            create_cellpose_engine(config)

    def test_missing_pretrained_model_file_raises(self, tmp_path):
        missing = tmp_path / "does_not_exist.pt"
        config = CellposeModelConfig(
            package_major=4, model_name="cpsam", pretrained_model_path=missing
        )

        with pytest.raises(MLDependencyError):
            create_cellpose_engine(config)


# ─── real adapters, fake underlying model, always subprocess-isolated ─────────
#
# Importing cellpose_v3/cellpose_v4 for real (even with a fake `_model`
# swapped in) pulls torch into sys.modules as a side effect. That is fine in
# isolation, but this test FILE is part of a shared pytest session with
# other files that assert "torch not in sys.modules" in-process -- so every
# scenario that needs a real CellposeV3Engine/CellposeV4Engine runs inside a
# throwaway subprocess (one per adapter, covering several scenarios each) and
# reports back over stdout as JSON. The subprocess never installs/downloads
# anything and never calls real cellpose -- `_model` is always the fake
# defined inline in the script.


def _run_json_snippet(code: str) -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO_ROOT
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for line in result.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[len("RESULT_JSON:") :])
    raise AssertionError(f"no RESULT_JSON line: stdout={result.stdout!r} stderr={result.stderr!r}")


_V3_ADAPTER_SCRIPT = """
import json
import numpy as np
from s2_adhesion.segmentation.cellpose_v3 import CellposeV3Engine
from s2_adhesion.segmentation.protocol import PreparedCellposeInput
from s2_adhesion.config import CellposeModelConfig, CellposeEvalConfig


class FakeV3Model:
    def __init__(self):
        self.eval_kwargs = None

    def eval(self, x, **kwargs):
        self.eval_kwargs = kwargs
        return np.zeros(x.shape[:3], dtype=np.uint32), None, None, 12.0


def make_prepared(n_channels):
    return PreparedCellposeInput(
        data=np.zeros((n_channels, 4, 8, 8), dtype=np.float32),
        channel_ids=tuple("ch%d" % i for i in range(n_channels)),
        anisotropy=5.0,
        original_shape_zyx=(4, 8, 8),
        resampled_spacing_um_zyx=(0.5, 0.1, 0.1),
        diameter_px=None,
    )


def make_engine(model):
    return CellposeV3Engine(
        model_name="cyto3", package_version="3.1.1.3", device="cpu",
        model_checksum=None, _model=model,
    )


results = {}

model_a = FakeV3Model()
result_a = make_engine(model_a).evaluate(
    make_prepared(1), config=CellposeModelConfig(package_major=3, model_name="cyto3")
)
results["single_channel"] = {
    "channels": model_a.eval_kwargs.get("channels"),
    "z_axis": model_a.eval_kwargs.get("z_axis"),
    "do_3D": model_a.eval_kwargs.get("do_3D"),
    "anisotropy": model_a.eval_kwargs.get("anisotropy"),
    "labels_dtype": str(result_a.labels.dtype),
    "diameters_px": result_a.diameters_px,
}

model_b = FakeV3Model()
make_engine(model_b).evaluate(
    make_prepared(2), config=CellposeModelConfig(package_major=3, model_name="cyto3")
)
results["two_channel"] = {"channels": model_b.eval_kwargs.get("channels")}

model_c = FakeV3Model()
eval_cfg = CellposeEvalConfig(
    flow_threshold=0.7, cellprob_threshold=-1.0, bsize=224, augment=True, batch_size=3
)
make_engine(model_c).evaluate(
    make_prepared(1),
    config=CellposeModelConfig(package_major=3, model_name="cyto3", eval=eval_cfg),
)
results["eval_forward"] = {
    "flow_threshold": model_c.eval_kwargs.get("flow_threshold"),
    "cellprob_threshold": model_c.eval_kwargs.get("cellprob_threshold"),
    "bsize": model_c.eval_kwargs.get("bsize"),
    "augment": model_c.eval_kwargs.get("augment"),
    "batch_size": model_c.eval_kwargs.get("batch_size"),
    "has_tile_key": "tile" in model_c.eval_kwargs,
}

print("RESULT_JSON:" + json.dumps(results))
"""


_V4_ADAPTER_SCRIPT = """
import json
import numpy as np
from s2_adhesion.segmentation.cellpose_v4 import CellposeV4Engine
from s2_adhesion.segmentation.protocol import PreparedCellposeInput
from s2_adhesion.config import CellposeModelConfig, CellposeEvalConfig
from s2_adhesion.errors import SegmentationError


class FakeV4Model:
    def __init__(self):
        self.eval_kwargs = None
        self.called = False

    def eval(self, x, **kwargs):
        self.called = True
        self.eval_kwargs = kwargs
        return np.zeros(x.shape[:3], dtype=np.uint32), None, None


def make_prepared(n_channels):
    return PreparedCellposeInput(
        data=np.zeros((n_channels, 4, 8, 8), dtype=np.float32),
        channel_ids=tuple("ch%d" % i for i in range(n_channels)),
        anisotropy=5.0,
        original_shape_zyx=(4, 8, 8),
        resampled_spacing_um_zyx=(0.5, 0.1, 0.1),
        diameter_px=None,
    )


def make_engine(model):
    return CellposeV4Engine(
        model_name="cpsam", package_version="4.2.1.1", device="cpu",
        model_checksum=None, _model=model,
    )


results = {}

model_a = FakeV4Model()
result_a = make_engine(model_a).evaluate(
    make_prepared(1), config=CellposeModelConfig(package_major=4, model_name="cpsam")
)
results["single_channel"] = {
    "has_channels_key": "channels" in model_a.eval_kwargs,
    "z_axis": model_a.eval_kwargs.get("z_axis"),
    "do_3D": model_a.eval_kwargs.get("do_3D"),
    "anisotropy": model_a.eval_kwargs.get("anisotropy"),
    "labels_dtype": str(result_a.labels.dtype),
}

model_b = FakeV4Model()
make_engine(model_b).evaluate(
    make_prepared(1),
    config=CellposeModelConfig(
        package_major=4, model_name="cpsam", eval=CellposeEvalConfig()
    ),
)
results["tiling_defaults_omitted"] = {
    "has_bsize": "bsize" in model_b.eval_kwargs,
    "has_tile_overlap": "tile_overlap" in model_b.eval_kwargs,
    "has_tile": "tile" in model_b.eval_kwargs,
}

model_c = FakeV4Model()
eval_cfg = CellposeEvalConfig(
    flow_threshold=0.55, cellprob_threshold=0.2, tile_overlap=0.1, augment=True, batch_size=4
)
make_engine(model_c).evaluate(
    make_prepared(1),
    config=CellposeModelConfig(package_major=4, model_name="cpsam", eval=eval_cfg),
)
results["eval_forward"] = {
    "flow_threshold": model_c.eval_kwargs.get("flow_threshold"),
    "cellprob_threshold": model_c.eval_kwargs.get("cellprob_threshold"),
    "augment": model_c.eval_kwargs.get("augment"),
    "batch_size": model_c.eval_kwargs.get("batch_size"),
    "has_tile_key": "tile" in model_c.eval_kwargs,
}

print("RESULT_JSON:" + json.dumps(results))
"""


def test_v3_engine_passes_channels_and_explicit_z_axis_and_forwards_eval_config():
    data = _run_json_snippet(_V3_ADAPTER_SCRIPT)

    single = data["single_channel"]
    assert single["channels"] == [0, 0]
    # Real cellpose 3.1.1.3 DOES accept z_axis, and warns "z_axis not
    # specified, assuming it is dim 0" when omitted. Passing it explicitly is
    # therefore correct for v3 as well as mandatory for v4 -- the two versions
    # differ in how they react to its absence, not in whether they accept it.
    assert single["z_axis"] == 0
    assert single["do_3D"] is True
    assert single["anisotropy"] == pytest.approx(5.0)
    assert single["labels_dtype"] == "uint32"
    assert single["diameters_px"] == pytest.approx(12.0)

    assert data["two_channel"]["channels"] == [1, 2]

    fwd = data["eval_forward"]
    assert fwd["flow_threshold"] == pytest.approx(0.7)
    assert fwd["cellprob_threshold"] == pytest.approx(-1.0)
    assert fwd["bsize"] == 224
    assert fwd["has_tile_key"] is False
    assert fwd["augment"] is True
    assert fwd["batch_size"] == 3


def test_v4_engine_passes_z_axis_not_channels_and_forwards_eval_config():
    data = _run_json_snippet(_V4_ADAPTER_SCRIPT)

    single = data["single_channel"]
    assert single["has_channels_key"] is False
    assert single["z_axis"] == 0
    assert single["do_3D"] is True
    assert single["anisotropy"] == pytest.approx(5.0)
    assert single["labels_dtype"] == "uint32"

    # Unset tiling options must be OMITTED, not passed as None, so cellpose
    # applies its own defaults. And `tile` must never be sent at all: no
    # released cellpose accepts it, and forwarding it raised TypeError on every
    # real 3D run until this was fixed.
    tiling = data["tiling_defaults_omitted"]
    assert tiling["has_bsize"] is False
    assert tiling["has_tile_overlap"] is False
    assert tiling["has_tile"] is False

    fwd = data["eval_forward"]
    assert fwd["flow_threshold"] == pytest.approx(0.55)
    assert fwd["cellprob_threshold"] == pytest.approx(0.2)
    assert fwd["augment"] is True
    assert fwd["batch_size"] == 4
    assert fwd["has_tile_key"] is False  # no boolean tile switch in the real v4 API


# ─── no ML imports from the non-adapter modules ────────────────────────────────


def test_importing_segmentation_modules_does_not_import_torch_or_cellpose():
    code = (
        "import sys\n"
        "import s2_adhesion.segmentation.protocol\n"
        "import s2_adhesion.segmentation.factory\n"
        "import s2_adhesion.segmentation.preprocess\n"
        "import s2_adhesion.segmentation.direct_cellpose\n"
        "assert 'torch' not in sys.modules, sys.modules.keys()\n"
        "assert 'cellpose' not in sys.modules, sys.modules.keys()\n"
        "print('OK')\n"
    )
    result = _run_snippet(code)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


@pytest.mark.needs_ml
def test_forwarded_kwargs_are_accepted_by_the_really_installed_cellpose():
    """Guard against signature drift in the actual library.

    Fake models accept whatever you hand them, so no amount of mocking can
    catch an argument cellpose does not have. That gap let a real bug through:
    both adapters forwarded ``tile=``, which neither cellpose 3.1.1.3 nor
    4.2.1.1 accepts, so every real 3D run raised
    ``TypeError: eval() got an unexpected keyword argument 'tile'`` while the
    whole mocked unit suite stayed green.

    This test reads the kwargs the adapter actually sends out of its source and
    checks them against ``inspect.signature`` of the installed library, so a
    future cellpose release that renames or drops a parameter fails here rather
    than in someone's overnight run.
    """
    import importlib
    import inspect
    import re
    from pathlib import Path

    cellpose = pytest.importorskip("cellpose")
    major = int(cellpose.version.split(".")[0])
    if major not in (3, 4):
        pytest.skip(f"unsupported cellpose major {major}")

    module_name = f"cellpose_v{major}"
    source = (
        Path(__file__).resolve().parents[2]
        / "src" / "s2_adhesion" / "segmentation" / f"{module_name}.py"
    ).read_text(encoding="utf-8")

    call_block = source.split("self._model.eval(")[1].split(")")[0]
    forwarded = set(re.findall(r"^\s*(\w+)=", call_block, re.M))
    # Names passed through the optional-kwargs dict rather than literally.
    forwarded |= {"bsize", "tile_overlap"}

    from cellpose import models

    accepted = set(inspect.signature(models.CellposeModel.eval).parameters)
    unsupported = forwarded - accepted
    assert not unsupported, (
        f"cellpose {cellpose.version} CellposeModel.eval() does not accept "
        f"{sorted(unsupported)}; {module_name}.py forwards them and every real "
        f"run will raise TypeError. Accepted parameters: {sorted(accepted)}"
    )
