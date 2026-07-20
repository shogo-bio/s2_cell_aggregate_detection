"""Integration smoke tests for the segmentation backend against REAL cellpose.

Two tests live here, deliberately separated by scope:

* ``test_cellpose3_cyto3_cpu_smoke`` -- runs a genuinely tiny cellpose 3
  (cyto3) CPU inference through the project's OWN segmentation backend
  (``DirectCellposeBackend``, wired up exactly the way production code would,
  via ``segmentation.factory.create_cellpose_engine`` -- never a raw
  ``cellpose.models`` call), and asserts the result is a valid ZYX ``uint32``
  ``LabelVolume`` with at least one instance and the input's exact spacing.
  Skips cleanly (not a failure) when cellpose is absent or its installed
  major version is not 3 -- in particular it always skips on this repo's main
  env, which pins cellpose 4.2.1.1; run it with the cellpose-3 env
  (``envs/cellpose-x64`` on this host, cellpose 3.1.1.3) to actually exercise
  it.

* ``test_cellpose4_adapter_signature_and_factory_gating`` -- validates ONLY
  constructor/eval-signature compatibility for cellpose 4: that the v4
  adapter (``CellposeV4Engine``) builds its ``eval()`` call with
  ``z_axis=0``, and that ``factory.create_cellpose_engine`` rejects a
  configured/installed major-version mismatch before touching any model. It
  NEVER runs cpsam inference and never even constructs a real
  ``models.CellposeModel`` -- see the prohibition comment on that test for
  why.

Both tests are ``@pytest.mark.slow`` and ``@pytest.mark.needs_ml`` (both
registered in ``pyproject.toml``), so the default ``pytest`` invocation
(``addopts = "-m 'not slow and not needs_ml'"``) never collects real work
here, and -- because that marker filter is what excludes them, not a
subprocess boundary -- they can never share a pytest session with any
default-run test that asserts "torch/cellpose not in sys.modules" (see
``tests/unit/test_cellpose_adapters.py``'s docstring for why that file uses
subprocess isolation instead: those checks run unmarked, in the default
suite, so a same-session in-process cellpose import there really would leak
torch into other tests). Reaching this file at all already requires an
explicit ``-m "slow and needs_ml"`` (or similar) override, which necessarily
deselects those unmarked default tests from the same run. That is why the
cellpose-4 test below imports ``cellpose_v4`` in-process rather than via a
subprocess snippet.
"""

from __future__ import annotations

import importlib.metadata
import time

import numpy as np
import pytest

from s2_adhesion.config import CellposeModelConfig, DirectInstanceConfig
from s2_adhesion.contracts import ChannelRole
from s2_adhesion.errors import MLDependencyError
from s2_adhesion.segmentation.direct_cellpose import DirectCellposeBackend
from s2_adhesion.segmentation.factory import create_cellpose_engine
from s2_adhesion.segmentation.protocol import PreparedCellposeInput, SegmentationRequest
from tests.conftest import make_image_volume
from tests.synthetic_volumes import ellipsoid

# A genuinely tiny grid: 4 x 48 x 48 voxels, isotropic 0.1 um spacing. Chosen
# and measured (not guessed) on 2026-07-18 against the real cellpose-3 env
# (envs/cellpose-x64, cellpose 3.1.1.3, CPU): a single ellipsoid blob at
# these exact parameters reliably yields exactly one cyto3 instance in
# ~5-20s wall time -- far under the 128.5s measured for the 10x256x256 3D
# benchmark in docs/cellpose_cpu_notes.md.
_SHAPE_ZYX = (4, 48, 48)
_SPACING_UM_ZYX = (0.1, 0.1, 0.1)
_DIAMETER_UM = 1.8


def _installed_cellpose_major() -> int | None:
    """The installed ``cellpose`` distribution's major version, or None.

    Reads package metadata only (mirrors ``factory._installed_cellpose_version``
    / ``_installed_major``, duplicated locally rather than imported since
    those are private to that module) -- never imports ``cellpose`` itself.
    """
    try:
        version = importlib.metadata.version("cellpose")
    except importlib.metadata.PackageNotFoundError:
        return None
    head = version.split(".", 1)[0]
    try:
        return int(head)
    except ValueError:
        return None


def _make_synthetic_field():
    """One small, single-blob synthetic field for the cyto3 smoke test.

    A single ellipsoid (radius ~1 um in-plane, ~0.35 um in Z -- deliberately
    smaller than the volume so it doesn't touch a border) over low-level
    noise, built with the same physical-coordinate convention as
    ``tests/synthetic_volumes.py``'s other generators.
    """
    centre_um = tuple(_SHAPE_ZYX[i] * _SPACING_UM_ZYX[i] / 2 for i in range(3))
    radii_um = (0.35, 1.0, 1.0)
    blob = ellipsoid(centre_um, radii_um, _SHAPE_ZYX, _SPACING_UM_ZYX).astype(np.float32)

    rng = np.random.default_rng(0)
    noise = rng.normal(0.05, 0.02, size=_SHAPE_ZYX).astype(np.float32)
    data = np.clip(noise * 0.3 + blob, 0.0, 1.0).astype(np.float32)[None, ...]

    return make_image_volume(
        data,
        spacing=_SPACING_UM_ZYX,
        channel_ids=("membrane",),
        roles=(ChannelRole.MEMBRANE,),
    )


# ─── cellpose 3: a real, tiny, CPU inference through the real backend ─────────


@pytest.mark.slow
@pytest.mark.needs_ml
def test_cellpose3_cyto3_cpu_smoke() -> None:
    major = _installed_cellpose_major()
    if major is None:
        pytest.skip("cellpose is not installed in this environment")
    if major != 3:
        pytest.skip(
            f"installed cellpose major is {major}, this smoke test needs a "
            "real cellpose 3.x (cyto3); run it with the cellpose-3 env "
            "(envs/cellpose-x64 on this host), not the main env (which "
            "pins cellpose 4)."
        )

    image = _make_synthetic_field()
    model_cfg = CellposeModelConfig(
        package_major=3, model_name="cyto3", device="cpu", diameter_um=_DIAMETER_UM
    )
    direct_cfg = DirectInstanceConfig(
        strategy="direct_cellpose", input_channel_ids=("membrane",), model=model_cfg
    )

    # Real factory -> real cellpose_v3 adapter -> real models.Cellpose. This
    # is the whole point of the test: exercise the project's own wiring, not
    # a raw cellpose call.
    engine = create_cellpose_engine(model_cfg)
    backend = DirectCellposeBackend(config=direct_cfg, engine=engine)

    t0 = time.perf_counter()
    result = backend.segment(SegmentationRequest(image=image, run_id="cellpose3-cpu-smoke"))
    elapsed = time.perf_counter() - t0

    labels = result.labels
    assert labels.cells.dtype == np.uint32
    assert labels.cells.ndim == 3
    assert labels.axes == "ZYX"
    assert labels.cells.shape == _SHAPE_ZYX
    assert labels.cell_ids().size >= 1, "expected at least one detected instance"
    assert labels.geometry.spacing_um_zyx == _SPACING_UM_ZYX
    # Generous budget: measured runs of this exact fixture were 5-20s; 90s
    # leaves headroom for a slower machine while still being far below the
    # 128.5s measured for the 10x256x256 3D benchmark (see the docs).
    assert elapsed < 90.0, (
        f"cyto3 CPU smoke run took {elapsed:.1f}s, over the 90s budget for "
        "a 4x48x48 volume -- see docs/cellpose_cpu_notes.md for the "
        "measured 3D baseline (10x256x256 -> 128.5s)"
    )


# ─── cellpose 4: signature/constructor compatibility ONLY, never real cpsam ──


@pytest.mark.slow
@pytest.mark.needs_ml
def test_cellpose4_adapter_signature_and_factory_gating() -> None:
    """Validates ONLY that the v4 adapter's eval-call shape and the factory's
    version gating are correct -- NEVER runs real cpsam inference.

    PROHIBITION: cpsam (cellpose 4's SAM/transformer model) measured at
    638.4s for a single 256x256 2D plane on this CPU-only, ARM64-host/
    x64-Python machine, with weights already cached (pure compute) -- see
    docs/cellpose_cpu_notes.md. A real ``eval()`` call here would make this
    test operationally unusable (and a 3D cpsam call did not even finish in
    10 minutes in the same measurement). Both checks below get everything
    they need from (a) a fake ``_model`` that only records kwargs -- never
    touches torch or a real network -- and (b) the factory's version-gating
    logic, which by construction never imports either adapter, let alone
    builds a model, when the major versions disagree. Real weights are never
    downloaded or loaded anywhere in this test.
    """
    try:
        from s2_adhesion.segmentation.cellpose_v4 import CellposeV4Engine
    except ImportError as exc:  # pragma: no cover - depends on local env
        pytest.skip(f"cellpose/torch not importable in this environment: {exc}")

    # ── 1. v4 adapter must pass z_axis=0 (cellpose 3 has no such argument,
    #        and cellpose 4 raises ValueError without it for 3D input) ──────
    class _RecordingFakeModel:
        """Records eval() kwargs. Never calls real cpsam or touches torch."""

        def __init__(self) -> None:
            self.eval_kwargs: dict | None = None

        def eval(self, x, **kwargs):
            self.eval_kwargs = kwargs
            return np.zeros(x.shape[:3], dtype=np.uint32), None, None

    fake_model = _RecordingFakeModel()
    engine = CellposeV4Engine(
        model_name="cpsam",
        package_version="0.0.0-fake",
        device="cpu",
        model_checksum=None,
        _model=fake_model,
    )
    prepared = PreparedCellposeInput(
        data=np.zeros((1, 2, 8, 8), dtype=np.float32),
        channel_ids=("membrane",),
        anisotropy=2.0,
        original_shape_zyx=(2, 8, 8),
        resampled_spacing_um_zyx=(0.2, 0.1, 0.1),
        diameter_px=None,
    )
    engine.evaluate(prepared, config=CellposeModelConfig(package_major=4, model_name="cpsam"))

    assert fake_model.eval_kwargs is not None, "engine never called eval()"
    assert fake_model.eval_kwargs.get("z_axis") == 0, (
        "cellpose 4's eval() requires an explicit z_axis= for 3D input "
        "(raises ValueError without it); cellpose 3 has no such argument"
    )
    assert fake_model.eval_kwargs.get("do_3D") is True

    # ── 2. factory rejects a configured/installed major mismatch, for
    #        whichever major is actually installed on this host ───────────
    installed_major = _installed_cellpose_major()
    if installed_major is None:
        pytest.skip(
            "cellpose is not installed; the factory-gating half of this "
            "test needs it to exercise a real mismatch"
        )
    mismatched_major = 3 if installed_major == 4 else 4
    mismatched_config = CellposeModelConfig(
        package_major=mismatched_major, model_name="does-not-matter"
    )
    with pytest.raises(MLDependencyError):
        create_cellpose_engine(mismatched_config)
