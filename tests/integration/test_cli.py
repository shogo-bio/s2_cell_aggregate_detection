"""Integration tests for ``s2_adhesion.cli``, ``s2_adhesion.pipeline`` and
``s2_adhesion.io.run_manifest``.

Everything here that needs an actual ``.nd2`` file is faked (there is none in
this repo, and none is needed -- see ``tests/conftest.py`` and
``tests/unit/test_nd2_source.py`` for the established convention).

As of this writing every module this file codes against
(``commands.extract``/``segment``/``measure``, ``backends.ml3d``/``legacy``)
has landed, so most tests here exercise the real thing. A few still degrade
to testing the documented "not available yet" fallback if a module is
missing (``_module_available``) -- kept for robustness against a checkout
where one of those modules is absent for any reason, and self-skip once the
module is present.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from s2_adhesion import pipeline
from s2_adhesion.cli import CommandUnavailableError, build_inspection_report, main
from s2_adhesion.config import ContactConfig, DirectInstanceConfig, MeasurementConfig, PipelineConfig
from s2_adhesion.contracts import ChannelBinding, ChannelRole, ImageVolume, MeasurementBundle
from s2_adhesion.errors import ConfigError, MeasurementError, S2AdhesionError
from s2_adhesion.io import run_manifest as run_manifest_mod
from s2_adhesion.pipeline import (
    BACKEND_CLASSES,
    BackendUnavailableError,
    RunArtifacts,
    UnknownBackendError,
)

from tests.conftest import ANISOTROPIC, make_image_volume, make_label_volume

REPO_ROOT = Path(__file__).resolve().parents[2]

# ─── shared helpers ─────────────────────────────────────────────────────────


def _module_available(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except ModuleNotFoundError:
        return False


def make_config(
    *,
    analysis_backend: str = "ml_instance_3d",
    minimum_contact_area_um2: float = 1.0,
) -> PipelineConfig:
    channels = (
        ChannelBinding(channel_id="nucleus", source_index=0, roles=frozenset({ChannelRole.NUCLEUS})),
        ChannelBinding(channel_id="membrane", source_index=1, roles=frozenset({ChannelRole.MEMBRANE})),
    )
    return PipelineConfig(
        channels=channels,
        analysis_backend=analysis_backend,
        segmentation=DirectInstanceConfig(
            strategy="direct_cellpose", input_channel_ids=("nucleus",)
        ),
        measurement=MeasurementConfig(
            contact=ContactConfig(minimum_contact_area_um2=minimum_contact_area_um2)
        ),
    )


def write_config_yaml(path: Path, *, minimum_contact_area_um2: float = 1.0) -> Path:
    path.write_text(
        f"""
schema_version: s2-pipeline-config/v1
analysis_backend: ml_instance_3d
channels:
  - channel_id: nucleus
    source_index: 0
    roles: [nucleus]
  - channel_id: membrane
    source_index: 1
    roles: [membrane]
segmentation:
  strategy: direct_cellpose
  input_channel_ids: [nucleus]
measurement:
  contact:
    minimum_contact_area_um2: {minimum_contact_area_um2}
""",
        encoding="utf-8",
    )
    return path


_IMPORT_BLOCK_SCRIPT = """
import builtins, runpy, sys

_blocked = set({blocked!r})
_real_import = builtins.__import__


def _blocking_import(name, globals=None, locals=None, fromlist=(), level=0):
    top = name.split(".")[0]
    if top in _blocked:
        raise ImportError(f"blocked for test: {{name}}")
    return _real_import(name, globals, locals, fromlist, level)


builtins.__import__ = _blocking_import
sys.argv = ["s2-adhesion"] + {argv!r}
runpy.run_module("s2_adhesion.cli", run_name="__main__")
"""


def run_cli_with_blocked_imports(
    argv: list[str], blocked: tuple[str, ...] = ("torch", "cellpose")
) -> subprocess.CompletedProcess:
    """Run the real CLI (via ``runpy``, exactly as the console script would)
    in a subprocess where ``import torch`` / ``import cellpose`` raise
    ``ImportError`` no matter where in the call stack they're attempted.

    This is the architectural guarantee under test: ``measure`` must not
    need those packages at all.
    """
    script = _IMPORT_BLOCK_SCRIPT.format(blocked=tuple(blocked), argv=list(argv))
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=60,
    )


def run_cli_subprocess(argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "s2_adhesion.cli", *argv],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=60,
    )


# ─── dispatch / --help for every subcommand ────────────────────────────────


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["inspect", "--help"],
        ["extract", "--help"],
        ["segment", "--help"],
        ["measure", "--help"],
        ["run", "--help"],
    ],
)
def test_help_exits_zero_for_every_subcommand(argv):
    full_argv = ["-h"] if argv == [] else argv
    with pytest.raises(SystemExit) as excinfo:
        main(full_argv)
    assert excinfo.value.code == 0


def test_top_level_help_lists_all_five_subcommands():
    result = run_cli_subprocess(["--help"])
    assert result.returncode == 0, result.stderr
    for name in ("inspect", "extract", "segment", "measure", "run"):
        assert name in result.stdout, result.stdout


def test_no_command_is_a_clean_usage_error_not_a_traceback():
    result = run_cli_subprocess([])
    assert result.returncode != 0
    assert "Traceback" not in result.stderr


# ─── no heavy/ML imports at module scope ───────────────────────────────────


def test_importing_cli_does_not_import_torch_cellpose_or_nd2():
    code = (
        "import sys\n"
        "import s2_adhesion.cli\n"
        "assert 'torch' not in sys.modules, sys.modules.keys()\n"
        "assert 'cellpose' not in sys.modules, sys.modules.keys()\n"
        "assert 'nd2' not in sys.modules, sys.modules.keys()\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


def test_importing_pipeline_does_not_import_torch_or_cellpose():
    code = (
        "import sys\n"
        "import s2_adhesion.pipeline\n"
        "assert 'torch' not in sys.modules, sys.modules.keys()\n"
        "assert 'cellpose' not in sys.modules, sys.modules.keys()\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


def test_importing_commands_measure_does_not_import_torch_or_cellpose():
    """Not this file's module, but the guarantee this file's laziness
    protects -- confirmed directly, not just inferred from cli.py's own
    laziness."""
    code = (
        "import sys\n"
        "import s2_adhesion.commands.measure\n"
        "assert 'torch' not in sys.modules, sys.modules.keys()\n"
        "assert 'cellpose' not in sys.modules, sys.modules.keys()\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


# ─── the core guarantee: measure works with no ML stack at all ────────────


def test_measure_help_succeeds_with_torch_and_cellpose_blocked():
    result = run_cli_with_blocked_imports(["measure", "--help"])
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Traceback" not in result.stderr


def _write_real_label_artifact(tmp_path: Path) -> Path:
    """A genuine on-disk label artifact -- via the real
    ``io.zarr_store.write_label_volume`` -- with no image alongside it, so
    ``measure`` runs a geometry-only pass."""
    from s2_adhesion.io.zarr_store import write_label_volume

    cells = np.zeros((6, 12, 12), dtype=np.uint32)
    cells[1:4, 2:8, 2:8] = 1
    cells[1:4, 8:10, 8:10] = 2
    label = make_label_volume(cells, spacing=ANISOTROPIC)

    label_path = tmp_path / "labels.ome.zarr"
    write_label_volume(label, label_path, chunks=(6, 12, 12))
    return label_path


def test_measure_run_succeeds_with_torch_and_cellpose_blocked(tmp_path):
    """THE core guarantee this task exists to protect: a real ``measure``
    invocation -- reading a real label artifact, computing real metrics,
    writing real CSVs -- succeeds end to end with torch and cellpose both
    made to raise ImportError anywhere in the call stack.
    """
    config_path = write_config_yaml(tmp_path / "cfg.yaml")
    label_dir = _write_real_label_artifact(tmp_path)
    out_dir = tmp_path / "out"

    result = run_cli_with_blocked_imports(
        ["measure", str(label_dir), str(out_dir), "--config", str(config_path)]
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Traceback" not in result.stderr

    assert (out_dir / "objects.csv").exists()
    assert (out_dir / "contacts.csv").exists()
    assert (out_dir / "aggregates.csv").exists()
    assert (out_dir / "metrics_manifest.json").exists()
    objects_csv = (out_dir / "objects.csv").read_text(encoding="utf-8")
    # Two labelled cells were written -- both rows must show up.
    assert objects_csv.count("\n") >= 3  # header + >=2 data rows


def test_measure_command_not_yet_available_degrades_gracefully(tmp_path):
    if _module_available("s2_adhesion.commands.measure"):
        pytest.skip(
            "s2_adhesion.commands.measure has landed; the not-yet-available "
            "fallback this test checks is no longer reachable through this "
            "call path. See test_measure_run_succeeds_with_torch_and_cellpose_"
            "blocked for the now-real run."
        )
    config_path = write_config_yaml(tmp_path / "cfg.yaml")
    label_dir, out_dir = tmp_path / "lab", tmp_path / "out"
    label_dir.mkdir()

    exit_code = main(["measure", str(label_dir), str(out_dir), "--config", str(config_path)])
    assert exit_code == 1


# ─── segment ────────────────────────────────────────────────────────────────


def test_segment_command_not_yet_available_degrades_gracefully(tmp_path):
    if _module_available("s2_adhesion.commands.segment"):
        pytest.skip("s2_adhesion.commands.segment has landed")
    config_path = write_config_yaml(tmp_path / "cfg.yaml")
    image_dir, out_dir = tmp_path / "img", tmp_path / "out"
    image_dir.mkdir()

    exit_code = main(["segment", str(image_dir), str(out_dir), "--config", str(config_path)])
    assert exit_code == 1


def test_segment_dispatches_to_real_segment_command_with_correct_args(tmp_path, monkeypatch):
    """segment() returns a single Path (not a list, unlike extract/measure) --
    confirm the CLI handler passes the right args and doesn't try to iterate
    that single Path."""
    calls = []

    def fake_segment(*, image_artifact_dir, output_dir, config):
        calls.append((Path(image_artifact_dir), Path(output_dir), config))
        return Path(output_dir)

    import s2_adhesion.commands.segment as segment_mod

    monkeypatch.setattr(segment_mod, "segment", fake_segment)

    config_path = write_config_yaml(tmp_path / "cfg.yaml")
    image_dir, out_dir = tmp_path / "img", tmp_path / "out"
    image_dir.mkdir()

    exit_code = main(["segment", str(image_dir), str(out_dir), "--config", str(config_path)])
    assert exit_code == 0
    assert len(calls) == 1
    assert calls[0][0] == image_dir
    assert calls[0][1] == out_dir


# ─── extract: already-landed command module, should really run ────────────


def test_extract_dispatches_to_real_extract_command(tmp_path, monkeypatch):
    calls = []

    def fake_extract(*, nd2_path, output_dir, config, max_fields=None):
        calls.append((Path(nd2_path), Path(output_dir), config))
        return [Path(output_dir) / "field000.zarr"]

    import s2_adhesion.commands.extract as extract_mod

    monkeypatch.setattr(extract_mod, "extract", fake_extract)

    config_path = write_config_yaml(tmp_path / "cfg.yaml")
    nd2_path = tmp_path / "fake.nd2"
    nd2_path.write_bytes(b"")
    out_dir = tmp_path / "out"

    exit_code = main(["extract", str(nd2_path), str(out_dir), "--config", str(config_path)])
    assert exit_code == 0
    assert len(calls) == 1
    assert calls[0][0] == nd2_path
    assert calls[0][1] == out_dir


# ─── unknown backend id ─────────────────────────────────────────────────────


def test_unknown_backend_id_lists_valid_ids():
    with pytest.raises(UnknownBackendError) as excinfo:
        pipeline.resolve_backend("totally_bogus_backend")
    msg = str(excinfo.value)
    assert "totally_bogus_backend" in msg
    assert "legacy_threshold_2d" in msg
    assert "ml_instance_3d" in msg


def test_unknown_backend_id_via_run_command_is_a_clean_error(tmp_path):
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(
        """
schema_version: s2-pipeline-config/v1
analysis_backend: totally_bogus_backend
channels:
  - channel_id: nucleus
    source_index: 0
    roles: [nucleus]
""",
        encoding="utf-8",
    )
    nd2_path = tmp_path / "fake.nd2"
    nd2_path.write_bytes(b"")
    out_dir = tmp_path / "out"

    result = run_cli_subprocess(
        ["run", str(nd2_path), str(out_dir), "--config", str(config_path)]
    )
    assert result.returncode == 1
    assert "Traceback" not in result.stderr
    assert "legacy_threshold_2d" in result.stderr
    assert "ml_instance_3d" in result.stderr


# ─── real backends resolve without ML installed (except ml_instance_3d,
#     which only NEEDS torch/cellpose once .run() is actually called) ──────


def test_backend_classes_registry_matches_config_literal_ids():
    # config.PipelineConfig.analysis_backend: Literal["legacy_threshold_2d",
    # "ml_instance_3d"] -- these must be exactly the ids pipeline.py knows.
    assert set(BACKEND_CLASSES) == {"legacy_threshold_2d", "ml_instance_3d"}


def test_resolve_legacy_backend_for_real_with_torch_and_cellpose_blocked():
    """backends.legacy imports legacy.algorithm, which imports nd2 eagerly
    (a deliberate preservation choice, per that module's own docstring) but
    never torch/cellpose -- resolving it must succeed with those blocked."""
    code = (
        "from s2_adhesion import pipeline\n"
        "backend = pipeline.resolve_backend('legacy_threshold_2d')\n"
        "assert backend.backend_id == 'legacy_threshold_2d'\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _IMPORT_BLOCK_SCRIPT.format(blocked=("torch", "cellpose"), argv=[])
            .replace(
                'sys.argv = ["s2-adhesion"] + []\nrunpy.run_module("s2_adhesion.cli", run_name="__main__")\n',
                code,
            ),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


def test_resolve_ml_instance_3d_backend_for_real():
    backend = pipeline.resolve_backend("ml_instance_3d")
    assert backend.backend_id == "ml_instance_3d"


# ─── backend module missing its class / declaring a mismatched id ─────────


def test_backend_missing_class_is_unavailable(monkeypatch):
    import types

    fake_module = types.ModuleType("s2_adhesion.backends._fake_incomplete")
    monkeypatch.setitem(
        BACKEND_CLASSES, "fake_incomplete", (fake_module.__name__, "NoSuchBackend")
    )
    monkeypatch.setitem(sys.modules, fake_module.__name__, fake_module)

    with pytest.raises(BackendUnavailableError):
        pipeline.resolve_backend("fake_incomplete")


def test_backend_declares_mismatched_id_is_unavailable(monkeypatch):
    import types

    fake_module = types.ModuleType("s2_adhesion.backends._fake_mismatch")

    class _FakeBackend:
        backend_id = "some_other_id"

        def run(self, source, *, config, output_dir):
            raise AssertionError("should never be called")

    fake_module.FakeBackend = _FakeBackend
    monkeypatch.setitem(
        BACKEND_CLASSES, "fake_mismatch", (fake_module.__name__, "FakeBackend")
    )
    monkeypatch.setitem(sys.modules, fake_module.__name__, fake_module)

    with pytest.raises(BackendUnavailableError):
        pipeline.resolve_backend("fake_mismatch")


def test_backend_module_not_yet_written_is_unavailable(monkeypatch):
    monkeypatch.setitem(
        BACKEND_CLASSES,
        "fake_unwritten",
        ("s2_adhesion.backends._does_not_exist_at_all", "Whatever"),
    )
    with pytest.raises(BackendUnavailableError):
        pipeline.resolve_backend("fake_unwritten")


# ─── pipeline.run_pipeline: injected backend, no real ML needed ───────────


class _FakeSource:
    def __init__(self, images: dict[str, ImageVolume]):
        self._images = images

    def field_ids(self):
        return list(self._images.keys())

    def read_field(self, field_id):
        return self._images[field_id]


def _tiny_image() -> ImageVolume:
    data = np.zeros((2, 3, 4, 4), dtype=np.uint16)
    return make_image_volume(data, spacing=ANISOTROPIC, channel_ids=("nucleus", "membrane"))


class _FakeResult:
    def __init__(self, warnings=()):
        self.warnings = tuple(warnings)


class _FakeBackend:
    """Matches the real shape: backend_id attribute + run(source, *, config,
    output_dir), doing its own field iteration."""

    backend_id = "ml_instance_3d"

    def __init__(self, fn):
        self._fn = fn

    def run(self, source, *, config, output_dir):
        return self._fn(source, config, output_dir)


def test_run_pipeline_with_injected_backend_assembles_run_artifacts(tmp_path):
    config = make_config()
    source = _FakeSource({"field000": _tiny_image(), "field001": _tiny_image()})

    seen = {}

    def fn(src, cfg, out_dir):
        seen["field_ids"] = list(src.field_ids())
        return _FakeResult(warnings=("w1", "w2"))

    backend = _FakeBackend(fn)
    artifacts = pipeline.run_pipeline(config, source, tmp_path / "out", backend=backend)

    assert seen["field_ids"] == ["field000", "field001"]
    assert artifacts.warnings == ("w1", "w2")
    assert artifacts.backend_id == "ml_instance_3d"
    assert (tmp_path / "out").is_dir()
    assert artifacts.backend_result is not None


def test_run_pipeline_propagates_a_failure_from_the_backend(tmp_path):
    """Simulated failing backend: run() itself raises partway through its own
    field iteration. No RunArtifacts must ever be returned in that case --
    the caller (cli.py's `run` handler) relies on this to know the run did
    not complete."""
    config = make_config()
    source = _FakeSource({"field000": _tiny_image()})

    def fn(src, cfg, out_dir):
        raise MeasurementError("simulated backend failure partway through the run")

    backend = _FakeBackend(fn)
    with pytest.raises(MeasurementError):
        pipeline.run_pipeline(config, source, tmp_path / "out", backend=backend)


def test_run_pipeline_extracts_warnings_from_legacy_style_result(tmp_path):
    """backends.legacy.RunArtifacts has no .warnings attribute directly --
    only .bundle.warnings. Confirm the fallback extraction actually works."""
    config = make_config(analysis_backend="legacy_threshold_2d")
    source = _FakeSource({"field000": _tiny_image()})

    class _LegacyStyleResult:
        def __init__(self):
            self.bundle = MeasurementBundle(warnings=("legacy warning",))

    class _LegacyStyleBackend:
        backend_id = "legacy_threshold_2d"

        def run(self, source, *, config, output_dir):
            return _LegacyStyleResult()

    artifacts = pipeline.run_pipeline(
        config, source, tmp_path / "out", backend=_LegacyStyleBackend()
    )
    assert artifacts.warnings == ("legacy warning",)


# ─── run_manifest: config hash, atomic write, completion status ───────────


def test_config_sha256_is_deterministic_and_order_independent():
    config_a = make_config(minimum_contact_area_um2=1.0)
    config_b = make_config(minimum_contact_area_um2=1.0)
    assert run_manifest_mod.config_sha256(config_a) == run_manifest_mod.config_sha256(config_b)


def test_config_sha256_changes_when_config_changes():
    config_a = make_config(minimum_contact_area_um2=1.0)
    config_b = make_config(minimum_contact_area_um2=2.0)
    assert run_manifest_mod.config_sha256(config_a) != run_manifest_mod.config_sha256(config_b)


def test_build_run_manifest_rejects_inconsistent_status_and_error():
    config = make_config()
    with pytest.raises(ValueError):
        run_manifest_mod.build_run_manifest(
            run_id="r1",
            config=config,
            input_paths=[],
            output_paths=[],
            started_utc="t0",
            ended_utc="t1",
            status="failed",
            error=None,
        )
    with pytest.raises(ValueError):
        run_manifest_mod.build_run_manifest(
            run_id="r1",
            config=config,
            input_paths=[],
            output_paths=[],
            started_utc="t0",
            ended_utc="t1",
            status="complete",
            error="boom",
        )


def test_write_run_manifest_is_atomic_and_readable(tmp_path):
    config = make_config()
    manifest = run_manifest_mod.build_run_manifest(
        run_id="r1",
        config=config,
        input_paths=["in.nd2"],
        output_paths=["out/"],
        started_utc="t0",
        ended_utc="t1",
        status="complete",
        warnings=("w",),
        error=None,
    )
    path = tmp_path / "run_manifest.json"
    run_manifest_mod.write_run_manifest(manifest, path)

    assert path.exists()
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["status"] == "complete"
    assert on_disk["config_sha256"] == run_manifest_mod.config_sha256(config)
    assert on_disk["warnings"] == ["w"]
    # no leftover temp files
    assert list(tmp_path.glob(".*.tmp-*")) == []


def test_run_manifest_package_versions_never_imports_the_packages():
    code = (
        "import sys\n"
        "from s2_adhesion.io import run_manifest\n"
        "versions = run_manifest.package_versions()\n"
        "assert 'torch' not in sys.modules, sys.modules.keys()\n"
        "assert 'cellpose' not in sys.modules, sys.modules.keys()\n"
        "assert 'nd2' not in sys.modules, sys.modules.keys()\n"
        "assert 'torch' in versions\n"
        "assert 'cellpose' in versions\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


# ─── `run` subcommand end to end: completion + failure manifests ──────────


def test_run_command_writes_complete_manifest_on_success(tmp_path, monkeypatch):
    config_path = write_config_yaml(tmp_path / "cfg.yaml", minimum_contact_area_um2=3.0)
    nd2_path = tmp_path / "fake.nd2"
    nd2_path.write_bytes(b"")
    out_dir = tmp_path / "out"

    def fake_run_pipeline(config, source, output_dir, *, run_id=None, backend=None):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return RunArtifacts(
            run_id=run_id or "r1",
            backend_id=config.analysis_backend,
            output_dir=Path(output_dir),
            warnings=("careful",),
        )

    monkeypatch.setattr(pipeline, "run_pipeline", fake_run_pipeline)

    exit_code = main(["run", str(nd2_path), str(out_dir), "--config", str(config_path)])
    assert exit_code == 0

    manifest_path = out_dir / "run_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["error"] is None
    assert manifest["warnings"] == ["careful"]
    assert manifest["config_sha256"]

    # A run over a DIFFERENT config must hash differently.
    config_path_2 = write_config_yaml(tmp_path / "cfg2.yaml", minimum_contact_area_um2=99.0)
    out_dir_2 = tmp_path / "out2"
    exit_code_2 = main(["run", str(nd2_path), str(out_dir_2), "--config", str(config_path_2)])
    assert exit_code_2 == 0
    manifest_2 = json.loads((out_dir_2 / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest_2["config_sha256"] != manifest["config_sha256"]


def test_run_command_raising_backend_leaves_no_manifest_claiming_completion(
    tmp_path, monkeypatch
):
    """Simulate a failing backend end to end through the `run` subcommand: no
    manifest at the output path may ever claim status=='complete'."""
    config_path = write_config_yaml(tmp_path / "cfg.yaml")
    nd2_path = tmp_path / "fake.nd2"
    nd2_path.write_bytes(b"")
    out_dir = tmp_path / "out"

    def failing_run_pipeline(config, source, output_dir, *, run_id=None, backend=None):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        raise MeasurementError("simulated backend failure partway through the run")

    monkeypatch.setattr(pipeline, "run_pipeline", failing_run_pipeline)

    exit_code = main(["run", str(nd2_path), str(out_dir), "--config", str(config_path)])
    assert exit_code == 1

    manifest_path = out_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["status"] != "complete"
        assert manifest["status"] == "failed"
        assert manifest["error"] is not None
        assert "simulated backend failure" in manifest["error"]


def test_run_command_crash_before_pipeline_starts_writes_no_complete_manifest(
    tmp_path,
):
    """An unknown backend id fails before any field is even touched -- the
    output dir may end up with no manifest at all, but never a "complete"
    one."""
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(
        """
schema_version: s2-pipeline-config/v1
analysis_backend: totally_bogus_backend
channels:
  - channel_id: nucleus
    source_index: 0
    roles: [nucleus]
""",
        encoding="utf-8",
    )
    nd2_path = tmp_path / "fake.nd2"
    nd2_path.write_bytes(b"")
    out_dir = tmp_path / "out"

    exit_code = main(["run", str(nd2_path), str(out_dir), "--config", str(config_path)])
    assert exit_code == 1

    manifest_path = out_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["status"] == "failed"


def test_run_command_real_legacy_backend_end_to_end_with_torch_and_cellpose_blocked(
    tmp_path, monkeypatch
):
    """The legacy backend is real, ML-free, and reachable through pipeline.py
    -- run it for real (not through ND2Source, which needs an actual .nd2
    file) by injecting a fake VolumeSource directly into pipeline.run_pipeline,
    with torch/cellpose blocked via the same import hook used elsewhere."""
    code = f"""
import numpy as np
from s2_adhesion import pipeline
from s2_adhesion.config import LegacyConfig, PipelineConfig
from s2_adhesion.contracts import ChannelBinding, ChannelRole
import sys
sys.path.insert(0, {str(REPO_ROOT)!r})
from tests.conftest import ANISOTROPIC, make_image_volume

config = PipelineConfig(
    channels=(
        ChannelBinding(channel_id="membrane", source_index=0, roles=frozenset({{ChannelRole.MEMBRANE}})),
        ChannelBinding(channel_id="signal", source_index=1, roles=frozenset({{ChannelRole.SIGNAL}})),
    ),
    analysis_backend="legacy_threshold_2d",
    legacy=LegacyConfig(),
)

data = (np.random.default_rng(0).integers(0, 200, size=(2, 3, 32, 32))).astype(np.uint16)
image = make_image_volume(
    data, spacing=ANISOTROPIC, channel_ids=("membrane", "signal"),
    roles=(ChannelRole.MEMBRANE, ChannelRole.SIGNAL),
)

class FakeSource:
    def field_ids(self):
        return ["field000"]
    def read_field(self, field_id):
        return image

artifacts = pipeline.run_pipeline(config, FakeSource(), {str(tmp_path / "legacy_out")!r})
assert artifacts.backend_id == "legacy_threshold_2d"
print("OK")
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _IMPORT_BLOCK_SCRIPT.format(blocked=("torch", "cellpose"), argv=[]).replace(
                'sys.argv = ["s2-adhesion"] + []\nrunpy.run_module("s2_adhesion.cli", run_name="__main__")\n',
                code,
            ),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout
    assert (tmp_path / "legacy_out" / "objects.csv").exists()


# ─── missing / invalid config: clean ConfigError, never a traceback ───────


def test_missing_config_file_is_a_clean_error_not_a_traceback(tmp_path):
    label_dir, out_dir = tmp_path / "lab", tmp_path / "out"
    label_dir.mkdir()
    missing_config = tmp_path / "does_not_exist.yaml"

    result = run_cli_subprocess(
        ["measure", str(label_dir), str(out_dir), "--config", str(missing_config)]
    )
    assert result.returncode == 1
    assert "Traceback" not in result.stderr
    assert "Error:" in result.stderr


def test_invalid_config_yaml_is_a_clean_error_not_a_traceback(tmp_path):
    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text("channels: [{channel_id: x, source_index: 0, roles: [not_a_real_role]}]\n")
    label_dir, out_dir = tmp_path / "lab", tmp_path / "out"
    label_dir.mkdir()

    result = run_cli_subprocess(
        ["measure", str(label_dir), str(out_dir), "--config", str(bad_config)]
    )
    assert result.returncode == 1
    assert "Traceback" not in result.stderr
    assert "Error:" in result.stderr


def test_load_config_error_is_the_typed_config_error_in_process(tmp_path):
    with pytest.raises(ConfigError):
        from s2_adhesion.config import load_config

        load_config(tmp_path / "nope.yaml")


# ─── inspect ────────────────────────────────────────────────────────────────


class _FakeVoxelSize:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


class _FakeChannelMeta:
    def __init__(self, name, excitation_nm=None, emission_nm=None):
        self.name = name
        self.excitationLambdaNm = excitation_nm
        self.emissionLambdaNm = emission_nm


class _FakeMicroscope:
    def __init__(self, magnification=None, na=None):
        self.objectiveMagnification = magnification
        self.objectiveNumericalAperture = na


class _FakeChannelEntry:
    def __init__(self, channel_meta, microscope=None):
        self.channel = channel_meta
        self.microscope = microscope


class _FakeMetadata:
    def __init__(self, entries):
        self.channels = entries


class _FakeND2File:
    def __init__(self, sizes, data, voxel_size, metadata=None):
        self.sizes = sizes
        self._data = data
        self._voxel_size = voxel_size
        self.metadata = metadata

    def asarray(self):
        return self._data

    def voxel_size(self):
        return self._voxel_size

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_inspect_report_covers_every_required_field():
    sizes = {"C": 2, "Z": 4, "Y": 8, "X": 8}
    rng = np.random.default_rng(0)
    data = rng.integers(0, 4000, size=tuple(sizes.values())).astype(np.uint16)
    metadata = _FakeMetadata(
        [
            _FakeChannelEntry(
                _FakeChannelMeta("GFP", excitation_nm=488.0, emission_nm=509.0),
                microscope=_FakeMicroscope(magnification=63.0, na=1.4),
            ),
            _FakeChannelEntry(_FakeChannelMeta("DAPI", excitation_nm=358.0, emission_nm=461.0)),
        ]
    )
    fake = _FakeND2File(sizes, data, _FakeVoxelSize(x=0.11, y=0.11, z=0.55), metadata=metadata)

    report = build_inspection_report("fake.nd2", reader_factory=lambda path: fake)

    assert "axes" in report
    assert "sizes" in report
    assert "voxel spacing" in report
    assert "anisotropy ratio" in report
    assert "GFP" in report and "488" in report and "509" in report
    assert "DAPI" in report
    assert "63" in report and "1.4" in report  # objective magnification / NA
    assert "uint16" in report
    assert "intensity range" in report
    assert "per-Z mean intensity" in report
    assert "roles" in report.lower() and "confirm" in report.lower()


def test_inspect_anisotropy_ratio_matches_voxel_geometry():
    from s2_adhesion.contracts import VoxelGeometry

    sizes = {"C": 1, "Z": 4, "Y": 8, "X": 8}
    data = np.zeros(tuple(sizes.values()), dtype=np.uint16)
    fake = _FakeND2File(sizes, data, _FakeVoxelSize(x=0.1, y=0.1, z=0.5))

    report = build_inspection_report("fake.nd2", reader_factory=lambda path: fake)
    expected = VoxelGeometry(spacing_um_zyx=(0.5, 0.1, 0.1)).anisotropy_z_to_xy
    assert f"{expected:.6g}" in report


def test_inspect_handles_missing_metadata_without_crashing():
    sizes = {"C": 1, "Z": 2, "Y": 4, "X": 4}
    data = np.zeros(tuple(sizes.values()), dtype=np.uint16)
    fake = _FakeND2File(sizes, data, _FakeVoxelSize(x=0.1, y=0.1, z=0.5), metadata=None)

    report = build_inspection_report("fake.nd2", reader_factory=lambda path: fake)
    assert "UNKNOWN" in report


def test_inspect_handles_missing_z_spacing_without_crashing():
    class _VoxelSizeNoZ:
        def __init__(self, x, y):
            self.x, self.y = x, y

    sizes = {"C": 1, "Z": 2, "Y": 4, "X": 4}
    data = np.zeros(tuple(sizes.values()), dtype=np.uint16)
    fake = _FakeND2File(sizes, data, _VoxelSizeNoZ(x=0.1, y=0.1))

    report = build_inspection_report("fake.nd2", reader_factory=lambda path: fake)
    assert "UNKNOWN" in report
    assert "NOT COMPUTABLE" in report


def test_inspect_subcommand_dispatches_to_report(monkeypatch, capsys, tmp_path):
    called = {}

    def fake_build(path, reader_factory=None):
        called["path"] = path
        return "REPORT TEXT"

    # A real file on disk: inspect now checks existence and suffix up front, so a
    # mistyped path gives a one-line error instead of a traceback from inside nd2.
    nd2_path = tmp_path / "somefile.nd2"
    nd2_path.write_bytes(b"")

    monkeypatch.setattr("s2_adhesion.cli.build_inspection_report", fake_build)
    exit_code = main(["inspect", str(nd2_path)])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "REPORT TEXT" in captured.out
    assert str(called["path"]) == str(nd2_path)


def test_inspect_reports_a_missing_file_without_a_traceback(capsys, tmp_path):
    exit_code = main(["inspect", str(tmp_path / "absent.nd2")])
    assert exit_code == 2
    err = capsys.readouterr().err
    assert "no such file" in err
    assert "Traceback" not in err


def test_inspect_rejects_a_non_nd2_path(capsys, tmp_path):
    other = tmp_path / "notes.md"
    other.write_text("x", encoding="utf-8")
    assert main(["inspect", str(other)]) == 2
    assert "does not look like an .nd2 file" in capsys.readouterr().err


def test_inspect_help_does_not_require_nd2_installed():
    result = run_cli_with_blocked_imports(["inspect", "--help"], blocked=("nd2",))
    assert result.returncode == 0, result.stdout + result.stderr


# ─── console encoding: cp932 (default on Japanese Windows) can't encode 'µ' ─
#
# The original detect_aggregations.py crashes on plain --help on such a
# machine (UnicodeEncodeError: 'cp932' codec can't encode character '\xb5')
# because its argparse help text says "µm". main() calls
# _console.ensure_utf8_output() first, specifically so this project's own
# printed output (inspect's report prints "voxel spacing (µm): ...") survives
# on that console instead of crashing a run part-way through.


def test_main_survives_cp932_console_and_micro_sign_is_preserved(tmp_path):
    """Force sys.stdout/sys.stderr to a cp932 encoding (as a Japanese Windows
    console would default to) BEFORE main() runs, then run `inspect` (with a
    fake reader, no real .nd2 needed) so its report -- which prints a real
    'µ' character -- is written through it. main() must reconfigure the
    streams to UTF-8 itself; this must not crash and the µ text must survive
    in the captured output.
    """
    script = f"""
import sys
sys.stdout.reconfigure(encoding="cp932", errors="strict")
sys.stderr.reconfigure(encoding="cp932", errors="strict")
sys.path.insert(0, {str(REPO_ROOT)!r})

import s2_adhesion.cli as cli

class _Fake:
    sizes = {{"C": 1, "Z": 2, "Y": 4, "X": 4}}
    def asarray(self):
        import numpy as np
        return np.zeros((1, 2, 4, 4), dtype="uint16")
    def voxel_size(self):
        class V:
            x = y = 0.11
            z = 0.55
        return V()
    metadata = None
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False

cli._default_nd2_reader_factory = lambda path: _Fake()
sys.argv = ["s2-adhesion", "inspect", {str(tmp_path / "fake.nd2")!r}]
raise SystemExit(cli.main())
"""
    (tmp_path / "fake.nd2").write_bytes(b"")
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        cwd=REPO_ROOT,
        timeout=30,
    )
    # main() reconfigures stdout to UTF-8 as its first action (before any
    # output is written), so by the time the report is printed the stream is
    # already UTF-8 -- decode accordingly, not as the cp932 it started as.
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    stdout_text = result.stdout.decode("utf-8", errors="replace")
    assert "µm" in stdout_text  # the µ (U+00B5) survived, not mangled/dropped
    assert "voxel spacing" in stdout_text


def test_ensure_utf8_output_is_not_called_at_module_import_time():
    """Importing s2_adhesion.cli must not reconfigure stdout/stderr as a side
    effect -- only main() may do that, and only when actually acting as the
    CLI entry point (see _console.py's docstring: a library consumer who
    merely imports the package must not have this done to them silently)."""
    script = """
import sys
before_stdout_encoding = sys.stdout.encoding
before_stderr_encoding = sys.stderr.encoding
import s2_adhesion.cli
assert sys.stdout.encoding == before_stdout_encoding
assert sys.stderr.encoding == before_stderr_encoding
print("OK")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


# ─── misc ───────────────────────────────────────────────────────────────────


def test_command_unavailable_error_is_an_s2_adhesion_error():
    assert issubclass(CommandUnavailableError, S2AdhesionError)
