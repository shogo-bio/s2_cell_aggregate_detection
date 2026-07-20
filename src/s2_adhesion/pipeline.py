"""Backend-neutral run orchestration.

This module knows nothing about cellpose, torch, or the legacy 2D threshold
algorithm. It only knows how to find, by id, the one object that actually
knows how to run a whole configured pipeline over a
:class:`~s2_adhesion.contracts.VolumeSource`: an "analysis backend".

Established backend shape (frozen by ``backends/ml3d.py``, matched by
``backends/legacy.py`` -- there is no ``backends/protocol.py`` on this
branch, confirmed by directory listing before writing this module): any
object exposing

    backend_id: str
    def run(self, source: VolumeSource, *, config: PipelineConfig,
             output_dir: Path) -> <backend-specific RunArtifacts>

``run`` does its OWN field iteration internally (each backend already needs
per-field control -- e.g. writing an image artifact before segmenting it) --
this module does not iterate ``source`` a second time, it only resolves the
backend and calls ``run`` once. The two existing backends' return types are
NOT identical (``backends.ml3d.MLInstance3DBackend`` returns
``commands.measure.RunArtifacts``; ``backends.legacy.LegacyThreshold2DBackend``
returns its own module-local ``RunArtifacts``) -- a real seam between two
independently-written modules, not something this file can paper over
without editing either. What every caller of :func:`run_pipeline` actually
needs (warnings, written paths, for ``io.run_manifest``) is extracted
best-effort, defensively, in :func:`_extract_warnings` /
:func:`_extract_output_paths`; the raw backend-native result is always kept
too, on :attr:`RunArtifacts.backend_result`, for callers that need the full
detail.

``BACKEND_CLASSES`` -- id -> (dotted module, class name) -- is the only place
that knows those module/class names. Backend modules are imported lazily,
inside :func:`resolve_backend`, never at the top of this file: the point is
that ``s2_adhesion.pipeline`` stays importable, and ``config.analysis_backend``
stays *resolvable* (its id is always knowable/validatable), on a machine with
no ML stack installed at all. Only actually *running* the ml_instance_3d
backend pays for importing torch/cellpose, and only at the moment it's
needed (``backends.ml3d`` imports ``commands.segment`` ->
``segmentation.factory``, which imports cellpose/torch lazily, inside
``create_cellpose_engine``, only when segmentation is actually invoked).
"""

from __future__ import annotations

import importlib
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .config import PipelineConfig
from .contracts import VolumeSource
from .errors import ConfigError, S2AdhesionError

__all__ = [
    "UnknownBackendError",
    "BackendUnavailableError",
    "AnalysisBackend",
    "RunArtifacts",
    "BACKEND_CLASSES",
    "resolve_backend",
    "run_pipeline",
]


class UnknownBackendError(ConfigError):
    """``analysis_backend`` names an id with no registered backend.

    A :class:`~s2_adhesion.errors.ConfigError` subclass: an unknown backend id
    is a configuration mistake, not a runtime/environment problem, and is
    reported the same clear, traceback-free way any other bad config is.
    """


class BackendUnavailableError(S2AdhesionError):
    """``backend_id`` is valid, but its backend isn't usable right now.

    Covers: the backend module doesn't exist yet, it exists but its expected
    class isn't there yet, the class can't be constructed with no arguments,
    or the constructed instance's ``backend_id`` doesn't match the id it was
    looked up under.
    """


# analysis_backend id (see config.PipelineConfig) -> (dotted backend module,
# class name). The only place that knows these.
BACKEND_CLASSES: dict[str, tuple[str, str]] = {
    "ml_instance_3d": ("s2_adhesion.backends.ml3d", "MLInstance3DBackend"),
    "legacy_threshold_2d": ("s2_adhesion.backends.legacy", "LegacyThreshold2DBackend"),
}


class AnalysisBackend(Protocol):
    """The shape every analysis backend on this branch follows."""

    backend_id: str

    def run(
        self, source: VolumeSource, *, config: PipelineConfig, output_dir: Path
    ) -> Any: ...


def resolve_backend(backend_id: str) -> AnalysisBackend:
    """Import, construct and return the :class:`AnalysisBackend` for
    ``backend_id``.

    Raises :class:`UnknownBackendError` if ``backend_id`` is not a registered
    id at all (the message lists every valid id). Raises
    :class:`BackendUnavailableError` if it *is* a valid id but can't be used
    right now -- module not written yet, class missing, class not
    zero-arg-constructible, or a mismatched declared ``backend_id``.

    A :class:`ModuleNotFoundError` for anything other than the backend module
    itself (e.g. a missing ``torch``/``cellpose`` the backend needs to
    actually run) is a real, honest dependency problem and is left to
    propagate rather than being reported as "unavailable".
    """
    if backend_id not in BACKEND_CLASSES:
        raise UnknownBackendError(
            f"unknown analysis_backend {backend_id!r}. Valid ids: "
            f"{sorted(BACKEND_CLASSES)}"
        )
    module_name, class_name = BACKEND_CLASSES[backend_id]
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            raise BackendUnavailableError(
                f"backend {backend_id!r} ({module_name}) is not available yet "
                "in this build."
            ) from exc
        raise

    backend_cls = getattr(module, class_name, None)
    if backend_cls is None:
        raise BackendUnavailableError(
            f"backend module {module_name} has no {class_name!r} yet."
        )
    try:
        backend = backend_cls()
    except TypeError as exc:
        raise BackendUnavailableError(
            f"{module_name}.{class_name} could not be constructed with no "
            f"arguments: {exc}"
        ) from exc

    declared_id = getattr(backend, "backend_id", None)
    if declared_id != backend_id:
        raise BackendUnavailableError(
            f"{module_name}.{class_name} declares backend_id={declared_id!r}, "
            f"but was looked up as {backend_id!r}."
        )
    return backend


def _extract_warnings(backend_result: Any) -> tuple[str, ...]:
    """Best-effort ``warnings`` out of a backend-native result.

    Tries a direct ``.warnings`` attribute first (``commands.measure.
    RunArtifacts``), then ``.bundle.warnings`` (``backends.legacy.
    RunArtifacts``). Never raises -- an unrecognised shape just yields no
    warnings rather than breaking manifest writing.
    """
    warnings = getattr(backend_result, "warnings", None)
    if warnings is not None:
        return tuple(warnings)
    bundle = getattr(backend_result, "bundle", None)
    if bundle is not None:
        return tuple(getattr(bundle, "warnings", ()))
    return ()


def _extract_output_paths(backend_result: Any) -> tuple[Path, ...]:
    """Best-effort list of paths a backend-native result claims to have
    written, across the (currently two, differently-shaped) known result
    types. Never raises; an unrecognised shape just yields no paths."""
    paths: list[Path] = []
    for attr in (
        "objects_csv",
        "contacts_csv",
        "aggregates_csv",
        "localization_profiles_csv",
        "metrics_manifest_json",
    ):
        value = getattr(backend_result, attr, None)
        if value is not None:
            paths.append(Path(value))
    label_dirs = getattr(backend_result, "label_artifact_dirs", None)
    if label_dirs:
        paths.extend(Path(p) for p in label_dirs)
    written_paths = getattr(backend_result, "written_paths", None)
    if written_paths:
        paths.extend(Path(p) for p in written_paths.values())
    return tuple(paths)


@dataclass(frozen=True, slots=True)
class RunArtifacts:
    """Everything one ``run`` produced, normalised just enough for
    ``io.run_manifest`` -- plus the untouched backend-native result for
    anyone who needs the full detail.
    """

    run_id: str
    backend_id: str
    output_dir: Path
    warnings: tuple[str, ...] = ()
    output_paths: tuple[Path, ...] = ()
    backend_result: Any = None


def run_pipeline(
    config: PipelineConfig,
    source: VolumeSource,
    output_dir: Path | str,
    *,
    run_id: str | None = None,
    backend: AnalysisBackend | None = None,
) -> RunArtifacts:
    """Run ``source`` through the configured backend and return
    :class:`RunArtifacts`.

    Resolves and calls the backend named by ``config.analysis_backend``
    exactly once -- iteration over ``source``'s fields is the backend's own
    job (it needs fine-grained per-field control this function has no
    reason to duplicate).

    ``backend`` is the injection point for tests -- pass a fake
    :class:`AnalysisBackend` to exercise this function (including a
    simulated mid-run failure) with neither ``backends.ml3d`` nor
    ``backends.legacy`` needing to run for real. Production callers leave it
    ``None``, and the backend is resolved via :func:`resolve_backend` --
    imported lazily, only now, not at module import time.

    Propagates whatever the backend raises without writing anything itself;
    callers that need a "did this complete" record (``cli.py``'s ``run``
    command, via ``io.run_manifest``) are responsible for catching that and
    recording failure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_run_id = run_id if run_id is not None else uuid.uuid4().hex
    analysis_backend = backend if backend is not None else resolve_backend(
        config.analysis_backend
    )

    result = analysis_backend.run(source, config=config, output_dir=output_dir)

    return RunArtifacts(
        run_id=resolved_run_id,
        backend_id=getattr(analysis_backend, "backend_id", config.analysis_backend),
        output_dir=output_dir,
        warnings=_extract_warnings(result),
        output_paths=_extract_output_paths(result),
        backend_result=result,
    )
