"""Builds a :class:`~.protocol.CellposeEngine` for a validated model config.

This module itself must NEVER import ``cellpose`` or ``torch`` -- that is the
whole point of the factory: a machine with no ML packages installed can still
import this module (and everything upstream of it) and only pays the price of
a heavy, version-specific import at the one call site that actually needs a
model, and only after confirming the installed package matches what was
configured.

Flow of :func:`create_cellpose_engine`:

1. Read ``config.package_major`` (3 or 4, frozen in ``config.py``).
2. Resolve the installed ``cellpose`` version via ``importlib.metadata``,
   which reads package metadata without importing the package.
3. If cellpose is absent, or its major version does not match
   ``config.package_major``, raise :class:`MLDependencyError` naming both the
   configured and installed majors -- before any model construction and
   before either adapter module is imported.
4. Only then dynamically import exactly one of ``cellpose_v3`` /
   ``cellpose_v4`` and ask it to build the engine.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
from pathlib import Path

from ..config import CellposeModelConfig, StarDistModelConfig
from ..errors import MLDependencyError
from .protocol import CellposeEngine, StarDistEngine

_ADAPTER_MODULE_BY_MAJOR: dict[int, str] = {
    3: "s2_adhesion.segmentation.cellpose_v3",
    4: "s2_adhesion.segmentation.cellpose_v4",
}


def _installed_cellpose_version() -> str | None:
    """The installed ``cellpose`` distribution version, or None if absent.

    Uses ``importlib.metadata`` only -- this reads installed-package metadata
    from disk and never imports the ``cellpose`` package itself.
    """
    try:
        return importlib.metadata.version("cellpose")
    except importlib.metadata.PackageNotFoundError:
        return None


def _installed_major(version: str) -> int:
    head = version.split(".", 1)[0]
    try:
        return int(head)
    except ValueError as exc:
        raise MLDependencyError(
            f"installed cellpose reports an unparseable version {version!r}"
        ) from exc


def _resolve_model_checksum(config: CellposeModelConfig) -> str | None:
    """SHA-256 of a locally pinned model file, verified against config if set.

    Built-in named models (e.g. ``cyto3``, resolved by cellpose's own cache)
    have no local path at this point and get no checksum -- that is a
    property of cellpose's model cache, not of this pipeline's provenance.
    """
    if config.pretrained_model_path is None:
        return None
    path = Path(config.pretrained_model_path)
    if not path.is_file():
        raise MLDependencyError(
            f"segmentation.model.pretrained_model_path {path} does not exist "
            "or is not a file"
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if (
        config.expected_model_sha256 is not None
        and digest != config.expected_model_sha256
    ):
        raise MLDependencyError(
            f"model checksum mismatch for {path}: configured "
            f"expected_model_sha256={config.expected_model_sha256!r}, "
            f"computed {digest!r}"
        )
    return digest


def create_cellpose_engine(config: CellposeModelConfig) -> CellposeEngine:
    """Build a :class:`CellposeEngine` matching ``config.package_major``.

    Raises :class:`MLDependencyError` if cellpose is not installed, or is
    installed at a different major version than ``config.package_major`` --
    always before importing either version-specific adapter module and before
    any model is constructed.
    """
    installed_version = _installed_cellpose_version()
    if installed_version is None:
        raise MLDependencyError(
            "cellpose is not installed, but segmentation config requests "
            f"package_major={config.package_major} (model={config.model_name!r}). "
            f"Install it with: pip install 's2-adhesion[cellpose{config.package_major}]' "
            f"or 'cellpose>={config.package_major},<{config.package_major + 1}'."
        )

    installed_major = _installed_major(installed_version)
    if installed_major != config.package_major:
        raise MLDependencyError(
            f"segmentation config requests cellpose package_major="
            f"{config.package_major}, but the installed cellpose is "
            f"{installed_version} (major {installed_major}). Either install "
            f"a matching cellpose ('cellpose>={config.package_major},"
            f"<{config.package_major + 1}') or change "
            "CellposeModelConfig.package_major to match what is installed. "
            "The two majors have incompatible model architectures and are "
            "never interchangeable at runtime."
        )

    model_checksum = _resolve_model_checksum(config)

    module_name = _ADAPTER_MODULE_BY_MAJOR[config.package_major]
    adapter = importlib.import_module(module_name)
    return adapter.build_engine(
        config, package_version=installed_version, model_checksum=model_checksum
    )


def create_stardist_engine(config: "StarDistModelConfig") -> "StarDistEngine":
    """Load a StarDist engine, importing the ML stack only at this point.

    Mirrors ``create_cellpose_engine``: the availability check happens here so
    every other module -- and every unit test -- can import freely without
    stardist or tensorflow installed.

    Unlike cellpose there is no major-version gate, because stardist has a single
    supported API surface here. What IS checked, inside ``stardist_3d``, is that
    a model was actually named: running the one published 3D model (a nuclei
    demo) on membrane-labelled cells by accident would give confident, wrong
    instances rather than an error.
    """
    try:
        installed = importlib.metadata.version("stardist")
    except importlib.metadata.PackageNotFoundError as exc:
        raise MLDependencyError(
            "StarDist segmentation was configured but the 'stardist' package is "
            "not installed in this environment.\n"
            "  install with:  uv pip install 'stardist' 'tensorflow'\n"
            "  stardist is TensorFlow-based and cellpose is PyTorch-based; this "
            "project keeps them in separate environments "
            "(dev/envs/stardist-x64 and dev/envs/s2-aggregate-x64)."
        ) from exc

    from . import stardist_3d  # imported here, never at module scope

    engine = stardist_3d.build_engine(config)
    if engine.package_version in ("unknown", None):
        engine.package_version = installed  # type: ignore[misc]
    return engine
