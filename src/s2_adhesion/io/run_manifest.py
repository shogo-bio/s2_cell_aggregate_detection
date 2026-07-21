"""``run_manifest.json``: what one ``s2-adhesion run`` did, for provenance.

Records, in one JSON object: a hash of the fully-resolved config (changes iff
any field of the config changes), the resolved config itself (not just the
path to the YAML -- defaults are expanded, so two configs that differ only in
an omitted-vs-explicit field are visibly identical here), input and output
paths, the installed version of every package this pipeline cares about
(best-effort: an absent package records ``null``, never raises), host
platform, start/end timestamps, warnings collected during the run, and a
completion status.

The completion marker is the LAST thing written, full stop. A run that
raised, or was killed, must never leave a ``run_manifest.json`` that claims
``status == "complete"``. This module enforces that structurally rather than
by convention: the entire manifest -- including its final ``status`` -- is
built as one in-memory ``dict`` by :func:`build_run_manifest`, and
:func:`write_run_manifest` performs exactly one write: a temp file followed
by an atomic ``os.replace`` into the target path. There is no intermediate
state where a partial or "in progress" manifest sits at the final path --
either the previous run's manifest is still there, the new (complete-or-
failed, but always fully-formed) one is there, or nothing is. Callers that
want to record a failure call this with ``status="failed"``; callers must
never call it with ``status="complete"`` before every other output for that
run is durably on disk.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import uuid
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from ..config import PipelineConfig

__all__ = [
    "RunStatus",
    "SCHEMA_VERSION_RUN_MANIFEST",
    "TRACKED_PACKAGES",
    "now_utc_iso",
    "config_to_jsonable",
    "config_sha256",
    "package_versions",
    "build_run_manifest",
    "write_run_manifest",
]

RunStatus = Literal["complete", "failed"]

SCHEMA_VERSION_RUN_MANIFEST = "s2-run-manifest/v1"

# Packages this manifest records the installed version of. Deliberately
# includes the ML extras (nd2/cellpose/torch): recording ``null`` for an
# absent package is itself useful provenance (e.g. "this measure-only run
# had no cellpose installed, as expected"). importlib.metadata never imports
# the package itself, so listing them here never pulls torch/cellpose in.
TRACKED_PACKAGES: tuple[str, ...] = (
    "numpy",
    "scipy",
    "scikit-image",
    "pandas",
    "zarr",
    "tifffile",
    "PyYAML",
    "nd2",
    "cellpose",
    "torch",
)


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def config_to_jsonable(value: Any) -> Any:
    """Recursively turn a :class:`PipelineConfig` (or any value nested in it)
    into plain, JSON-serialisable, deterministically-ordered data.

    Frozen dataclasses become dicts keyed by field name (in declaration
    order); ``StrEnum``/``Enum`` members become their ``.value``; ``Path``
    becomes ``str``; ``frozenset``/``set`` become a *sorted* list, so that
    e.g. ``ChannelBinding.roles`` never flips the hash depending on set
    iteration order; tuples/lists become lists; mappings become
    ``str``-keyed dicts. Every other value (str/int/float/bool/None) passes
    through unchanged.
    """
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: config_to_jsonable(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (frozenset, set)):
        return sorted(config_to_jsonable(v) for v in value)
    if isinstance(value, (tuple, list)):
        return [config_to_jsonable(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): config_to_jsonable(v) for k, v in value.items()}
    return value


def config_sha256(config: PipelineConfig) -> str:
    """Stable hash of the fully-resolved config.

    Changes if, and only if, some field of ``config`` (including a nested
    dataclass field, at any depth) changes -- two configs that are `==` after
    :func:`config_to_jsonable` always hash identically regardless of
    construction order, and any actual difference always changes the hash.
    """
    payload = json.dumps(config_to_jsonable(config), sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def package_versions(packages: Sequence[str] = TRACKED_PACKAGES) -> dict[str, str | None]:
    """Installed version of each of ``packages``, or ``None`` if not installed.

    Uses ``importlib.metadata`` only, which reads installed-package metadata
    from disk -- this never imports any of ``packages`` as a side effect, so
    calling it (even with ``torch``/``cellpose`` listed) never pulls them in.
    """
    out: dict[str, str | None] = {}
    for name in packages:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def build_run_manifest(
    *,
    run_id: str,
    config: PipelineConfig,
    input_paths: Sequence[str | Path],
    output_paths: Sequence[str | Path],
    started_utc: str,
    ended_utc: str,
    status: RunStatus,
    warnings: Sequence[str] = (),
    error: str | None = None,
) -> dict[str, Any]:
    """Build the manifest dict in memory. Pure -- touches no disk.

    ``status="failed"`` requires ``error`` to be set (a failed run manifest
    with no recorded reason is not useful); ``status="complete"`` requires
    ``error`` to be ``None`` (a manifest cannot simultaneously claim it
    completed and record a fatal error).
    """
    if status == "failed" and error is None:
        raise ValueError("build_run_manifest(status='failed') requires `error`")
    if status == "complete" and error is not None:
        raise ValueError("build_run_manifest(status='complete') must not carry an `error`")
    return {
        "schema_version": SCHEMA_VERSION_RUN_MANIFEST,
        "run_id": run_id,
        "status": status,
        "config_sha256": config_sha256(config),
        "config": config_to_jsonable(config),
        "input_paths": [str(p) for p in input_paths],
        "output_paths": [str(p) for p in output_paths],
        "package_versions": package_versions(),
        "host_platform": platform.platform(),
        "python_version": sys.version,
        "started_utc": started_utc,
        "ended_utc": ended_utc,
        "warnings": list(warnings),
        "error": error,
    }


def write_run_manifest(manifest: Mapping[str, Any], path: Path | str) -> Path:
    """Write ``manifest`` to ``path`` as one atomic operation.

    ``manifest`` must already be complete -- including its final ``status``
    -- by the time this is called; see the module docstring for why. Writes
    to a temp sibling file first, then ``os.replace``s it into ``path``, so
    the target path is always either absent, holding a previous run's
    manifest, or holding this fully-formed one -- never a half-written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    text = json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=True)
    try:
        tmp_path.write_text(text, encoding="utf-8")
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise
    return path
