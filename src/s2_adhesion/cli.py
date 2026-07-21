"""``s2-adhesion`` command-line entry point.

Five subcommands: ``inspect``, ``extract``, ``segment``, ``measure``, ``run``.

DISPATCH IS LAZY, ON PURPOSE, AND MUST STAY THAT WAY. Every subcommand's
*handler* imports the module that actually does the work from inside the
handler function -- never at the top of this file. The architectural
guarantee this protects: ``s2-adhesion measure --help`` (and a real
``measure`` run) must work on a machine with no ``torch``/``cellpose``/``nd2``
installed at all, because measurement reads label artifacts that were
computed elsewhere and never needs an ML package. If this module imported
``s2_adhesion.commands.measure`` (or anything it transitively pulls in) at
module scope, merely running ``s2-adhesion --help`` -- let alone
``measure --help`` -- would require the full ML stack, which is exactly
backwards for a laptop-safe measurement command. A test asserts this by
import-hooking ``torch``/``cellpose``/``nd2`` to raise in a subprocess; do
not "tidy" these imports to the top of the file.

Command modules this file calls into (``commands.extract``,
``commands.segment``, ``commands.measure``) and orchestration modules it
calls into (``pipeline``, ``io.run_manifest``) are looked up and imported
only inside the handler that needs them, via :func:`_load_command` /
``pipeline.resolve_backend``. Some of those modules are, as of this writing,
still being written by other agents -- :func:`_load_command` and
:func:`pipeline.resolve_backend` both turn "the module doesn't exist yet"
into a typed, catchable error (:class:`CommandUnavailableError` /
``pipeline.BackendUnavailableError``) reported as a clean one-line message,
never a raw traceback.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from ._console import ensure_utf8_output
from .contracts import VoxelGeometry
from .errors import S2AdhesionError

__all__ = ["main", "build_inspection_report", "CommandUnavailableError"]


class CommandUnavailableError(S2AdhesionError):
    """The module behind a subcommand isn't usable yet on this build.

    Covers both "the module doesn't exist yet" (this codebase is under
    concurrent development -- see the module docstring) and "the module
    exists but doesn't expose the entry point this CLI expects yet". Either
    way, reported the same clean way, never a traceback.
    """


# ─── lazy command-module loading ───────────────────────────────────────────


def _load_command(module_name: str, func_name: str) -> Callable[..., Any]:
    """Import ``module_name`` and return its ``func_name`` callable.

    Raises :class:`CommandUnavailableError` if the module itself does not
    exist yet, or exists but has no ``func_name`` attribute. A
    ``ModuleNotFoundError`` for anything OTHER than ``module_name`` (e.g. a
    missing ``torch``/``cellpose`` the command module needs to actually run)
    is a real, honest dependency problem and is left to propagate -- it must
    never be swallowed and reported as merely "unavailable".
    """
    import importlib

    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            raise CommandUnavailableError(
                f"{module_name} is not available yet in this build; "
                f"'s2-adhesion' cannot run this command until it lands."
            ) from exc
        raise
    func = getattr(module, func_name, None)
    if func is None:
        raise CommandUnavailableError(
            f"{module_name} exists but does not expose a {func_name}(...) "
            "entry point yet."
        )
    return func


# ─── inspect ────────────────────────────────────────────────────────────────
#
# The historical script only ever assigned the three channels display colours
# -- nobody has documented what they actually stain. Until that is resolved
# against the experimental record, every nucleus-, organelle- and
# localization-derived number downstream is unusable. `inspect` is how you
# find out what's actually in a file, with no config and no assumptions about
# channel roles.
#
# Testable without a real .nd2 file: `build_inspection_report` accepts an
# injected `reader_factory` returning anything duck-typing the same surface
# `s2_adhesion.io.nd2_source.Nd2Like` reads (`.sizes`, `.asarray()`,
# `.voxel_size()`, `.metadata`, context-manager protocol). Production callers
# leave `reader_factory` as the default, which imports `nd2` lazily -- the
# only place in this whole file `nd2` is ever imported -- so `inspect` is the
# one command that needs the `nd2` extra.


def _default_nd2_reader_factory(path: Path) -> Any:
    import nd2  # lazy: only `inspect` needs this optional extra

    return nd2.ND2File(path)


def _channel_meta_list(metadata: Any) -> list[dict[str, Any]]:
    """Best-effort per-channel name/wavelength/objective metadata.

    Every field is read via ``getattr(..., None)``: nd2 metadata schemas vary
    across acquisition software versions and some fields are simply absent
    on some files. `inspect`'s whole job is to surface what IS there and be
    honest about what ISN'T -- it must never raise just because one file
    lacks one optional field.
    """
    channels = getattr(metadata, "channels", None) if metadata is not None else None
    if not channels:
        return []
    out: list[dict[str, Any]] = []
    for ch in channels:
        inner = getattr(ch, "channel", None)
        microscope = getattr(ch, "microscope", None)
        out.append(
            {
                "name": getattr(inner, "name", None) if inner is not None else None,
                "excitation_nm": getattr(inner, "excitationLambdaNm", None)
                if inner is not None
                else None,
                "emission_nm": getattr(inner, "emissionLambdaNm", None)
                if inner is not None
                else None,
                "objective_magnification": getattr(
                    microscope, "objectiveMagnification", None
                )
                if microscope is not None
                else None,
                "objective_na": getattr(microscope, "objectiveNumericalAperture", None)
                if microscope is not None
                else None,
            }
        )
    return out


def _select_first_field_czyx(
    raw: NDArray[Any], sizes: Any
) -> NDArray[Any]:
    """Reduce an arbitrarily-ordered nd2 array to one CZYX field/timepoint.

    Every axis other than C/Z/Y/X is collapsed by taking index 0. Unlike
    ``io.nd2_source._to_canonical_czyx`` (used by the real pipeline), this
    never raises on a non-singleton extra axis: this function backs a
    diagnostic report that must stay informative even on files with an
    unusual axis layout (multi-position, multi-timepoint, ...) -- it only
    ever describes the first field/timepoint and says so.
    """
    dim_keys = list(sizes.keys())
    arr = raw
    keys = list(dim_keys)
    for k in [k for k in dim_keys if k not in "CZYX"]:
        axis = keys.index(k)
        arr = np.take(arr, 0, axis=axis)
        keys.pop(axis)
    for k in "CZ":
        if k not in keys:
            arr = arr[np.newaxis, ...]
            keys.insert(0, k)
    missing = [k for k in "YX" if k not in keys]
    if missing:
        raise ValueError(f"nd2 file reports no {missing} axis; sizes={dict(sizes)!r}")
    perm = [keys.index(k) for k in "CZYX"]
    return np.transpose(arr, perm)


def build_inspection_report(
    path: str | Path, reader_factory: Callable[[Path], Any] | None = None
) -> str:
    """Human-readable report on one nd2 file: everything needed to write a
    channel-role config, and nothing that pretends to already know it.

    Prints (see the docstring of this module and the task this implements):
    axes and sizes, voxel spacing x/y/z, the Z/XY anisotropy ratio (usable
    directly as cellpose's ``anisotropy`` argument), per-channel name and
    excitation/emission wavelength where available, objective magnification
    and NA where available, dtype, per-channel intensity ranges, and a
    per-Z mean-intensity profile -- plus an explicit note that channel ROLES
    are not, and cannot be, inferred here.
    """
    path = Path(path)
    factory = reader_factory if reader_factory is not None else _default_nd2_reader_factory

    with factory(path) as reader:
        sizes = dict(reader.sizes)
        raw = np.asarray(reader.asarray())
        voxel = reader.voxel_size()
        metadata = getattr(reader, "metadata", None)

    lines: list[str] = []
    lines.append(f"nd2 inspection: {path}")
    lines.append("")
    lines.append(f"axes (as reported by the file): {''.join(sizes.keys())}")
    lines.append(f"sizes: {sizes}")
    lines.append(f"dtype: {raw.dtype}")
    lines.append("")

    x = getattr(voxel, "x", None)
    y = getattr(voxel, "y", None)
    z = getattr(voxel, "z", None)
    if x is not None and y is not None and z is not None:
        lines.append(f"voxel spacing (µm): x={x:.6g}  y={y:.6g}  z={z:.6g}")
        if x > 0 and y > 0 and z > 0:
            geometry = VoxelGeometry(spacing_um_zyx=(float(z), float(y), float(x)))
            lines.append(
                "Z/XY anisotropy ratio (z / min(x,y)): "
                f"{geometry.anisotropy_z_to_xy:.6g}"
                "  -- use this directly as cellpose's `anisotropy` argument"
            )
        else:
            lines.append(
                "Z/XY anisotropy ratio: NOT COMPUTABLE -- spacing is not "
                "strictly positive on every axis"
            )
    else:
        missing = [name for name, v in (("x", x), ("y", y), ("z", z)) if v is None]
        lines.append(f"voxel spacing (µm): UNKNOWN -- file did not report {missing}")
        lines.append("Z/XY anisotropy ratio: NOT COMPUTABLE")
    lines.append("")

    channels_meta = _channel_meta_list(metadata)
    n_channels = sizes.get("C", 1)
    lines.append(f"channels ({n_channels}):")
    for i in range(n_channels):
        meta = channels_meta[i] if i < len(channels_meta) else {}
        name = meta.get("name")
        exc_nm = meta.get("excitation_nm")
        em_nm = meta.get("emission_nm")
        lines.append(
            f"  [{i}] name={name if name is not None else 'UNKNOWN'}"
            f"  excitation_nm={exc_nm if exc_nm is not None else 'UNKNOWN'}"
            f"  emission_nm={em_nm if em_nm is not None else 'UNKNOWN'}"
        )
    lines.append("")

    obj_mag = next(
        (m.get("objective_magnification") for m in channels_meta if m.get("objective_magnification") is not None),
        None,
    )
    obj_na = next(
        (m.get("objective_na") for m in channels_meta if m.get("objective_na") is not None),
        None,
    )
    lines.append(
        f"objective: magnification={obj_mag if obj_mag is not None else 'UNKNOWN'}"
        f"  NA={obj_na if obj_na is not None else 'UNKNOWN'}"
    )
    lines.append("")

    try:
        field0 = _select_first_field_czyx(raw, sizes)
        n_c = field0.shape[0]
        note = " (first field/timepoint only)" if any(k not in "CZYX" for k in sizes) else ""
        lines.append(f"per-channel intensity range{note}:")
        for c in range(n_c):
            sub = field0[c]
            lines.append(
                f"  [{c}] min={float(sub.min()):.6g}  max={float(sub.max()):.6g}"
                f"  mean={float(sub.mean()):.6g}"
            )
        lines.append("")

        n_z = field0.shape[1]
        per_z = field0.reshape(n_c, n_z, -1).astype(np.float64).mean(axis=(0, 2))
        lines.append(f"per-Z mean intensity profile{note} (averaged over channels):")
        lines.append("  " + ", ".join(f"z{z}={v:.6g}" for z, v in enumerate(per_z)))
    except Exception as exc:  # diagnostic-only section; never abort the report over it
        lines.append(f"per-channel / per-Z intensity profiling FAILED: {exc}")
    lines.append("")

    lines.append(
        "NOTE: channel ROLES (nucleus / membrane / signal / organelle_marker / ...) "
        "are configuration, never inferred from channel index, colour, or "
        "brightness -- this report cannot and does not guess them. Confirm "
        "each channel's role against the experimental record (acquisition "
        "log, dye/antibody list) before writing a PipelineConfig; see "
        "configs/example.yaml. Every nucleus-, organelle- and "
        "localization-derived measurement downstream is meaningless if a "
        "role is wrong."
    )
    return "\n".join(lines)


def _cmd_inspect(args: argparse.Namespace) -> int:
    path = Path(args.nd2_path)

    # Checked here rather than left to the reader: nd2 raises FileNotFoundError
    # from several frames deep inside its own reader, and a mistyped path is the
    # most likely first thing anyone hits.
    if not path.exists():
        print(f"error: no such file: {path}", file=sys.stderr)
        return 2
    if path.is_dir():
        print(
            f"error: {path} is a directory. inspect takes a single .nd2 file.",
            file=sys.stderr,
        )
        return 2
    if path.suffix.lower() != ".nd2":
        print(
            f"error: {path} does not look like an .nd2 file (suffix {path.suffix!r}).",
            file=sys.stderr,
        )
        return 2

    try:
        report = build_inspection_report(path)
    except ImportError:
        print(
            "error: reading .nd2 files needs the optional 'nd2' dependency.\n"
            "       install it with:  uv pip install 's2-adhesion[nd2]'",
            file=sys.stderr,
        )
        return 3
    except Exception as exc:  # noqa: BLE001 - surface the library's reason, not a traceback
        print(f"error: could not read {path}: {exc}", file=sys.stderr)
        return 3

    print(report)
    return 0


# ─── extract ────────────────────────────────────────────────────────────────


def _cmd_extract(args: argparse.Namespace) -> int:
    from .config import load_config

    config = load_config(args.config)
    extract = _load_command("s2_adhesion.commands.extract", "extract")
    # commands.extract.extract(nd2_path, output_dir, config, *, source=None,
    # writer=None) -> list[Path], one artifact per field.
    written = extract(
        nd2_path=args.nd2_path, output_dir=args.output_dir, config=config,
        max_fields=args.max_fields,
    )
    for p in written:
        print(p)
    return 0


# ─── segment ────────────────────────────────────────────────────────────────


def _cmd_segment(args: argparse.Namespace) -> int:
    from .config import load_config

    config = load_config(args.config)
    segment = _load_command("s2_adhesion.commands.segment", "segment")
    # commands.segment.segment(image_artifact_dir, output_dir, config) -> Path
    # -- ONE label artifact path, not a list (unlike extract/measure, segment
    # operates on a single already-extracted image artifact per invocation).
    written = segment(
        image_artifact_dir=args.image_dir, output_dir=args.output_dir, config=config
    )
    print(written)
    return 0


# ─── measure ────────────────────────────────────────────────────────────────


def _cmd_measure(args: argparse.Namespace) -> int:
    from .config import load_config

    config = load_config(args.config)
    measure = _load_command("s2_adhesion.commands.measure", "measure")
    # commands.measure.measure(label_artifact_dir, output_dir, config,
    # image_artifact_dir=None) -> RunArtifacts. Label dir is the required
    # input (a geometry-only run has no image at all), which is why it comes
    # BEFORE output_dir -- genuinely easy to get backwards, so every argument
    # here is passed by keyword on purpose; do not switch these back to
    # positional.
    measure(
        label_artifact_dir=args.label_dir,
        output_dir=args.output_dir,
        config=config,
        image_artifact_dir=args.image_dir,
    )
    print(f"measure complete: {args.output_dir}")
    return 0


# ─── run ────────────────────────────────────────────────────────────────────


def _cmd_review(args: argparse.Namespace) -> int:
    from .commands.review import review_run

    written = review_run(args.run_dir, verify_hashes=not args.no_verify)
    if not written:
        print(f"no field artifacts found under {args.run_dir}", file=sys.stderr)
        return 1
    for pth in written:
        print(pth)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from . import pipeline
    from .config import load_config
    from .io import run_manifest
    from .io.nd2_source import ND2Source

    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    run_id = uuid.uuid4().hex
    started_utc = run_manifest.now_utc_iso()
    manifest_path = output_dir / "run_manifest.json"
    source = ND2Source(
        args.nd2_path, config, dataset_id=args.dataset_id,
        max_fields=args.max_fields,
    )

    try:
        artifacts = pipeline.run_pipeline(
            config, source, output_dir, run_id=run_id
        )
    except Exception as exc:
        ended_utc = run_manifest.now_utc_iso()
        manifest = run_manifest.build_run_manifest(
            run_id=run_id,
            config=config,
            input_paths=[args.nd2_path],
            output_paths=[output_dir],
            started_utc=started_utc,
            ended_utc=ended_utc,
            status="failed",
            warnings=(),
            error=str(exc),
        )
        run_manifest.write_run_manifest(manifest, manifest_path)
        raise

    ended_utc = run_manifest.now_utc_iso()
    manifest = run_manifest.build_run_manifest(
        run_id=run_id,
        config=config,
        input_paths=[args.nd2_path],
        output_paths=[output_dir, *artifacts.output_paths],
        started_utc=started_utc,
        ended_utc=ended_utc,
        status="complete",
        warnings=artifacts.warnings,
        error=None,
    )
    run_manifest.write_run_manifest(manifest, manifest_path)
    print(f"run {run_id} complete -> {output_dir}")
    return 0


# ─── argument parser / dispatch ────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="s2-adhesion",
        description="3D instance-based quantification of S2 cell adhesion.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_inspect = sub.add_parser(
        "inspect",
        help="Inspect an nd2 file: axes, spacing, anisotropy, channel "
        "metadata, intensity ranges. No config needed -- and no channel "
        "roles are inferred.",
    )
    p_inspect.add_argument("nd2_path", type=Path, help="Path to a .nd2 file")
    p_inspect.set_defaults(handler=_cmd_inspect)

    p_extract = sub.add_parser(
        "extract", help="Extract every field of an nd2 file to image artifacts."
    )
    p_extract.add_argument("nd2_path", type=Path)
    p_extract.add_argument("output_dir", type=Path)
    p_extract.add_argument("--config", type=Path, required=True)
    p_extract.add_argument(
        "--max-fields", type=int, default=None,
        help="Only extract the first N fields.",
    )
    p_extract.set_defaults(handler=_cmd_extract)

    p_segment = sub.add_parser(
        "segment", help="Segment image artifacts into label artifacts."
    )
    p_segment.add_argument("image_dir", type=Path)
    p_segment.add_argument("output_dir", type=Path)
    p_segment.add_argument("--config", type=Path, required=True)
    p_segment.set_defaults(handler=_cmd_segment)

    p_measure = sub.add_parser(
        "measure",
        help="Measure a label artifact (+ optional image artifact) into "
        "objects/contacts/aggregates/localization CSVs. Never needs torch "
        "or cellpose.",
    )
    p_measure.add_argument("label_dir", type=Path, help="Label artifact directory (required)")
    p_measure.add_argument("output_dir", type=Path)
    p_measure.add_argument("--config", type=Path, required=True)
    p_measure.add_argument(
        "--image-dir",
        dest="image_dir",
        type=Path,
        default=None,
        help="Optional image artifact directory; omitted, this is a "
        "geometry-only run (contact/aggregate metrics still computed; "
        "intensity/nuclei/localization fields recorded missing).",
    )
    p_measure.set_defaults(handler=_cmd_measure)

    p_run = sub.add_parser(
        "run",
        help="Run the full configured pipeline (extract+segment+measure, or "
        "the legacy 2D pipeline) over one nd2 file end to end.",
    )
    p_run.add_argument("nd2_path", type=Path)
    p_run.add_argument("output_dir", type=Path)
    p_run.add_argument("--config", type=Path, required=True)
    p_run.add_argument("--dataset-id", type=str, default=None)
    p_run.add_argument(
        "--max-fields", type=int, default=None,
        help="Only process the first N fields (try the pipeline on a couple first).",
    )
    p_run.set_defaults(handler=_cmd_run)

    p_review = sub.add_parser(
        "review",
        help="Render per-field comparison images (original + segmentation outline) "
             "from a completed run directory.",
    )
    p_review.add_argument("run_dir", type=Path, help="A pipeline run output directory.")
    p_review.add_argument(
        "--no-verify", action="store_true",
        help="Skip artifact content-hash verification when reading.",
    )
    p_review.set_defaults(handler=_cmd_review)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    # Reconfigures process-global stdout/stderr -- must happen only at this
    # real CLI entry point, never at module import time (a library consumer
    # who merely imports s2_adhesion must not have this done to them behind
    # their back). Every unit this project prints is in micrometres ('µ',
    # U+00B5), which a cp932 console (default on Japanese Windows) cannot
    # encode without this -- see _console.py's docstring.
    ensure_utf8_output()
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except S2AdhesionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
