"""Compatibility entry point for the original 2D threshold pipeline.

Preserves the original ``detect_aggregations.py`` CLI byte-for-byte: same
positional arguments, same flags, same defaults, same CSV columns
(``field_id``, ``aggregation_id``, ``area_px``, ``area_um2``,
``active_channels``). This exists as an importable command -- not just the
standalone root script -- so a future ``s2_adhesion.cli`` can register it as
a subcommand without re-implementing argument parsing, and so anyone running
the pipeline mid-study can keep getting the exact old behaviour through the
installed package.

Nothing here touches pixels; all computation is delegated to
``s2_adhesion.legacy.algorithm``, which is itself extracted verbatim (in
behaviour) from the original script. ``build_parser`` here IS
``s2_adhesion.legacy.algorithm.build_parser`` -- re-exported, not
reimplemented -- so every original flag and default is guaranteed identical
by construction rather than by keeping two argument lists in sync by hand.
"""

from __future__ import annotations

from pathlib import Path

from ..legacy.algorithm import Config, build_parser, process_nd2

__all__ = ["build_parser", "run_legacy", "main"]

_DEFAULTS = Config()


def run_legacy(
    nd2_path: Path,
    output_dir: Path,
    *,
    s2_diameter_um: float = _DEFAULTS.s2_diameter_um,
    min_cells: int = _DEFAULTS.aggregation_min_cells,
    morph_close_radius_um: float = _DEFAULTS.morph_close_radius_um,
    binary_threshold: int = _DEFAULTS.binary_threshold,
    min_active_channels: int = _DEFAULTS.min_active_channels,
    debug: bool = _DEFAULTS.save_debug,
) -> None:
    """Run the original pipeline on one nd2 file.

    Mirrors ``detect_aggregations.main()`` exactly (same ``Config``
    construction, same call into ``process_nd2``), just callable directly
    without going through ``argparse`` -- e.g. for a future ``s2_adhesion.cli``
    subcommand that already has its own parsed arguments.
    """
    nd2_path = Path(nd2_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config(
        s2_diameter_um=s2_diameter_um,
        aggregation_min_cells=min_cells,
        morph_close_radius_um=morph_close_radius_um,
        binary_threshold=binary_threshold,
        min_active_channels=min_active_channels,
        save_debug=debug,
    )
    process_nd2(nd2_path, output_dir, cfg)


def main(argv: list[str] | None = None) -> None:
    # Console encoding guard, called here (a CLI entry point) and nowhere at
    # import time -- see s2_adhesion._console. `run_legacy()` itself is a
    # plain library function and stays free of this side effect; only this
    # argparse-driven entry point applies it.
    from .._console import ensure_utf8_output
    ensure_utf8_output()

    args = build_parser().parse_args(argv)
    run_legacy(
        args.nd2_path,
        args.output_dir,
        s2_diameter_um=args.s2_diameter,
        min_cells=args.min_cells,
        morph_close_radius_um=args.morph_close_radius,
        binary_threshold=args.binary_threshold,
        min_active_channels=args.min_active_channels,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
