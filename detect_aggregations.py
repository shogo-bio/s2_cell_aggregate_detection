#!/usr/bin/env python3
"""
S2 cell aggregation detection from nd2 fluorescence images.

Segmentation strategy: cell occupancy estimation (not boundary detection).
Pipeline per field: MIP → normalize → Gaussian blur → Otsu threshold
→ binary_fill_holes → morphology close → connected components → area filter.

Usage:
    python detect_aggregations.py input.nd2 output_dir/ [options]

THIN WRAPPER. All computation now lives in ``s2_adhesion.legacy.algorithm``,
extracted verbatim (in behaviour) from this script's pre-refactor version so
the new 3D pipeline can be compared against this exact algorithm on identical
data (see ``tests/unit/test_legacy_regression.py``). This file's command-line
interface, argument names, defaults and printed output are unchanged -- do
not add logic here; add it to ``s2_adhesion/legacy/algorithm.py`` and this
wrapper picks it up automatically. Every public name from that module is
re-exported here too, for anyone with an existing ``from detect_aggregations
import <name>`` in a notebook or ad-hoc script.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from s2_adhesion.legacy.algorithm import *  # noqa: F401,F403
    from s2_adhesion.legacy.algorithm import __all__, main
except ImportError:
    # Fallback for a checkout that was never `pip install -e .`-ed: make the
    # in-repo package importable without requiring an editable install.
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from s2_adhesion.legacy.algorithm import *  # noqa: F401,F403
    from s2_adhesion.legacy.algorithm import __all__, main


if __name__ == "__main__":
    main()
