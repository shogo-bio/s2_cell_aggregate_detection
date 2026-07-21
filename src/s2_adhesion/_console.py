"""Console output encoding guard.

Every user-facing string in this project quotes physical units, and those units
contain 'µ' (U+00B5). On a Japanese Windows install the console default code page
is cp932, which cannot encode it, so *any* printed micrometre value raises

    UnicodeEncodeError: 'cp932' codec can't encode character '\\xb5'

This is not hypothetical: the original detect_aggregations.py crashes on
``--help`` alone on such a machine, because argparse writes the help text
containing "µm" straight to stdout. The same applies to every progress line that
reports a spacing or a threshold.

Call :func:`ensure_utf8_output` at each CLI entry point -- never at import time.
Reconfiguring streams is a process-global side effect, so a library consumer who
merely imports this package must not have it done to them behind their back.
"""

from __future__ import annotations

import sys
from typing import IO, Any


def _reconfigure(stream: IO[Any] | None) -> None:
    # Only real text streams expose reconfigure(); pytest's capture objects and
    # anything already replaced by the caller may not, and must be left alone.
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is None:
        return
    try:
        reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, OSError):
        # A detached or non-seekable stream cannot be reconfigured. Degrading to
        # the platform default is better than refusing to run at all.
        pass


def ensure_utf8_output() -> None:
    """Make stdout and stderr able to carry 'µ' and other non-ASCII output.

    Safe to call more than once, and a no-op where the streams already handle
    UTF-8 (Linux, macOS, and Windows terminals with a UTF-8 code page).

    ``errors="replace"`` rather than ``"strict"`` is deliberate: if some stream
    still cannot represent a character, a mangled unit symbol is an acceptable
    outcome, whereas a crash part-way through a long analysis run is not.
    """
    _reconfigure(sys.stdout)
    _reconfigure(sys.stderr)
