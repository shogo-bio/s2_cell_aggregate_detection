"""The review command's error handling (rendering itself needs real artifacts)."""

from __future__ import annotations

import subprocess
import sys

import pytest

from s2_adhesion.commands.review import review_run


def test_review_run_rejects_a_non_run_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="run directory"):
        review_run(tmp_path)


def test_review_run_returns_empty_when_no_matching_labels(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    # an image artifact with no matching label artifact -> nothing rendered
    (tmp_path / "images" / "field000.image.ome.zarr").mkdir()
    assert review_run(tmp_path) == []


def test_review_module_imports_without_torch_or_cellpose():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, s2_adhesion.commands.review; "
            "leaked=[m for m in sys.modules if m.split('.')[0] in ('torch','cellpose')]; "
            "print(leaked); sys.exit(1 if leaked else 0)",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout
