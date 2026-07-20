"""Unit tests for segmentation/postprocess.py -- pure numpy/scipy, no ML."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from s2_adhesion.segmentation import postprocess

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_importing_postprocess_avoids_ml_dependencies() -> None:
    """Importing postprocess must never pull torch or cellpose into sys.modules.

    Run in a fresh subprocess so pollution from other test modules in the same
    pytest session (e.g. a sibling agent's cellpose backend tests) cannot make
    this pass or fail for the wrong reason.
    """
    script = (
        "import sys\n"
        "import s2_adhesion.segmentation.postprocess\n"
        "assert 'torch' not in sys.modules, sorted(sys.modules)\n"
        "assert 'cellpose' not in sys.modules, sorted(sys.modules)\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


# ─── split_disconnected_labels ─────────────────────────────────────────────


def test_split_disconnected_labels_splits_one_id_into_two() -> None:
    labels = np.zeros((5, 10, 10), dtype=np.uint32)
    labels[1:3, 1:3, 1:3] = 1
    labels[1:3, 6:8, 6:8] = 1  # same id, spatially disconnected (26-connectivity)

    original_block_a = labels[1:3, 1:3, 1:3].copy()
    original_block_b = labels[1:3, 6:8, 6:8].copy()

    out = postprocess.split_disconnected_labels(labels)

    assert out.dtype == np.uint32
    nonzero_ids = sorted(int(i) for i in np.unique(out) if i > 0)
    assert len(nonzero_ids) == 2

    id_a = int(out[1, 1, 1])
    id_b = int(out[1, 6, 6])
    assert id_a != id_b
    # Each component keeps exactly its own voxels, nothing lost or merged.
    assert np.array_equal(out[1:3, 1:3, 1:3] == id_a, original_block_a == 1)
    assert np.array_equal(out[1:3, 6:8, 6:8] == id_b, original_block_b == 1)
    assert np.sum(out == id_a) == original_block_a.size
    assert np.sum(out == id_b) == original_block_b.size


def test_split_disconnected_labels_leaves_connected_id_alone() -> None:
    labels = np.zeros((4, 6, 6), dtype=np.uint32)
    labels[1:3, 1:5, 1:5] = 7  # one solid connected block

    out = postprocess.split_disconnected_labels(labels)

    nonzero_ids = np.unique(out[out > 0])
    assert len(nonzero_ids) == 1
    assert np.array_equal(out > 0, labels > 0)


def test_split_disconnected_labels_touching_only_at_a_corner_is_one_component() -> None:
    # 26-connectivity: two voxels sharing only a corner are still connected.
    labels = np.zeros((2, 2, 2), dtype=np.uint32)
    labels[0, 0, 0] = 3
    labels[1, 1, 1] = 3  # opposite corner of the 2x2x2 block -- corner-adjacent

    out = postprocess.split_disconnected_labels(labels)

    assert len(np.unique(out[out > 0])) == 1


# ─── fill_internal_holes ────────────────────────────────────────────────────


def test_fill_internal_holes_fills_enclosed_background_voxel() -> None:
    labels = np.zeros((5, 5, 5), dtype=np.uint32)
    labels[1:4, 1:4, 1:4] = 9  # solid 3x3x3 cube
    labels[2, 2, 2] = 0  # carve out the single centre voxel -> a hole

    out = postprocess.fill_internal_holes(labels)

    assert out[2, 2, 2] == 9
    assert out.dtype == np.uint32
    # Everything else about the cube is unchanged.
    expected = labels.copy()
    expected[2, 2, 2] = 9
    assert np.array_equal(out, expected)


def test_fill_internal_holes_does_not_touch_other_instances_or_exterior_background() -> None:
    labels = np.zeros((5, 5, 5), dtype=np.uint32)
    labels[1:4, 1:4, 1:4] = 1
    labels[2, 2, 2] = 0  # hole inside instance 1
    labels[0, 0, 0] = 2  # a separate, distant instance

    out = postprocess.fill_internal_holes(labels)

    assert out[2, 2, 2] == 1  # hole filled
    assert out[0, 0, 0] == 2  # untouched
    assert out[4, 4, 4] == 0  # exterior background untouched
    assert int(np.sum(out == 2)) == 1


# ─── filter_by_physical_volume ──────────────────────────────────────────────


def test_filter_by_physical_volume_removes_same_object_differently_at_two_spacings() -> None:
    # Identical voxel geometry (a 4x4x4 = 64-voxel cube) at two very different
    # spacings. The threshold must be evaluated in physical um^3, so the same
    # object survives at one spacing and is removed at the other even though
    # its voxel count never changes.
    labels = np.zeros((6, 6, 6), dtype=np.uint32)
    labels[1:5, 1:5, 1:5] = 1
    voxel_count = int(np.sum(labels == 1))
    assert voxel_count == 64

    coarse_spacing = (1.0, 1.0, 1.0)  # 64 voxels * 1 um^3 = 64 um^3
    fine_spacing = (0.25, 0.25, 0.25)  # 64 voxels * 0.015625 um^3 = 1 um^3
    min_volume_um3 = 10.0

    kept = postprocess.filter_by_physical_volume(labels, coarse_spacing, min_volume_um3)
    dropped = postprocess.filter_by_physical_volume(labels, fine_spacing, min_volume_um3)

    assert np.any(kept == 1), "64 um^3 object must survive a 10 um^3 floor"
    assert not np.any(dropped == 1), "1 um^3 object must be removed by a 10 um^3 floor"


def test_filter_by_physical_volume_keeps_ids_it_does_not_remove() -> None:
    labels = np.zeros((4, 4, 4), dtype=np.uint32)
    labels[0, 0, 0] = 1
    labels[1:3, 1:3, 1:3] = 5

    out = postprocess.filter_by_physical_volume(labels, (1.0, 1.0, 1.0), min_volume_um3=2.0)

    assert out[0, 0, 0] == 0  # single voxel -> 1 um^3, below floor
    assert np.array_equal(out == 5, labels == 5)  # 8-voxel block survives, id preserved


def test_filter_by_physical_volume_rejects_non_positive_spacing() -> None:
    labels = np.zeros((2, 2, 2), dtype=np.uint32)
    with pytest.raises(ValueError):
        postprocess.filter_by_physical_volume(labels, (0.0, 1.0, 1.0), min_volume_um3=1.0)


# ─── relabel_sequential ──────────────────────────────────────────────────────


def test_relabel_sequential_is_dense_and_order_preserving() -> None:
    labels = np.array([[0, 5, 5], [9, 9, 9], [0, 0, 5]], dtype=np.uint32)

    out = postprocess.relabel_sequential(labels)

    assert set(np.unique(out).tolist()) == {0, 1, 2}
    assert np.array_equal(out == 0, labels == 0)
    id_for_5 = int(out[0, 1])
    id_for_9 = int(out[1, 0])
    assert {id_for_5, id_for_9} == {1, 2}
    assert id_for_5 < id_for_9  # original ascending order (5 before 9) preserved
    assert np.array_equal(out == id_for_5, labels == 5)
    assert np.array_equal(out == id_for_9, labels == 9)


def test_relabel_sequential_is_idempotent_and_deterministic() -> None:
    rng = np.random.default_rng(0)
    labels = rng.choice([0, 3, 17, 42], size=(4, 5, 5)).astype(np.uint32)

    once = postprocess.relabel_sequential(labels)
    twice = postprocess.relabel_sequential(once)
    again = postprocess.relabel_sequential(labels)

    assert np.array_equal(once, twice)
    assert np.array_equal(once, again)
    assert once.dtype == np.uint32
