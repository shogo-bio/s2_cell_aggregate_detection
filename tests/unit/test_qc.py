"""Tests for s2_adhesion.metrics.qc against synthetic label volumes.

All fixtures come from tests/synthetic_volumes.py; ground truth here is
either exact voxel bookkeeping (border/component checks) or hand-built
arrays where the right answer is obvious by construction.
"""

from __future__ import annotations

import numpy as np

from s2_adhesion.metrics import qc
from tests.synthetic_volumes import cell_chain, clipped_sphere, sphere

SPACING = (0.1, 0.1, 0.1)
SHAPE = (40, 40, 40)


# ─── voxel_count_observed ───────────────────────────────────────────────────


def test_voxel_count_observed_matches_exact_counts():
    labels = np.zeros((10, 10, 10), dtype=np.uint32)
    labels[0:2, 0:2, 0:2] = 1  # 8 voxels
    labels[5:6, 5:6, 5:9] = 2  # 4 voxels
    counts = qc.voxel_count_observed(labels)
    assert counts == {1: 8, 2: 4}


def test_voxel_count_observed_ignores_background():
    labels = np.zeros((5, 5, 5), dtype=np.uint32)
    counts = qc.voxel_count_observed(labels)
    assert counts == {}


# ─── touches_xy_border / touches_z_border: all six faces ──────────────────


def test_interior_sphere_touches_no_border():
    lab = sphere((2.0, 2.0, 2.0), 1.5, SHAPE, SPACING)
    assert qc.touches_xy_border(lab) == {1: False}
    assert qc.touches_z_border(lab) == {1: False}


def test_all_six_faces_are_individually_detected():
    radius_um = 1.5
    for face in ("z_min", "z_max", "y_min", "y_max", "x_min", "x_max"):
        lab = clipped_sphere(face, radius_um, SHAPE, SPACING)
        xy = qc.touches_xy_border(lab)
        z = qc.touches_z_border(lab)
        axis = face.split("_")[0]
        if axis == "z":
            assert z == {1: True}, face
            assert xy == {1: False}, face
        else:
            assert xy == {1: True}, face
            assert z == {1: False}, face


def test_touches_xy_border_true_for_all_four_xy_edges():
    # y_min
    lab = np.zeros(SHAPE, dtype=np.uint32)
    lab[5:8, 0:3, 5:8] = 1
    assert qc.touches_xy_border(lab) == {1: True}
    # y_max
    lab = np.zeros(SHAPE, dtype=np.uint32)
    lab[5:8, -3:, 5:8] = 1
    assert qc.touches_xy_border(lab) == {1: True}
    # x_min
    lab = np.zeros(SHAPE, dtype=np.uint32)
    lab[5:8, 5:8, 0:3] = 1
    assert qc.touches_xy_border(lab) == {1: True}
    # x_max
    lab = np.zeros(SHAPE, dtype=np.uint32)
    lab[5:8, 5:8, -3:] = 1
    assert qc.touches_xy_border(lab) == {1: True}


def test_touches_z_border_true_for_both_z_edges():
    lab = np.zeros(SHAPE, dtype=np.uint32)
    lab[0:3, 5:8, 5:8] = 1
    assert qc.touches_z_border(lab) == {1: True}
    assert qc.touches_xy_border(lab) == {1: False}

    lab = np.zeros(SHAPE, dtype=np.uint32)
    lab[-3:, 5:8, 5:8] = 1
    assert qc.touches_z_border(lab) == {1: True}
    assert qc.touches_xy_border(lab) == {1: False}


def test_border_touch_is_per_label_not_global():
    """One label touching a border must not mark an untouched label."""
    lab = np.zeros(SHAPE, dtype=np.uint32)
    lab[5:8, 0:3, 5:8] = 1  # touches y_min
    lab[20:23, 20:23, 20:23] = 2  # interior
    xy = qc.touches_xy_border(lab)
    assert xy == {1: True, 2: False}


# ─── connected_component_count: 26-connectivity ────────────────────────────


def test_single_blob_has_one_component():
    lab = sphere((2.0, 2.0, 2.0), 1.5, SHAPE, SPACING)
    assert qc.connected_component_count(lab) == {1: 1}


def test_reused_label_two_disjoint_blobs_reports_two_components():
    """A deliberately reused label id (segmentation bug) reports count 2."""
    lab = np.zeros((20, 20, 20), dtype=np.uint32)
    lab[2:4, 2:4, 2:4] = 1
    lab[14:16, 14:16, 14:16] = 1  # far away, same id
    assert qc.connected_component_count(lab) == {1: 2}
    assert qc.valid_for_geometry(lab) == {1: False}
    assert qc.qc_code(lab) == {1: "disconnected"}


def test_corner_touching_voxels_count_as_one_26_connected_component():
    """Two voxels touching only at a shared corner are connected under
    26-connectivity but NOT under 6- or 18-connectivity -- this pins down
    which structuring element is in use. (1,1,1) and (2,2,2) differ by 1 on
    every axis (Chebyshev distance 1), so they share only a single vertex."""
    lab = np.zeros((5, 5, 5), dtype=np.uint32)
    lab[1, 1, 1] = 1
    lab[2, 2, 2] = 1
    assert qc.connected_component_count(lab) == {1: 1}


def test_face_touching_voxels_are_one_component_regardless_of_connectivity():
    lab = np.zeros((5, 5, 5), dtype=np.uint32)
    lab[1, 1, 1] = 1
    lab[1, 1, 2] = 1
    assert qc.connected_component_count(lab) == {1: 1}


def test_chain_members_are_each_a_single_component():
    """Every ball in a touching chain must itself stay one connected piece."""
    lab = cell_chain(4, (4.0, 3.0, 3.0), 1.5, 2.0, (80, 60, 120), SPACING, axis="x")
    cc = qc.connected_component_count(lab)
    assert cc == {1: 1, 2: 1, 3: 1, 4: 1}


# ─── valid_for_geometry ─────────────────────────────────────────────────────


def test_valid_for_geometry_true_only_when_untruncated_and_connected():
    lab = sphere((2.0, 2.0, 2.0), 1.5, SHAPE, SPACING)
    assert qc.valid_for_geometry(lab) == {1: True}


def test_valid_for_geometry_false_for_each_clipped_face():
    for face in ("z_min", "z_max", "y_min", "y_max", "x_min", "x_max"):
        lab = clipped_sphere(face, 1.5, SHAPE, SPACING)
        assert qc.valid_for_geometry(lab) == {1: False}, face


# ─── qc_code: stable vocabulary, never dropped ─────────────────────────────


def test_qc_code_empty_string_for_a_clean_object():
    lab = sphere((2.0, 2.0, 2.0), 1.5, SHAPE, SPACING)
    assert qc.qc_code(lab) == {1: ""}


def test_qc_code_truncated_xy():
    lab = clipped_sphere("y_min", 1.5, SHAPE, SPACING)
    assert qc.qc_code(lab) == {1: "truncated_xy"}


def test_qc_code_truncated_z():
    lab = clipped_sphere("z_min", 1.5, SHAPE, SPACING)
    assert qc.qc_code(lab) == {1: "truncated_z"}


def test_qc_code_combines_multiple_failures_in_stable_order():
    """A label touching both an xy face, a z face, AND reused elsewhere must
    report all three codes, always in the same order."""
    lab = np.zeros((20, 20, 20), dtype=np.uint32)
    lab[0:3, 0:3, 5:8] = 1  # touches z_min and y_min (xy border)
    lab[14:16, 14:16, 14:16] = 1  # disconnected duplicate, interior
    assert qc.qc_code(lab) == {1: "truncated_xy;truncated_z;disconnected"}
    assert qc.valid_for_geometry(lab) == {1: False}


def test_truncated_objects_are_kept_not_dropped():
    """QC never removes a row: a truncated object still appears in every dict."""
    lab = clipped_sphere("x_max", 1.5, SHAPE, SPACING)
    assert 1 in qc.voxel_count_observed(lab)
    assert 1 in qc.touches_xy_border(lab)
    assert 1 in qc.touches_z_border(lab)
    assert 1 in qc.connected_component_count(lab)
    assert 1 in qc.valid_for_geometry(lab)
    assert 1 in qc.qc_code(lab)
    assert qc.voxel_count_observed(lab)[1] > 0
