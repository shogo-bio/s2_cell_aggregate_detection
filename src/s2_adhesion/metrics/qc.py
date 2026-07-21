"""Per-label QC on instance label volumes.

Pure numpy/scipy -- no ML imports, so this module works with no ML package
installed at all. Every function takes a raw ``ZYX`` ``uint32`` label array
(background 0, as frozen in ``contracts.LabelVolume``) and returns a dict
keyed by label id. Truncated or disconnected objects are always KEPT in
these dicts -- QC records the problem, it never drops the row.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import generate_binary_structure, label as ndi_label

# 26-connectivity in 3D: every voxel within a Chebyshev distance of 1,
# including face, edge and corner neighbours.
_STRUCTURE_26 = generate_binary_structure(3, 3)


def _label_ids(labels: NDArray[np.uint32]) -> NDArray[np.uint32]:
    ids = np.unique(labels)
    return ids[ids > 0]


def voxel_count_observed(labels: NDArray[np.uint32]) -> dict[int, int]:
    """Raw voxel count per label id, with no validity filtering."""
    ids, counts = np.unique(labels, return_counts=True)
    return {int(i): int(c) for i, c in zip(ids, counts) if i != 0}


def touches_xy_border(labels: NDArray[np.uint32]) -> dict[int, bool]:
    """True if the label has any voxel at x in {0, X-1} or y in {0, Y-1}."""
    ids = _label_ids(labels)
    border = np.zeros(labels.shape, dtype=bool)
    border[:, 0, :] = True
    border[:, -1, :] = True
    border[:, :, 0] = True
    border[:, :, -1] = True
    touching = set(np.unique(labels[border]).tolist()) - {0}
    return {int(i): int(i) in touching for i in ids}


def touches_z_border(labels: NDArray[np.uint32]) -> dict[int, bool]:
    """True if the label has any voxel at z in {0, Z-1}."""
    ids = _label_ids(labels)
    border = np.zeros(labels.shape, dtype=bool)
    border[0, :, :] = True
    border[-1, :, :] = True
    touching = set(np.unique(labels[border]).tolist()) - {0}
    return {int(i): int(i) in touching for i in ids}


def connected_component_count(labels: NDArray[np.uint32]) -> dict[int, int]:
    """Number of 26-connected components sharing each label id.

    A well-formed instance is exactly 1; >1 means the same id was reused for
    two physically separate objects (a segmentation bug this QC exists to
    catch), and 0 never occurs since ids come from voxels present in labels.
    """
    ids = _label_ids(labels)
    out: dict[int, int] = {}
    for i in ids:
        mask = labels == i
        _, n = ndi_label(mask, structure=_STRUCTURE_26)
        out[int(i)] = int(n)
    return out


def valid_for_geometry(labels: NDArray[np.uint32]) -> dict[int, bool]:
    """True iff the object is untruncated AND single-component.

    Downstream geometry must null every canonical field when this is False,
    while still keeping observed_* quantities.
    """
    xy = touches_xy_border(labels)
    z = touches_z_border(labels)
    cc = connected_component_count(labels)
    return {i: (not xy[i] and not z[i] and cc[i] == 1) for i in xy}


def qc_code(labels: NDArray[np.uint32]) -> dict[int, str]:
    """Stable semicolon-joined QC codes, e.g. 'truncated_z;disconnected'.

    Never free text: codes are drawn from a fixed vocabulary
    ({'truncated_xy', 'truncated_z', 'disconnected'}) in a fixed order, so the
    same failure combination always serialises to the same string. Empty
    string means the object passed every check.
    """
    xy = touches_xy_border(labels)
    z = touches_z_border(labels)
    cc = connected_component_count(labels)
    out: dict[int, str] = {}
    for i in xy:
        codes: list[str] = []
        if xy[i]:
            codes.append("truncated_xy")
        if z[i]:
            codes.append("truncated_z")
        if cc[i] != 1:
            codes.append("disconnected")
        out[i] = ";".join(codes)
    return out
