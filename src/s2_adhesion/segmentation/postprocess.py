"""Shared, backend-agnostic label cleanup.

Pure numpy/scipy -- no skimage, no ML. Every backend (nucleus-seeded watershed,
direct cellpose, anything added later) is expected to route its raw instance
labels through these functions before they become a :class:`LabelVolume`, so
"one label id split across a gap" or "volume threshold measured in voxels
instead of micrometres" cannot silently differ between backends.

Every function that filters or measures size takes ``spacing_um_zyx`` (or
receives voxel counts alongside it) and works in physical units, per
``contracts.py``: distances are micrometres, volumes um^3, no pixel unit
escapes into a physical field. The same config must behave identically at
different Z steps.

Label arrays in and out are always ``uint32`` ZYX, background is 0, per the
``LabelVolume`` contract in ``contracts.py``.
"""

from __future__ import annotations

from scipy import ndimage as ndi

import numpy as np
from numpy.typing import NDArray

LabelArray = NDArray[np.uint32]

# 26-connectivity in 3D: every voxel with a face, edge, or corner neighbour is
# considered connected. This is the connectivity referenced throughout this
# module and by the nuclear watershed pipeline.
_FULL_CONNECTIVITY_RANK = 3


def _nonzero_ids(labels: NDArray[np.integer]) -> list[int]:
    """Sorted positive label ids present, as plain Python ints (deterministic order)."""
    ids = np.unique(labels)
    return sorted(int(i) for i in ids if i > 0)


def split_disconnected_labels(labels: NDArray[np.integer]) -> LabelArray:
    """Give every 26-connected component its own id.

    If one label id occupies two or more spatially disconnected components
    (e.g. a watershed or cellpose artefact merged two unrelated blobs under
    one id, or simply because upstream never guaranteed connectivity), each
    component becomes a separate positive id. Background (0) is untouched.

    Renumbering is deterministic: original ids are visited in ascending
    order, and within an id, connected components are visited in the order
    ``scipy.ndimage.label`` discovers them (raster / lexicographic scan
    order) -- rerunning on the same input always yields the same output.
    """
    labels = np.asarray(labels)
    structure = ndi.generate_binary_structure(_FULL_CONNECTIVITY_RANK, _FULL_CONNECTIVITY_RANK)
    out: LabelArray = np.zeros(labels.shape, dtype=np.uint32)
    next_id = 1
    for original_id in _nonzero_ids(labels):
        mask = labels == original_id
        components, n_components = ndi.label(mask, structure=structure)
        for component_index in range(1, n_components + 1):
            out[components == component_index] = next_id
            next_id += 1
    return out


def fill_internal_holes(labels: NDArray[np.integer]) -> LabelArray:
    """Fill background voxels fully enclosed within a single instance, in 3D.

    Each instance's holes are filled independently via
    ``scipy.ndimage.binary_fill_holes`` on that instance's own boolean mask.
    A voxel is only ever claimed if it is still background (0) in the output
    at the time its owning instance is processed, so one instance's fill can
    never overwrite another instance's voxels -- only genuine background gets
    reassigned. Instances are processed in ascending id order for
    determinism.
    """
    labels = np.asarray(labels)
    out: LabelArray = labels.astype(np.uint32, copy=True)
    for original_id in _nonzero_ids(labels):
        mask = labels == original_id
        filled = ndi.binary_fill_holes(mask)
        newly_filled = filled & ~mask
        claimable = newly_filled & (out == 0)
        out[claimable] = original_id
    return out


def filter_by_physical_volume(
    labels: NDArray[np.integer],
    spacing_um_zyx: tuple[float, float, float],
    min_volume_um3: float,
) -> LabelArray:
    """Drop instances whose physical volume is below ``min_volume_um3``.

    The threshold is evaluated in um^3 (``voxel_count * dz * dy * dx``), never
    in raw voxel counts, so the same ``min_volume_um3`` removes the same
    physical object regardless of Z step or pixel size. Surviving ids and
    their voxels are left exactly as given -- this function only zeroes out
    instances below threshold, it does not renumber.
    """
    dz, dy, dx = spacing_um_zyx
    if dz <= 0 or dy <= 0 or dx <= 0:
        raise ValueError(f"spacing_um_zyx must be strictly positive, got {spacing_um_zyx}")
    voxel_volume_um3 = dz * dy * dx

    labels = np.asarray(labels)
    out: LabelArray = labels.astype(np.uint32, copy=True)
    ids, counts = np.unique(labels[labels > 0], return_counts=True)
    for label_id, voxel_count in zip(ids, counts):
        physical_volume_um3 = float(voxel_count) * voxel_volume_um3
        if physical_volume_um3 < min_volume_um3:
            out[out == label_id] = 0
    return out


def relabel_sequential(labels: NDArray[np.integer]) -> LabelArray:
    """Renumber surviving positive ids to a dense ``1..N`` range.

    Deterministic and reproducible: ids are visited in ascending order of
    their original value and mapped to ``1, 2, 3, ...`` in that same order,
    so relative id ordering is preserved and the mapping never depends on
    array memory layout, hashing, or iteration order. Background (0) is
    always 0. Idempotent: relabelling an already-sequential array returns it
    unchanged.
    """
    labels = np.asarray(labels)
    out: LabelArray = np.zeros(labels.shape, dtype=np.uint32)
    for new_id, old_id in enumerate(_nonzero_ids(labels), start=1):
        out[labels == old_id] = new_id
    return out
