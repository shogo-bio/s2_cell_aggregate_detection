"""Per-cell nucleus assignment and nucleus-vs-cytoplasm intensity summaries.

Pure numpy/scipy -- no ML imports, so this module works with no ML package
installed at all (mirrors ``metrics.qc`` and ``metrics.geometry``).

Nuclei are assigned to cells by MAXIMUM OVERLAP, evaluated in both directions
at once:

* a cell may report a ``primary_nucleus_id`` only if exactly one nucleus
  instance touches it (any nonzero voxel overlap) -- a cell touched by two or
  more nucleus instances is ambiguous (a real binucleate cell looks
  identical, in voxel terms, to two adjacent nuclei mis-segmented into one
  cell region), and
* that one touching nucleus must, in turn, have this cell as its STRICT,
  unique maximum-overlap cell among every cell it touches -- a nucleus that
  straddles two cells with tied (or otherwise ambiguous) overlap is not
  arbitrarily handed to one side.

Both conditions fail closed: any ambiguity nulls ``primary_nucleus_id`` and
every nucleus-shape/intensity field for the affected cell(s) rather than
guessing. ``nucleus_count`` still records how many nucleus instances touched
the cell, so "no nucleus present" (0), "one clean nucleus" (1, fields
populated) and "ambiguous" (1 with no assignment, or >=2) are distinguishable
in the output.

``nuclear_enrichment`` is an intensity-ratio readout only. Confocal axial
resolution (~0.5-0.8 um) is far coarser than any subcellular structure this
ratio might be used to reason about, so it must never be read as a claim
about which side of a membrane, or which molecular compartment, a signal
sits in -- only that voxels labelled "nucleus" are, on average, brighter or
dimmer than voxels labelled "rest of the cell".
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from ..contracts import Scalar, VoxelGeometry

_FIELDS = (
    "nucleus_count",
    "primary_nucleus_id",
    "nucleus_volume_um3",
    "nucleus_to_cell_volume_fraction",
    "nucleus_centroid_distance_um",
    "nucleus_offset_normalized",
    "nucleus_signal_fraction",
    "nuclear_enrichment",
)


def _all_none_row() -> dict[str, Scalar]:
    return {k: None for k in _FIELDS}


def _cell_ids(labels: NDArray[np.uint32]) -> list[int]:
    ids = np.unique(labels)
    return sorted(int(i) for i in ids if i != 0)


def _overlap_crosstab(
    cells: NDArray[np.uint32], nuclei: NDArray[np.uint32]
) -> dict[int, dict[int, int]]:
    """``{cell_id: {nucleus_id: overlap_voxel_count}}`` for every touching pair.

    Built from a single flattened pass rather than one mask per label pair,
    so it stays cheap even with many instances.
    """
    both = (cells > 0) & (nuclei > 0)
    if not np.any(both):
        return {}
    c = cells[both].astype(np.int64)
    n = nuclei[both].astype(np.int64)
    pairs, counts = np.unique(np.stack([c, n], axis=1), axis=0, return_counts=True)
    out: dict[int, dict[int, int]] = {}
    for (cid, nid), cnt in zip(pairs.tolist(), counts.tolist()):
        out.setdefault(cid, {})[nid] = cnt
    return out


def _nucleus_best_cells(
    overlap: Mapping[int, Mapping[int, int]],
) -> dict[int, tuple[set[int], int]]:
    """For each nucleus id: ``(set of cells tied for max overlap, that max count)``.

    A singleton set means the nucleus has one strict, unambiguous best cell.
    """
    by_nucleus: dict[int, dict[int, int]] = {}
    for cid, nucleus_counts in overlap.items():
        for nid, cnt in nucleus_counts.items():
            by_nucleus.setdefault(nid, {})[cid] = cnt

    out: dict[int, tuple[set[int], int]] = {}
    for nid, cell_counts in by_nucleus.items():
        best = max(cell_counts.values())
        winners = {cid for cid, cnt in cell_counts.items() if cnt == best}
        out[nid] = (winners, best)
    return out


def compute_nuclei(
    cells: NDArray[np.uint32],
    nuclei: NDArray[np.uint32] | None,
    geometry: VoxelGeometry,
    signal: NDArray[np.floating] | None = None,
    eps: float = 0.0,
) -> dict[int, dict[str, Scalar]]:
    """Per-cell nucleus assignment and summary metrics, keyed by cell id.

    ``cells`` and ``nuclei`` are raw ``ZYX`` ``uint32`` label arrays
    (background 0) of identical shape, matching ``LabelVolume.cells`` /
    ``LabelVolume.nuclei``. ``signal`` is an optional raw ``ZYX`` intensity
    array (any one channel the caller wants summarised against the nucleus);
    when omitted ``nucleus_signal_fraction`` and ``nuclear_enrichment`` are
    ``None`` but the shape-only fields are still populated.

    ``eps`` damps ``nuclear_enrichment`` near zero signal and should be the
    caller's background noise estimate for ``signal`` (e.g. its standard
    deviation), never an arbitrary constant; 0.0 (the default) applies no
    damping and yields ``None`` wherever a mean would otherwise be divided
    by zero.

    If ``nuclei`` is ``None`` (no nucleus channel was segmented for this
    field), every field is ``None`` for every cell -- a nucleus must never be
    inferred from a bright region of some other channel.
    """
    cell_ids = _cell_ids(cells)

    if nuclei is None:
        return {cid: _all_none_row() for cid in cell_ids}

    voxel_volume = geometry.voxel_volume_um3
    overlap = _overlap_crosstab(cells, nuclei)
    nucleus_best = _nucleus_best_cells(overlap)

    out: dict[int, dict[str, Scalar]] = {}
    for cid in cell_ids:
        row = _all_none_row()
        touching = overlap.get(cid, {})
        row["nucleus_count"] = len(touching)
        out[cid] = row

        if len(touching) != 1:
            continue  # 0 -> no nucleus (valid); >=2 -> ambiguous (QC fail)

        (nid,) = touching.keys()
        winners, _best = nucleus_best[nid]
        if winners != {cid}:
            continue  # this nucleus's best cell is tied or is some other cell

        row["primary_nucleus_id"] = nid

        cell_mask = cells == cid
        nucleus_mask = nuclei == nid  # full physical extent of the nucleus
        cell_voxel_count = int(np.count_nonzero(cell_mask))
        nucleus_voxel_count = int(np.count_nonzero(nucleus_mask))

        nucleus_volume_um3 = nucleus_voxel_count * voxel_volume
        cell_volume_um3 = cell_voxel_count * voxel_volume
        row["nucleus_volume_um3"] = nucleus_volume_um3
        row["nucleus_to_cell_volume_fraction"] = (
            nucleus_volume_um3 / cell_volume_um3 if cell_volume_um3 > 0 else None
        )

        cell_coords = np.argwhere(cell_mask)
        nucleus_coords = np.argwhere(nucleus_mask)
        cell_centroid_um = geometry.index_to_um(cell_coords).mean(axis=0)
        nucleus_centroid_um = geometry.index_to_um(nucleus_coords).mean(axis=0)
        offset = nucleus_centroid_um - cell_centroid_um
        distance_um = float(np.linalg.norm(offset))
        row["nucleus_centroid_distance_um"] = distance_um

        # Equivalent-sphere radius of the cell, computed locally rather than
        # importing metrics.geometry -- that module is owned by a concurrent
        # agent and this is a two-line formula, not a shared dependency.
        cell_radius_um = (3.0 * cell_volume_um3 / (4.0 * np.pi)) ** (1.0 / 3.0)
        row["nucleus_offset_normalized"] = (
            distance_um / cell_radius_um if cell_radius_um > 0 else None
        )

        if signal is not None:
            signal_in_cell = signal[cell_mask].astype(np.float64)
            signal_in_nucleus = signal[nucleus_mask].astype(np.float64)
            total_cell_signal = float(signal_in_cell.sum())
            row["nucleus_signal_fraction"] = (
                float(signal_in_nucleus.sum()) / total_cell_signal
                if total_cell_signal != 0
                else None
            )

            cytoplasm_mask = cell_mask & ~nucleus_mask
            if np.any(cytoplasm_mask):
                nucleus_mean = float(signal_in_nucleus.mean()) if signal_in_nucleus.size else 0.0
                cytoplasm_mean = float(signal[cytoplasm_mask].astype(np.float64).mean())
                denom = cytoplasm_mean + eps
                numer = nucleus_mean + eps
                row["nuclear_enrichment"] = (
                    float(np.log2(numer / denom)) if denom > 0 and numer > 0 else None
                )

    return out
