"""Aggregate-level 3D geometry and connectivity.

An AGGREGATE is a connected component of the QUALIFYING contact graph: nodes are
cell ids present in a label volume, edges are ``ContactRecord`` rows with
``qualifies_as_contact`` true. An isolated cell (no qualifying contact) is an
aggregate of size one.

The original 2D pipeline could only measure the occupied area of a clump. It
could not count cells, so "how many cells per aggregate" and "how tightly
packed" were unanswerable -- exactly the quantities that characterise
adhesion. This module produces them.

Pure numpy/scipy/scikit-image -- no ML imports, so measurement works with no ML
package installed. This module does not import ``metrics.geometry`` or
``metrics.qc``: per-cell volumes and truncation flags are accepted as plain
arguments so this module has no dependency on those concurrently-developed
modules and stays usable in isolation.

Nulling: if ANY member of an aggregate is truncated (cut off by the field of
view), the aggregate's convex hull has no physical meaning, so
``aggregate_convex_hull_volume_um3`` and ``packing_fraction`` -- the only
hull-derived quantities -- are ``None``. A hull additionally needs at least 4
non-coplanar points; where scipy cannot build one (``QhullError``) the same
two fields are ``None`` and a QC code is recorded. Neither case raises.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import binary_erosion
from scipy.spatial import ConvexHull, QhullError
from skimage.measure import marching_cubes, mesh_surface_area

from ..contracts import AggregateRecord, ContactRecord, Scalar, VoxelGeometry
from ..errors import MeasurementError

# Fixed vocabulary, fixed order -- mirrors metrics.qc.qc_code so the same
# failure combination always serialises to the same string.
_QC_TRUNCATED_MEMBER = "truncated_member"
_QC_HULL_DEGENERATE = "hull_degenerate"


def _all_cell_ids(labels: NDArray[np.uint32]) -> list[int]:
    ids = np.unique(labels)
    return sorted(int(i) for i in ids if i != 0)


def _connected_components(
    all_ids: Sequence[int], edges: Sequence[tuple[int, int]]
) -> list[list[int]]:
    """Union-find over ``all_ids``; every id ends up in exactly one group.

    Grouping is by the ``find`` root only, so the result does not depend on
    dict/set iteration order -- callers still sort each group and the list of
    groups for a fully deterministic result.
    """
    parent: dict[int, int] = {i: i for i in all_ids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            lo, hi = (ra, rb) if ra < rb else (rb, ra)
            parent[hi] = lo

    for a, b in edges:
        union(a, b)

    groups: dict[int, list[int]] = {}
    for i in all_ids:
        groups.setdefault(find(i), []).append(i)
    return sorted((sorted(g) for g in groups.values()), key=lambda g: g[0])


def _equivalent_sphere_diameter_um(volume_um3: float | None) -> float | None:
    if volume_um3 is None:
        return None
    return (6.0 * volume_um3 / math.pi) ** (1.0 / 3.0)


def _principal_axis_lengths_um(
    points_um: NDArray[np.float64],
) -> tuple[float, float, float]:
    """2*sqrt(5*eigenvalue) of the covariance of physical points, descending.

    Equal-weighted covariance stands in for volume-weighting because every
    voxel carries the same physical volume.
    """
    cov = np.cov(points_um.T, bias=True)
    eigvals = np.linalg.eigvalsh(np.atleast_2d(cov))  # ascending
    eigvals = np.clip(eigvals, 0.0, None)[::-1]  # descending, no negative noise
    lengths = 2.0 * np.sqrt(5.0 * eigvals)
    if lengths.shape[0] < 3:
        # Degenerate (fewer than 3 independent directions, e.g. a single voxel
        # sliver): pad with zeros rather than raise.
        lengths = np.pad(lengths, (0, 3 - lengths.shape[0]))
    return float(lengths[0]), float(lengths[1]), float(lengths[2])


def _sphericity(volume_um3: float | None, surface_area_um2: float | None) -> float | None:
    if volume_um3 is None or surface_area_um2 is None or surface_area_um2 <= 0:
        return None
    return (math.pi ** (1.0 / 3.0)) * (6.0 * volume_um3) ** (2.0 / 3.0) / surface_area_um2


def _surface_area_um2_from_mask(
    mask_crop: NDArray[np.bool_], spacing_um_zyx: tuple[float, float, float]
) -> float | None:
    """Marching cubes on a 1-voxel-padded binary mask, already cropped to bbox."""
    padded = np.pad(mask_crop, 1, mode="constant", constant_values=False).astype(np.float32)
    try:
        verts, faces, _normals, _values = marching_cubes(
            padded, level=0.5, spacing=spacing_um_zyx
        )
    except (RuntimeError, ValueError):
        return None
    return float(mesh_surface_area(verts, faces))


def _boundary_points_um(
    mask_crop: NDArray[np.bool_],
    idx_min: NDArray[np.int64],
    geometry: VoxelGeometry,
) -> NDArray[np.float64]:
    """Physical coordinates of the union's boundary voxels, cropped to bbox.

    Boundary = mask minus its 26-connected interior. Any thinning strategy
    that never removes a true extreme point leaves the convex hull unchanged,
    so this is purely an efficiency reduction of the point cloud, not an
    approximation of the hull itself.
    """
    interior = binary_erosion(mask_crop, structure=np.ones((3, 3, 3), dtype=bool))
    boundary = mask_crop & ~interior
    local_coords = np.argwhere(boundary)
    if local_coords.size == 0:
        # Fully eroded (can happen for very thin slivers) -- fall back to the
        # full mask rather than handing ConvexHull an empty point set.
        local_coords = np.argwhere(mask_crop)
    global_coords = local_coords + idx_min
    return geometry.index_to_um(global_coords)


def _convex_hull_volume_um3(points_um: NDArray[np.float64]) -> float | None:
    try:
        hull = ConvexHull(points_um)
    except QhullError:
        return None
    return float(hull.volume)


def compute_aggregates(
    labels: NDArray[np.uint32],
    geometry: VoxelGeometry,
    contacts: Sequence[ContactRecord],
    cell_volumes_um3: Mapping[int, float | None],
    truncated_cell_ids: Collection[int],
    *,
    dataset_id: str,
    field_id: str,
    segmentation_run_id: str,
) -> list[AggregateRecord]:
    """Connected components of the qualifying contact graph, with geometry.

    ``labels`` is a raw ``ZYX`` ``uint32`` label array (background 0), matching
    the convention used by ``metrics.qc`` and ``metrics.geometry``. All cell
    ids come from ``np.unique(labels)`` -- a cell with no qualifying contact
    still yields an aggregate of size one. ``cell_volumes_um3`` and
    ``truncated_cell_ids`` must cover every such id; a missing volume raises
    ``MeasurementError`` rather than silently propagating a null.

    Aggregate ids are assigned 1..K in ascending order of each aggregate's
    smallest member cell id, so two runs over the same inputs produce
    identical ids and member orderings.
    """
    all_ids = _all_cell_ids(labels)
    id_set = set(all_ids)
    truncated_set = set(truncated_cell_ids)

    edges: list[tuple[int, int]] = []
    for c in contacts:
        if not c.qualifies_as_contact:
            continue
        if c.cell_id_a not in id_set or c.cell_id_b not in id_set:
            raise MeasurementError(
                f"contact ({c.cell_id_a}, {c.cell_id_b}) references a cell id "
                "not present in labels"
            )
        edges.append((c.cell_id_a, c.cell_id_b))

    groups = _connected_components(all_ids, edges)

    spacing = geometry.spacing_um_zyx

    records: list[AggregateRecord] = []
    for agg_id, member_ids in enumerate(groups, start=1):
        member_set = set(member_ids)
        n = len(member_ids)

        missing = [i for i in member_ids if i not in cell_volumes_um3]
        if missing:
            raise MeasurementError(
                f"aggregate {agg_id}: no cell_volumes_um3 entry for cell ids {missing}"
            )
        member_volumes = [cell_volumes_um3[i] for i in member_ids]
        volume_um3: float | None
        if any(v is None for v in member_volumes):
            volume_um3 = None
        else:
            volume_um3 = float(sum(member_volumes))  # type: ignore[arg-type]

        contains_truncated = any(i in truncated_set for i in member_ids)

        internal_edges = [
            c
            for c in contacts
            if c.qualifies_as_contact
            and c.cell_id_a in member_set
            and c.cell_id_b in member_set
        ]
        degree = {i: 0 for i in member_ids}
        total_contact_area = 0.0
        for c in internal_edges:
            degree[c.cell_id_a] += 1
            degree[c.cell_id_b] += 1
            total_contact_area += c.contact_area_um2
        edge_count = len(internal_edges)
        mean_coordination = (2.0 * edge_count / n) if n > 0 else 0.0
        max_coordination = max(degree.values()) if degree else 0
        contact_area_per_cell = total_contact_area / n if n > 0 else 0.0

        # Union mask and its bounding box, once, reused for every voxel-based
        # quantity below.
        member_arr = np.asarray(member_ids, dtype=labels.dtype)
        mask_full = np.isin(labels, member_arr)
        coords_full = np.argwhere(mask_full)
        if coords_full.size == 0:
            raise MeasurementError(
                f"aggregate {agg_id}: member ids {member_ids} have no voxels in labels"
            )
        idx_min = coords_full.min(axis=0)
        idx_max = coords_full.max(axis=0)
        slc = tuple(slice(int(idx_min[i]), int(idx_max[i]) + 1) for i in range(3))
        mask_crop = mask_full[slc]

        points_um_full = geometry.index_to_um(coords_full)
        extent_voxels = (idx_max - idx_min + 1).astype(np.float64)
        extent_um = extent_voxels * np.asarray(spacing)

        ax1, ax2, ax3 = _principal_axis_lengths_um(points_um_full)

        surface_area = _surface_area_um2_from_mask(mask_crop, spacing)
        equivalent_diam = _equivalent_sphere_diameter_um(volume_um3)
        sphericity = _sphericity(volume_um3, surface_area)

        qc_codes: list[str] = []
        hull_volume: float | None = None
        packing_fraction: float | None = None
        if contains_truncated:
            qc_codes.append(_QC_TRUNCATED_MEMBER)
        else:
            boundary_pts = _boundary_points_um(mask_crop, idx_min, geometry)
            hull_volume = _convex_hull_volume_um3(boundary_pts)
            if hull_volume is None or hull_volume <= 0:
                qc_codes.append(_QC_HULL_DEGENERATE)
                hull_volume = None
            elif volume_um3 is not None:
                packing_fraction = volume_um3 / hull_volume

        valid_for_geometry = hull_volume is not None

        values: dict[str, Scalar] = {
            "aggregate_cell_count": n,
            "aggregate_volume_um3": volume_um3,
            "aggregate_convex_hull_volume_um3": hull_volume,
            "packing_fraction": packing_fraction,
            "aggregate_total_contact_area_um2": total_contact_area,
            "aggregate_mean_coordination": mean_coordination,
            "aggregate_max_coordination": max_coordination,
            "aggregate_external_surface_area_um2": surface_area,
            "aggregate_equivalent_sphere_diameter_um": equivalent_diam,
            "aggregate_sphericity": sphericity,
            "aggregate_axis_length_1_um": ax1,
            "aggregate_axis_length_2_um": ax2,
            "aggregate_axis_length_3_um": ax3,
            "aggregate_extent_z_um": float(extent_um[0]),
            "aggregate_extent_y_um": float(extent_um[1]),
            "aggregate_extent_x_um": float(extent_um[2]),
            "aggregate_contact_area_per_cell_um2": contact_area_per_cell,
            "aggregate_valid_for_geometry": valid_for_geometry,
            "aggregate_qc_code": ";".join(qc_codes),
        }

        records.append(
            AggregateRecord(
                dataset_id=dataset_id,
                field_id=field_id,
                segmentation_run_id=segmentation_run_id,
                aggregate_id=agg_id,
                member_cell_ids=tuple(sorted(member_ids)),
                contains_truncated_cell=contains_truncated,
                values=values,
            )
        )

    return records
