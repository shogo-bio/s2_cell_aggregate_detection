"""Per-cell 3D geometry with anisotropic spacing.

Pure numpy/scipy/scikit-image -- no ML imports. Every physical quantity
respects the ``VoxelGeometry`` convention frozen in ``contracts.py``: voxel
centres sit at ``origin + (index + 0.5) * spacing``. Canonical geometry
fields (everything except ``voxel_count_observed`` and
``volume_um3_observed``) are ``None`` for any cell that
``metrics.qc.valid_for_geometry`` marks invalid (truncated or
disconnected) -- truncation biases volume, surface area and every
hull-derived quantity low, so those numbers must not be reported as if they
were measurements of the whole cell.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import binary_erosion, gaussian_filter, generate_binary_structure, zoom
from scipy.spatial import ConvexHull, QhullError
from skimage.measure import marching_cubes, mesh_surface_area

from ..contracts import Scalar, VoxelGeometry
from . import qc as _qc

# Face-connectivity (6-neighbour) structure used to find the surface shell of
# a label: a voxel is "surface" if erosion removes it, i.e. at least one face
# neighbour is background.
_STRUCTURE_6 = generate_binary_structure(3, 1)

# Padding (in ORIGINAL, pre-resample voxels) around a label's tight bounding
# box before meshing. Must leave enough true background on every axis that
# the isotropic resample + Gaussian smoothing below don't see the crop edge.
_SURFACE_CROP_PAD_VOXELS = 2

# Marching cubes meshed directly on a raw anisotropic binary mask
# over-estimates surface area, and the bias grows with the anisotropy ratio:
# measured on an analytic r=5um sphere, meshing the raw mask directly gives
# ~8.6% error at isotropic 0.1um spacing but ~20.1% at 5x anisotropic
# (0.5, 0.1, 0.1) um spacing -- the same staircase effect documented for
# contact-area estimation on ContactEstimator. Resampling to the finest
# axis's spacing before meshing roughly halves that; a small deterministic
# Gaussian pre-smoothing on the resampled mask removes most of what remains,
# at the cost of a small, deterministic under-estimate of curvature.
# sigma=2.0 isotropic-target voxels was chosen by sweeping sigma against
# analytic spheres r=1.5..8um at both isotropic and 5x-anisotropic spacing.
# With this value, measured on r=5um: aniso error 3.3%, iso error 0.25%,
# aniso-vs-iso agreement 3.6% -- all comfortably inside the 10% / 5% targets.
# The one regime that does not fully clear 5% agreement is small radius at
# high anisotropy (r=1.5um: ~6.8% agreement; r=2um: ~6.2%; r=3um: ~4.8%,
# already inside target) -- too few z-slices at 0.5um spacing to resolve a
# small sphere is a real resolution limit, not something smoothing alone
# fixes. tests/unit/test_geometry.py exercises this at r>=4um, where the
# 5% target is met with margin, and separately documents the r=1.5um
# residual as an expected, bounded limitation.
_SURFACE_SMOOTHING_SIGMA_TARGET_VOXELS = 2.0

# The 8 corners of a voxel relative to its centre, as +/-1 multipliers of
# half the physical spacing on each axis.
_HULL_CORNER_OFFSETS = np.array(
    [[dz, dy, dx] for dz in (-1.0, 1.0) for dy in (-1.0, 1.0) for dx in (-1.0, 1.0)]
)


def equivalent_sphere_diameter_um(volume_um3: float | None) -> float | None:
    """Diameter of the sphere with the same volume: (6V/pi)**(1/3)."""
    if volume_um3 is None:
        return None
    return (6.0 * volume_um3 / math.pi) ** (1.0 / 3.0)


def principal_axis_lengths_um(
    points_um: NDArray[np.float64],
) -> tuple[float, float, float]:
    """2*sqrt(5*eigenvalue) of the covariance of physical points, descending.

    ``points_um`` is an (N, 3) array of physical voxel-centre coordinates.
    Equal-weighted covariance stands in for volume-weighting because every
    voxel here carries the same physical volume. For a uniform-density
    ellipsoid with semi-axis a this recovers 2*a exactly, since the variance
    of mass along that axis is a^2/5.
    """
    cov = np.cov(points_um.T, bias=True)
    eigvals = np.linalg.eigvalsh(cov)  # ascending
    eigvals = np.clip(eigvals, 0.0, None)[::-1]  # descending, no negative noise
    lengths = 2.0 * np.sqrt(5.0 * eigvals)
    return float(lengths[0]), float(lengths[1]), float(lengths[2])


def sphericity(volume_um3: float | None, surface_area_um2: float | None) -> float | None:
    """pi**(1/3) * (6V)**(2/3) / A. 1.0 for a sphere, < 1.0 otherwise."""
    if volume_um3 is None or surface_area_um2 is None or surface_area_um2 <= 0:
        return None
    return (math.pi ** (1.0 / 3.0)) * (6.0 * volume_um3) ** (2.0 / 3.0) / surface_area_um2


def _surface_area_um2(
    submask: NDArray[np.bool_],
    spacing_um_zyx: tuple[float, float, float],
) -> float | None:
    """Marching cubes surface area of one label, resampled to isotropic voxels.

    See ``_SURFACE_SMOOTHING_SIGMA_TARGET_VOXELS`` above for why the raw
    anisotropic mask is not meshed directly.
    """
    padded = np.pad(
        submask, _SURFACE_CROP_PAD_VOXELS, mode="constant", constant_values=False
    ).astype(np.float32)

    target = min(spacing_um_zyx)
    zoom_factors = [s / target for s in spacing_um_zyx]
    if any(abs(f - 1.0) > 1e-9 for f in zoom_factors):
        padded = zoom(padded, zoom_factors, order=1)
    if _SURFACE_SMOOTHING_SIGMA_TARGET_VOXELS > 0:
        padded = gaussian_filter(padded, sigma=_SURFACE_SMOOTHING_SIGMA_TARGET_VOXELS)

    try:
        verts, faces, _normals, _values = marching_cubes(
            padded, level=0.5, spacing=(target, target, target)
        )
    except (RuntimeError, ValueError):
        return None
    return float(mesh_surface_area(verts, faces))


def _convex_hull_volume_um3(
    submask: NDArray[np.bool_],
    origin_index_zyx: NDArray[np.int64],
    geometry: VoxelGeometry,
) -> float | None:
    """Convex hull volume over voxel CORNERS, not centres.

    ``volume_um3`` counts whole voxels, so the hull must enclose the same
    solid those voxels tile -- a hull over voxel centres is systematically
    smaller by about half a voxel on every side, which can push
    ``cell_solidity_3d`` above 1 (geometrically impossible: a solid cannot
    have more volume than its own convex hull). Only surface voxels are
    used: every interior voxel's corners already lie inside the hull the
    surface voxels define, so including them cannot change the hull.
    """
    eroded = binary_erosion(submask, structure=_STRUCTURE_6, border_value=0)
    surface = submask & ~eroded
    local_coords = np.argwhere(surface)
    if local_coords.size == 0:
        return None
    global_coords = local_coords + origin_index_zyx
    centres_um = geometry.index_to_um(global_coords)
    half_spacing = np.asarray(geometry.spacing_um_zyx) / 2.0
    corners = (
        centres_um[:, None, :] + (_HULL_CORNER_OFFSETS * half_spacing)[None, :, :]
    ).reshape(-1, 3)
    try:
        hull = ConvexHull(corners)
    except QhullError:
        return None
    return float(hull.volume)


def compute_geometry(
    labels: NDArray[np.uint32], geometry: VoxelGeometry
) -> dict[int, dict[str, Scalar]]:
    """Per-label geometry dict, keyed by label id.

    Every returned dict has ``voxel_count_observed`` and
    ``volume_um3_observed`` populated unconditionally. Every other field is
    ``None`` when the label fails ``metrics.qc.valid_for_geometry``.
    """
    voxel_counts = _qc.voxel_count_observed(labels)
    valid = _qc.valid_for_geometry(labels)
    voxel_volume = geometry.voxel_volume_um3
    spacing = geometry.spacing_um_zyx

    out: dict[int, dict[str, Scalar]] = {}
    for label_id, voxel_count in voxel_counts.items():
        volume_um3_observed = voxel_count * voxel_volume
        row: dict[str, Scalar] = {
            "voxel_count_observed": voxel_count,
            "volume_um3_observed": volume_um3_observed,
            "volume_um3": None,
            "centroid_z_um": None,
            "centroid_y_um": None,
            "centroid_x_um": None,
            "bbox_extent_z_um": None,
            "bbox_extent_y_um": None,
            "bbox_extent_x_um": None,
            "surface_area_um2": None,
            "equivalent_sphere_diameter_um": None,
            "principal_axis_length_1_um": None,
            "principal_axis_length_2_um": None,
            "principal_axis_length_3_um": None,
            "elongation": None,
            "flatness": None,
            "sphericity": None,
            "cell_convex_hull_volume_um3": None,
            "cell_solidity_3d": None,
        }
        out[label_id] = row

        if not valid[label_id]:
            continue

        coords = np.argwhere(labels == label_id)  # (N, 3) int index (z, y, x)
        points_um = geometry.index_to_um(coords)  # (N, 3) physical (z, y, x)

        volume_um3 = volume_um3_observed
        row["volume_um3"] = volume_um3

        centroid = points_um.mean(axis=0)
        row["centroid_z_um"], row["centroid_y_um"], row["centroid_x_um"] = (
            float(centroid[0]),
            float(centroid[1]),
            float(centroid[2]),
        )

        idx_min = coords.min(axis=0)
        idx_max = coords.max(axis=0)  # inclusive
        extent_voxels = (idx_max - idx_min + 1).astype(np.float64)
        bbox_extent_um = extent_voxels * np.asarray(spacing)
        row["bbox_extent_z_um"], row["bbox_extent_y_um"], row["bbox_extent_x_um"] = (
            float(bbox_extent_um[0]),
            float(bbox_extent_um[1]),
            float(bbox_extent_um[2]),
        )

        # Bounding-box slice of just this label -- avoids scipy.ndimage.find_objects,
        # whose output list is sized by max label id and so would be unsafe for the
        # large/sparse ids contracts.py explicitly allows.
        slc = tuple(slice(int(idx_min[i]), int(idx_max[i]) + 1) for i in range(3))
        submask = labels[slc] == label_id

        surface = _surface_area_um2(submask, spacing)
        row["surface_area_um2"] = surface

        row["equivalent_sphere_diameter_um"] = equivalent_sphere_diameter_um(volume_um3)

        ax1, ax2, ax3 = principal_axis_lengths_um(points_um)
        row["principal_axis_length_1_um"] = ax1
        row["principal_axis_length_2_um"] = ax2
        row["principal_axis_length_3_um"] = ax3
        row["elongation"] = (ax1 / ax2) if ax2 > 0 else None
        row["flatness"] = (ax2 / ax3) if ax3 > 0 else None

        row["sphericity"] = sphericity(volume_um3, surface)

        hull_volume = _convex_hull_volume_um3(submask, idx_min, geometry)
        row["cell_convex_hull_volume_um3"] = hull_volume
        if hull_volume is not None and hull_volume > 0:
            row["cell_solidity_3d"] = volume_um3 / hull_volume

    return out
