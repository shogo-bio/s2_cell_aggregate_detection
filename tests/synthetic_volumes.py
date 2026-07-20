"""Analytic fixture generators for geometry/QC tests.

Every generator returns a raw ``uint32`` ``ZYX`` label array (background 0);
callers wrap the array via the ``make_label_volume`` / ``make_image_volume``
helpers in ``conftest.py``. Every generator places objects using PHYSICAL
micrometre coordinates against the same convention frozen in
``s2_adhesion.contracts.VoxelGeometry``: voxel centres sit at
``origin + (index + 0.5) * spacing``, with ``origin`` fixed at ``(0, 0, 0)``
for all generators here.

Each generator is paired with an ``analytic_*`` function giving the
closed-form ground truth it is built to approximate, so tests compare against
math rather than against another numerical computation.
"""

from __future__ import annotations

import math
from typing import Literal, Sequence

import numpy as np
from numpy.typing import NDArray

Vec3: type = tuple  # (z, y, x) physical or index triples throughout this module

Orientation = Literal["axis_aligned", "diagonal_xy", "tilted_3d"]
Face = Literal["z_min", "z_max", "y_min", "y_max", "x_min", "x_max"]
Axis = Literal["z", "y", "x"]

_AXIS_INDEX = {"z": 0, "y": 1, "x": 2}

_ORIENTATION_DIRECTION: dict[Orientation, tuple[float, float, float]] = {
    # axis-aligned: separation purely along x
    "axis_aligned": (0.0, 0.0, 1.0),
    # 45 degrees within the XY plane (no z component)
    "diagonal_xy": (0.0, 1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)),
    # tilted with a component along every axis
    "tilted_3d": (1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0)),
}


def _physical_centres(
    shape: tuple[int, int, int], spacing: tuple[float, float, float]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Physical (z, y, x) coordinates of every voxel centre, as open grids."""
    dz, dy, dx = spacing
    Z, Y, X = shape
    z = (np.arange(Z, dtype=np.float64) + 0.5) * dz
    y = (np.arange(Y, dtype=np.float64) + 0.5) * dy
    x = (np.arange(X, dtype=np.float64) + 0.5) * dx
    zz = z[:, None, None]
    yy = y[None, :, None]
    xx = x[None, None, :]
    return zz, yy, xx


# ─── Sphere ─────────────────────────────────────────────────────────────────


def sphere(
    centre_um: tuple[float, float, float],
    radius_um: float,
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
) -> NDArray[np.uint32]:
    """Single label (1) ball of the given physical radius centred at centre_um."""
    zz, yy, xx = _physical_centres(shape, spacing)
    cz, cy, cx = centre_um
    d2 = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2
    out = np.zeros(shape, dtype=np.uint32)
    out[d2 <= radius_um**2] = 1
    return out


def analytic_sphere_volume(radius_um: float) -> float:
    return (4.0 / 3.0) * math.pi * radius_um**3


def analytic_sphere_surface_area(radius_um: float) -> float:
    return 4.0 * math.pi * radius_um**2


def analytic_sphere_equivalent_diameter_um(radius_um: float) -> float:
    return 2.0 * radius_um


# ─── Nearest-centre partitions (touching cells) ────────────────────────────


def _nearest_centre_partition(
    centres_um: Sequence[tuple[float, float, float]],
    radius_um: float,
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
) -> NDArray[np.uint32]:
    """Voxels within radius_um of >=1 centre, labelled by the NEAREST centre.

    For equal-radius balls this makes the interface between any two
    overlapping balls the exact perpendicular-bisector (radical) plane, which
    is what gives the analytic contact-disc formula its closed form.
    """
    zz, yy, xx = _physical_centres(shape, spacing)
    n = len(centres_um)
    best_d2 = np.full(shape, np.inf, dtype=np.float64)
    best_label = np.zeros(shape, dtype=np.uint32)
    within_any = np.zeros(shape, dtype=bool)
    r2 = radius_um**2
    for i, (cz, cy, cx) in enumerate(centres_um, start=1):
        d2 = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2
        inside = d2 <= r2
        within_any |= inside
        closer = d2 < best_d2
        take = inside & closer
        best_d2 = np.where(take, d2, best_d2)
        best_label = np.where(take, np.uint32(i), best_label)
    out = np.where(within_any, best_label, np.uint32(0)).astype(np.uint32)
    return out


def sphere_pair_bisected(
    centre_um: tuple[float, float, float],
    radius_um: float,
    separation_um: float,
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
    orientation: Orientation = "axis_aligned",
) -> NDArray[np.uint32]:
    """Two equal balls of radius_um, centres separation_um apart, labels {1, 2}.

    Each voxel in the union is owned by its nearest centre, so the interface
    is the flat radical plane rather than a curved spherical-cap surface --
    do NOT rasterise one sphere over another, which gives a different
    (curved) interface and a different analytic answer.
    """
    dz, dy, dx = _ORIENTATION_DIRECTION[orientation]
    cz, cy, cx = centre_um
    half = separation_um / 2.0
    centre_a = (cz - half * dz, cy - half * dy, cx - half * dx)
    centre_b = (cz + half * dz, cy + half * dy, cx + half * dx)
    return _nearest_centre_partition([centre_a, centre_b], radius_um, shape, spacing)


def analytic_contact_disc_area(radius_um: float, separation_um: float) -> float:
    """Area of the flat radical-plane interface between two touching equal balls.

    Zero when the balls do not overlap (separation_um >= 2*radius_um).
    """
    half = separation_um / 2.0
    if half >= radius_um:
        return 0.0
    return math.pi * (radius_um**2 - half**2)


def cell_chain(
    n: int,
    start_um: tuple[float, float, float],
    radius_um: float,
    separation_um: float,
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
    axis: Axis = "x",
) -> NDArray[np.uint32]:
    """n balls in a line along axis, labels 1..n, each touching only its neighbours.

    separation_um must satisfy radius_um <= separation_um < 2*radius_um so that
    adjacent balls overlap (touch) while balls two apart (distance
    2*separation_um) do not.
    """
    if not (radius_um <= separation_um < 2.0 * radius_um):
        raise ValueError(
            "cell_chain requires radius_um <= separation_um < 2*radius_um so that "
            f"neighbours touch but next-neighbours don't; got radius_um={radius_um}, "
            f"separation_um={separation_um}"
        )
    axis_idx = _AXIS_INDEX[axis]
    centres = []
    for i in range(n):
        c = list(start_um)
        c[axis_idx] = start_um[axis_idx] + i * separation_um
        centres.append(tuple(c))
    return _nearest_centre_partition(centres, radius_um, shape, spacing)


# ─── Ellipsoid ──────────────────────────────────────────────────────────────


def ellipsoid(
    centre_um: tuple[float, float, float],
    radii_um_zyx: tuple[float, float, float],
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
) -> NDArray[np.uint32]:
    """Single label (1) ellipsoid with physical semi-axes radii_um_zyx = (rz, ry, rx)."""
    zz, yy, xx = _physical_centres(shape, spacing)
    cz, cy, cx = centre_um
    rz, ry, rx = radii_um_zyx
    s = ((zz - cz) / rz) ** 2 + ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2
    out = np.zeros(shape, dtype=np.uint32)
    out[s <= 1.0] = 1
    return out


def analytic_ellipsoid_volume(radii_um_zyx: tuple[float, float, float]) -> float:
    rz, ry, rx = radii_um_zyx
    return (4.0 / 3.0) * math.pi * rz * ry * rx


def analytic_ellipsoid_principal_axis_lengths_um(
    radii_um_zyx: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Full axis lengths (2*semi-axis), sorted descending.

    For a uniform-density ellipsoid with semi-axis a, the variance of mass
    along that axis is a^2/5, so 2*sqrt(5*variance) recovers 2*a exactly --
    this is the same formula geometry.py applies to voxel-centre covariance.
    """
    return tuple(sorted((2.0 * r for r in radii_um_zyx), reverse=True))  # type: ignore[return-value]


# ─── Cuboid (exact, no discretisation error when grid-aligned) ─────────────


def cuboid(
    origin_um: tuple[float, float, float],
    extent_um: tuple[float, float, float],
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
) -> NDArray[np.uint32]:
    """Single label (1) axis-aligned box [origin_um, origin_um + extent_um).

    Exact to floating point when origin_um and extent_um are integer
    multiples of spacing: every voxel is either fully in or fully out.
    """
    zz, yy, xx = _physical_centres(shape, spacing)
    oz, oy, ox = origin_um
    ez, ey, ex = extent_um
    inside = (
        (zz >= oz) & (zz < oz + ez)
        & (yy >= oy) & (yy < oy + ey)
        & (xx >= ox) & (xx < ox + ex)
    )
    out = np.zeros(shape, dtype=np.uint32)
    out[inside] = 1
    return out


def analytic_cuboid_volume(extent_um: tuple[float, float, float]) -> float:
    ez, ey, ex = extent_um
    return ez * ey * ex


# ─── Deliberately truncated sphere ─────────────────────────────────────────


def clipped_sphere(
    face: Face,
    radius_um: float,
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
    poke_fraction: float = 0.5,
) -> NDArray[np.uint32]:
    """Single label (1) ball centred so it is cut by exactly one chosen face.

    The ball is centred in the middle of the volume along the two axes not
    named by ``face`` (so it is not accidentally clipped elsewhere, provided
    the volume is at least 2*radius_um wide on those axes) and placed
    poke_fraction * radius_um past the named face along the clipped axis.
    """
    axis_name, side = face.rsplit("_", 1)
    axis_idx = _AXIS_INDEX[axis_name]
    extent_um = tuple(shape[i] * spacing[i] for i in range(3))

    centre = [extent_um[i] / 2.0 for i in range(3)]
    depth = radius_um * (1.0 - poke_fraction)
    if side == "min":
        centre[axis_idx] = depth
    elif side == "max":
        centre[axis_idx] = extent_um[axis_idx] - depth
    else:
        raise ValueError(f"face must end in '_min' or '_max', got {face!r}")

    return sphere(tuple(centre), radius_um, shape, spacing)  # type: ignore[arg-type]
