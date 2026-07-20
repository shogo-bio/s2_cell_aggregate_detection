"""Cell-cell contact interface area, orientation/reliability metadata, and
per-cell adhesion aggregates -- including coordination number, the project's
primary adhesion endpoint.

Pure numpy/scipy/scikit-image -- no ML imports, so this module works with no
ML package installed at all, matching every other module in ``metrics``.

STATUS (revised after a cross-model design review of the default production
path): contact AREA is a SECONDARY, orientation-conditioned measurement, not
the headline endpoint it was originally framed as. Measured on the default
production path (anisotropic acquisition 0.5/0.1/0.1 um, MARCHING_CUBES,
``resample_isotropic_before_contact=True``) against the analytic bisected
contact disc:

    axis-aligned   45deg-in-XY   tilted-in-3-axes   spread
        6.3%           15.6%          45.9%         39.6pp

Resampling roughly halves the raw anisotropic error but cannot recover
axial information that was never sampled. So COORDINATION NUMBER (does a
qualifying contact exist at all) is now the primary adhesion endpoint --
counting is more robust to the underlying area bias than trusting the area
number itself.

That promotion comes with its own warning, not a free pass: coordination
number is topological, and "topological" does NOT mean "immune to
anisotropy." A single spurious voxel bridge invents a whole graph edge; a
single missing voxel destroys one. The broad axial PSF and coarse Z sampling
make both especially likely along Z. The failure mode changes from a graded
area bias into a discrete edge error -- arguably worse, because a wrong edge
is invisible in the output (there is no small number sitting next to it
that looks suspicious the way a biased area does). Sections 3 below give
coordination number its own stability evidence, per contact and per cell.

Practical consequence for every downstream consumer: both contact area and
coordination number must be reported STRATIFIED by their reliability/
stability flags, never silently filtered to only the "reliable" subset. A
biological condition that changes packing geometry (e.g. more oblique
contacts) would change which contacts get excluded by such a filter, and
that is a selection-bias channel baked directly into the headline metric.
Report the strata; let downstream analysis decide.

Two area estimators are implemented, see ``ContactEstimator`` (frozen in
``contracts.py``) for the full measured error table:

* ``FACE_COUNT`` counts shared voxel faces and weights each by the physical
  area of the face perpendicular to that axis. It is exact -- zero tolerance
  -- for axis-aligned interfaces, but inflates by up to sqrt(3) as the
  interface tilts relative to the grid. That staircase bias is intrinsic to
  digital geometry and is NOT fixed by isotropic resampling. It is retained
  for exact digital unit tests and sensitivity analysis, not as a headline
  metric.

* ``MARCHING_CUBES`` meshes one cell's boundary and keeps the triangles that
  sit against the other cell. Its bias is larger in the axis-aligned case but
  far more nearly constant across orientations, especially once voxels are
  isotropic -- so ``resample_isotropic_before_contact`` defaults to true and
  MARCHING_CUBES is the default estimator (see ``ContactConfig``). Even so,
  see the production-path table above: "more nearly constant" is not "flat."

Known prototype weakness and what changed here: the reference implementation
selects "triangles against the other cell" with a single binary morphological
dilation (1 index-step, 26-connectivity) of the neighbouring cell's mask. That
selection band is PHYSICALLY ANISOTROPIC whenever spacing is anisotropic --
one index step in Z can be 5x the physical distance of one index step in XY --
which inflates the band asymmetrically and biases area high. This module
selects instead by true physical (Euclidean) distance, computed with a
spacing-aware distance transform, using a band width of ``max(spacing)`` (the
coarsest voxel dimension, so the band always reaches at least as far as the
old dilation did along every axis). Measured on this module's own bisected
r=5um / d=8um sphere-pair harness (see ``tests/unit/test_contacts.py``): the
physical-distance band gives lower absolute error than the index-dilation
band at every orientation and spacing tested.

WHAT THIS MODULE DELIBERATELY DOES NOT DO: it does not rescale contact area
by a fitted function of orientation angle to "correct" the bias. A phantom of
three sphere orientations does not span contact size, curvature, subvoxel
phase, PSF shape, noise, or segmentation error -- a correction calibrated on
it would launder unmodelled error into the number rather than remove it, and
would look more precise than the measurement actually is. Emitting the
orientation metadata below and letting a consumer stratify by it is the
honest stopping point. Do not add a bias-correction term here without new,
broader validation data; a categorical reliability flag is as far as three
measured orientations can honestly take you (see ``_contact_area_reliability``).
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from typing import Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage as ndi
from skimage.measure import marching_cubes

from ..config import ContactConfig
from ..contracts import ContactEstimator, ContactRecord, LabelVolume, Scalar

Pair = tuple[int, int]
Spacing = tuple[float, float, float]

_STRUCTURE_26 = ndi.generate_binary_structure(3, 3)


def _ordered(a: int, b: int) -> Pair:
    return (a, b) if a < b else (b, a)


# ─── FACE_COUNT ────────────────────────────────────────────────────────────


def _face_adjacency_areas(
    lab: NDArray[np.uint32], spacing: Spacing
) -> dict[Pair, float]:
    """Exact analytic contact area per touching label pair, by face counting.

    Anisotropic-aware: a face shared along axis ``ax`` is weighted by the
    physical area of the face perpendicular to that axis (dy*dx for a step
    along z, dz*dx along y, dz*dy along x) -- adapted directly from the
    validated prototype, generalised from a single (a, b) pair to every
    touching pair present in the volume in one pass per axis.
    """
    dz, dy, dx = spacing
    face_area = (dy * dx, dz * dx, dz * dy)
    areas: dict[Pair, float] = defaultdict(float)
    for ax in range(3):
        n = lab.shape[ax]
        if n < 2:
            continue
        p = np.take(lab, np.arange(0, n - 1), axis=ax)
        q = np.take(lab, np.arange(1, n), axis=ax)
        mask = (p != q) & (p != 0) & (q != 0)
        if not mask.any():
            continue
        pa = p[mask].astype(np.int64)
        qa = q[mask].astype(np.int64)
        lo = np.minimum(pa, qa)
        hi = np.maximum(pa, qa)
        uniq, counts = np.unique(np.stack([lo, hi], axis=1), axis=0, return_counts=True)
        for (a, b), c in zip(uniq.tolist(), counts.tolist()):
            areas[(a, b)] += c * face_area[ax]
    return areas


def face_count_pair_area_um2(
    lab: NDArray[np.uint32], spacing: Spacing, a: int, b: int
) -> float:
    """Exact analytic FACE_COUNT contact area between exactly two label ids.

    Zero tolerance for axis-aligned interfaces -- exposed standalone (rather
    than only through the multi-pair dict) so unit tests can assert exactness
    directly against ``n_faces * face_area`` for a single known pair.
    """
    return _face_adjacency_areas(lab, spacing).get(_ordered(int(a), int(b)), 0.0)


# ─── MARCHING_CUBES ────────────────────────────────────────────────────────


def _candidate_pairs_26(lab: NDArray[np.uint32]) -> set[Pair]:
    """Every unordered pair of distinct nonzero labels touching under full
    26-connectivity (face, edge, or corner).

    Cast wider than face adjacency deliberately: marching-cubes surfaces are
    triangulated, not axis-aligned, so two cells that only share an edge or
    corner in the digital grid can still yield a (typically tiny) meshed
    interface. Implemented as 13 canonical shift offsets (one per antipodal
    direction pair), each a vectorised comparison -- no per-label loop.
    """
    offsets = [o for o in itertools.product((-1, 0, 1), repeat=3) if o != (0, 0, 0)]
    canonical = [
        o for o in offsets
        if o[0] > 0 or (o[0] == 0 and o[1] > 0) or (o[0] == 0 and o[1] == 0 and o[2] > 0)
    ]

    def sl(o: int, n: int) -> tuple[slice, slice]:
        if o == 1:
            return slice(0, n - 1), slice(1, n)
        if o == -1:
            return slice(1, n), slice(0, n - 1)
        return slice(0, n), slice(0, n)

    pairs: set[Pair] = set()
    for oz, oy, ox in canonical:
        zp, zq = sl(oz, lab.shape[0])
        yp, yq = sl(oy, lab.shape[1])
        xp, xq = sl(ox, lab.shape[2])
        p = lab[zp, yp, xp]
        q = lab[zq, yq, xq]
        mask = (p != q) & (p != 0) & (q != 0)
        if not mask.any():
            continue
        pa = p[mask].astype(np.int64)
        qa = q[mask].astype(np.int64)
        lo = np.minimum(pa, qa)
        hi = np.maximum(pa, qa)
        uniq = np.unique(np.stack([lo, hi], axis=1), axis=0)
        for a, b in uniq.tolist():
            pairs.add((a, b))
    return pairs


def _interface_triangles(
    lab: NDArray[np.uint32], spacing: Spacing, a: int, b: int, band_um: float | None = None
) -> dict[str, NDArray[np.float64]] | None:
    """Mesh label ``a``'s boundary and keep the triangles that sit physically
    close to label ``b`` (see module docstring for the physical-distance
    selection band). Returns ``None`` when no such interface mesh exists
    (e.g. a candidate pair that only touches at a single corner/edge with no
    real shared area).

    This is the single source of triangle geometry shared by ``contact_area``
    itself (when ``estimator`` is MARCHING_CUBES) and by the orientation
    diagnostics below, which are computed regardless of which estimator
    produced the headline area -- a FACE_COUNT contact still gets an
    orientation descriptor and a reliability flag for its area, because the
    same digitisation bias applies whichever estimator reports the number.

    ``band_um`` overrides the default selection-band width (``max(spacing)``)
    -- used by ``_topology_stability`` to ask "how much area would a
    one-voxel-more-generous match against ``b`` have picked up," without
    constructing and re-meshing a separately dilated label array.
    """
    coords = np.argwhere((lab == a) | (lab == b))
    if coords.size == 0:
        return None

    default_band_um = max(spacing)
    band_um = default_band_um if band_um is None else band_um
    pad_reach_um = max(default_band_um, band_um)
    pad = tuple(int(np.ceil(pad_reach_um / s)) + 1 for s in spacing)
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    shape = np.asarray(lab.shape)
    lo_p = np.maximum(lo - np.asarray(pad), 0)
    hi_p = np.minimum(hi + np.asarray(pad), shape - 1)
    slc = tuple(slice(int(lo_p[i]), int(hi_p[i]) + 1) for i in range(3))
    crop = lab[slc]

    A = crop == a
    if not A.any():
        return None
    Ap = np.pad(A.astype(np.float32), 1)
    try:
        verts, faces, _normals, _values = marching_cubes(Ap, level=0.5, spacing=spacing)
    except (RuntimeError, ValueError):
        return None
    if faces.size == 0:
        return None

    tri = verts[faces]
    cen = tri.mean(axis=1)
    idx = np.rint(cen / np.asarray(spacing)).astype(int) - 1
    np.clip(idx, 0, np.asarray(crop.shape) - 1, out=idx)

    Bmask = crop == b
    if not Bmask.any():
        return None
    dist = ndi.distance_transform_edt(~Bmask, sampling=spacing)
    on_iface = dist[idx[:, 0], idx[:, 1], idx[:, 2]] <= band_um
    if not on_iface.any():
        return None

    sel_tri = tri[on_iface]
    sel_idx = idx[on_iface]
    edge1 = sel_tri[:, 1] - sel_tri[:, 0]
    edge2 = sel_tri[:, 2] - sel_tri[:, 0]
    cross = np.cross(edge1, edge2)
    areas = np.linalg.norm(cross, axis=1) / 2.0
    valid = areas > 0
    if not valid.any():
        return None

    cross = cross[valid]
    areas = areas[valid]
    normals = cross / np.linalg.norm(cross, axis=1, keepdims=True)
    centroids = sel_tri[valid].mean(axis=1)
    z_plane_idx = sel_idx[valid, 0]

    return {
        "areas": areas,
        "normals": normals,
        "centroids": centroids,
        "z_plane_idx": z_plane_idx.astype(np.float64),
        "total_area_um2": np.array(float(areas.sum())),
    }


def _mc_pair_area_um2(lab: NDArray[np.uint32], spacing: Spacing, a: int, b: int) -> float:
    mesh = _interface_triangles(lab, spacing, a, b)
    return float(mesh["total_area_um2"]) if mesh is not None else 0.0


# ─── orientation descriptor (Section 1) ────────────────────────────────────


def _weighted_percentile(
    values: NDArray[np.float64], weights: NDArray[np.float64], q: float
) -> float:
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cw = np.cumsum(w)
    cw = cw / cw[-1]
    idx = int(np.searchsorted(cw, q / 100.0))
    idx = min(idx, len(v) - 1)
    return float(v[idx])


def _area_weighted_plane_residual_um(
    points_um: NDArray[np.float64], weights: NDArray[np.float64]
) -> float:
    """RMS distance of the interface triangle centroids (physical um, z/y/x)
    to their own area-weighted best-fit plane. Near zero for a flat disc;
    grows for a curved or noisy interface patch. Fit by weighted PCA: the
    plane normal is the eigenvector of the weighted covariance with the
    smallest eigenvalue."""
    if points_um.shape[0] < 3:
        return 0.0
    w = weights / weights.sum()
    mean = np.sum(points_um * w[:, None], axis=0)
    centered = points_um - mean
    cov = (centered * w[:, None]).T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)
    normal = eigvecs[:, int(np.argmin(eigvals))]
    residuals = centered @ normal
    return float(np.sqrt(np.sum(w * residuals**2)))


_ORIENTATION_KEYS = (
    "interface_normal_angle_to_z_mean_deg",
    "interface_normal_angle_to_z_std_deg",
    "interface_normal_angle_to_z_p10_deg",
    "interface_normal_angle_to_z_p50_deg",
    "interface_normal_angle_to_z_p90_deg",
    "n_axial_planes_supporting",
    "interface_planarity_residual_um",
)


def _orientation_stats_from_mesh(mesh: dict[str, NDArray[np.float64]] | None) -> dict[str, Scalar]:
    """Area-weighted distribution of interface-normal angle to the optical
    (Z) axis, plus how many distinct Z planes the interface spans and a
    planarity residual. Deliberately a distribution, not a single angle: the
    design review that motivated this module was explicit that one
    centroid-derived angle cannot describe an interface that is curved,
    between unequal cells, or displaced -- see
    ``_crude_centroid_vector_angle_to_z_deg`` for that cruder, separately
    named fallback, which this function does NOT use.

    Angles are folded into [0, 90] degrees (``abs`` of the normal's Z
    component before ``arccos``): a triangle normal's sign is a meshing
    artefact (which side it happens to point), not orientation information,
    so signed angles would just add meaningless bimodality.
    """
    if mesh is None or mesh["areas"].size == 0:
        return {k: None for k in _ORIENTATION_KEYS}

    areas = mesh["areas"]
    normals = mesh["normals"]
    cos_to_z = np.clip(np.abs(normals[:, 0]), 0.0, 1.0)
    angles_deg = np.degrees(np.arccos(cos_to_z))

    w = areas / areas.sum()
    mean = float(np.sum(w * angles_deg))
    variance = float(np.sum(w * (angles_deg - mean) ** 2))
    std = float(np.sqrt(max(variance, 0.0)))

    n_planes = int(np.unique(mesh["z_plane_idx"]).size)
    residual = _area_weighted_plane_residual_um(mesh["centroids"], areas)

    return {
        "interface_normal_angle_to_z_mean_deg": mean,
        "interface_normal_angle_to_z_std_deg": std,
        "interface_normal_angle_to_z_p10_deg": _weighted_percentile(angles_deg, areas, 10.0),
        "interface_normal_angle_to_z_p50_deg": _weighted_percentile(angles_deg, areas, 50.0),
        "interface_normal_angle_to_z_p90_deg": _weighted_percentile(angles_deg, areas, 90.0),
        "n_axial_planes_supporting": n_planes,
        "interface_planarity_residual_um": residual,
    }


def _crude_centroid_vector_angle_to_z_deg(
    centroid_a: NDArray[np.float64] | None, centroid_b: NDArray[np.float64] | None
) -> float | None:
    """Crude fallback orientation descriptor: angle of the centroid-to-centroid
    vector to Z. Clearly named and kept separate from the mesh-derived
    descriptor above because it is known-bad for unequal, deformed, or
    displaced cells (where the line between centroids need not be normal to
    the true contact interface at all) -- it is provided only as a cheap
    sanity check that requires no mesh, never as the reliability basis."""
    if centroid_a is None or centroid_b is None:
        return None
    vec = np.asarray(centroid_b, dtype=np.float64) - np.asarray(centroid_a, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        return None
    cos_to_z = float(np.clip(abs(vec[0]) / norm, 0.0, 1.0))
    return float(np.degrees(np.arccos(cos_to_z)))


# ─── categorical reliability flag (Section 2) ──────────────────────────────

# Thresholds, not a fitted curve: we have measured exactly three orientations
# of one phantom geometry (axis-aligned, 45deg-in-XY, tilted-in-3-axes), which
# is enough to say "this region behaves like the case we validated" or "this
# region does not," but nowhere near enough to fit error-vs-angle -- the
# phantom does not vary contact size, curvature, subvoxel phase, PSF, noise,
# or segmentation error, so a continuous "predicted relative error" would be
# false precision laundered through a regression on three points. See the
# module docstring's "WHAT THIS MODULE DELIBERATELY DOES NOT DO" section.
#
# The thresholds below are set by measurement, NOT by the intuition that an
# interface facing the optical axis must be the best-sampled one. Error is not
# monotonic in the normal's angle to Z. Measured on the default production path
# (spacing 0.5/0.1/0.1 um, MARCHING_CUBES, isotropic resampling) against the
# analytic bisected contact disc:
#
#     interface normal      angle to Z     error
#     along Z                      0 deg    4.5%   <- best
#     along X                     90 deg   13.5%
#     along Y                     90 deg   13.4%
#     45 deg within XY            90 deg   15.9%
#     equal parts X, Y, Z         55 deg   42.8%   <- WORST
#
# The error PEAKS in the middle, near 55 deg. That is the body-diagonal
# direction, where the interface is maximally misaligned with every voxel face
# and the staircase bias is largest -- so both extremes beat the middle.
#
# Two effects compose. Alignment of the normal to any grid axis controls the
# staircase bias, and how finely the interface PLANE itself is sampled controls
# the rest: a normal along Z puts the interface in the finely sampled XY plane
# (0.1 x 0.1 um), while a normal along X puts it in YZ, sampled 0.1 x 0.5 um.
# That is why along-Z (4.5%) beats along-X (13.5%) even though both are
# perfectly axis-aligned.
#
# An earlier version of this function graded monotonically in angle-to-Z. It
# gave the 42.8% case "medium" and the 13.5% case "low" -- inverting the two
# where it matters most. A reliability flag that ranks the worst measurement
# above a middling one is worse than no flag, because it is exactly the field a
# downstream user would filter or weight on.
_RELIABILITY_HIGH_MAX_ANGLE_DEG = 20.0   # near-Z normals: measured 4.5%
_RELIABILITY_MEDIUM_MIN_ANGLE_DEG = 75.0  # near-XY-plane normals: 13.4-15.9%


def _contact_area_reliability(mean_angle_to_z_deg: float | None) -> str:
    """Grade an interface by how well this sampling geometry measures it.

    Categorical rather than a continuous predicted error: five phantom
    orientations cannot support a calibration curve, and publishing one would
    be false precision. See the module docstring's "WHAT THIS MODULE
    DELIBERATELY DOES NOT DO" section.
    """
    if mean_angle_to_z_deg is None:
        return "low"  # no orientation evidence at all -- treat conservatively
    if mean_angle_to_z_deg <= _RELIABILITY_HIGH_MAX_ANGLE_DEG:
        return "high"
    if mean_angle_to_z_deg >= _RELIABILITY_MEDIUM_MIN_ANGLE_DEG:
        return "medium"
    # Intermediate angles are the diagonal regime: worst measured staircase bias.
    return "low"


# ─── topological stability (Section 3) ─────────────────────────────────────


def _topology_stability(
    lab: NDArray[np.uint32],
    spacing: Spacing,
    a: int,
    b: int,
    nominal_area_um2: float | None = None,
) -> dict[str, Scalar]:
    """Perturb the digital boundary by one voxel and see whether this
    contact's existence and area are stable to it.

    ``contact_survives_erosion``: erode the COMBINED footprint ``A | B`` by
    one voxel (full 26-connectivity structuring element) as a single object,
    then ask whether any connected component of what remains still contains
    both some A-voxels and some B-voxels. Eroding the union rather than each
    label separately is deliberate: eroding A and B independently and then
    checking whether the shrunken cores still touch is a broken test, because
    the shared A/B boundary layer is, by construction, always "surface" from
    each label's own point of view (a voxel of A touching B has a non-A
    neighbour) -- so independent erosion removes it from BOTH sides even for
    an arbitrarily large, robust, perfectly flat interface, and the test
    would wrongly flag every contact as unstable. Eroding the union only
    removes voxels touching true background, so an interior interface
    (however it is split between the two labels) survives as long as the
    LOCAL NECK connecting the two cells is wider than one voxel everywhere.
    A neck that is exactly one voxel wide -- the spurious-bridge failure mode
    the module docstring warns about for coordination number -- is deleted
    in its entirety by this erosion, disconnecting the two labels within the
    eroded union.

    ``contact_area_ratio_dilated``: re-run the same physical-distance
    triangle selection (``_interface_triangles``) with the band widened by
    one extra voxel step, divided by the nominal (default-band) area. This
    asks "how much more area would a one-voxel-more-generous match against
    the other label have picked up" without constructing and re-meshing a
    separately dilated label array (which has its own pathology: naively
    dilating both labels and excluding the now-doubly-claimed voxels can
    swallow an entire flat, well-aligned interface into the excluded region
    and report a nonsensical zero). ``None`` when there is no nominal meshed
    area to divide by (e.g. a corner-only touch with no real interface mesh).

    ``nominal_area_um2`` lets a caller that already meshed this pair (e.g.
    ``compute_contacts`` under MARCHING_CUBES) pass that area straight
    through instead of paying for a second meshing pass; when omitted it is
    computed fresh with the default selection band.

    The erosion check operates on a small local crop containing only these
    two cells' bounding box (plus margin), and only compares these two
    labels' binary masks to each other -- other labels elsewhere in the
    volume never participate, so this is purely a statement about this one
    pair.
    """
    coords = np.argwhere((lab == a) | (lab == b))
    if coords.size == 0:
        return {"contact_survives_erosion": False, "contact_area_ratio_dilated": None}

    pad = 2
    lo = np.maximum(coords.min(axis=0) - pad, 0)
    shape = np.asarray(lab.shape)
    hi = np.minimum(coords.max(axis=0) + pad, shape - 1)
    slc = tuple(slice(int(lo[i]), int(hi[i]) + 1) for i in range(3))
    crop = lab[slc]

    A = crop == a
    B = crop == b

    union_er = ndi.binary_erosion(A | B, structure=_STRUCTURE_26, border_value=0)
    labeled, n_components = ndi.label(union_er, structure=_STRUCTURE_26)
    survives = False
    for comp_id in range(1, n_components + 1):
        comp = labeled == comp_id
        if np.any(comp & A) and np.any(comp & B):
            survives = True
            break

    if nominal_area_um2 is None:
        nominal_mesh = _interface_triangles(lab, spacing, a, b)
        nominal_area_um2 = float(nominal_mesh["total_area_um2"]) if nominal_mesh is not None else 0.0

    if nominal_area_um2 > 0:
        widened_band = max(spacing) + max(spacing)
        dilated_mesh = _interface_triangles(lab, spacing, a, b, band_um=widened_band)
        dilated_area = float(dilated_mesh["total_area_um2"]) if dilated_mesh is not None else 0.0
        ratio = dilated_area / nominal_area_um2
    else:
        ratio = None

    return {
        "contact_survives_erosion": survives,
        "contact_area_ratio_dilated": ratio,
    }


# ─── resampling ────────────────────────────────────────────────────────────


def _resample_isotropic_nn(
    lab: NDArray[np.uint32], spacing: Spacing
) -> tuple[NDArray[np.uint32], Spacing]:
    """Nearest-neighbour resample of the LABEL array to isotropic spacing at
    the finest axis's resolution. Nearest-neighbour is the only resampling
    that cannot invent a label value that never existed in the source."""
    target = min(spacing)
    if all(abs(s - target) < 1e-9 for s in spacing):
        return lab, spacing
    factors = tuple(s / target for s in spacing)
    resampled = ndi.zoom(lab, zoom=factors, order=0, mode="nearest")
    return resampled.astype(np.uint32), (target, target, target)


def _centroids_um(
    lab: NDArray[np.uint32], spacing: Spacing, ids: Iterable[int]
) -> dict[int, NDArray[np.float64]]:
    sp = np.asarray(spacing, dtype=np.float64)
    out: dict[int, NDArray[np.float64]] = {}
    for i in ids:
        coords = np.argwhere(lab == i)
        if coords.size == 0:
            continue
        out[i] = (coords.astype(np.float64) + 0.5).mean(axis=0) * sp
    return out


# ─── public API ────────────────────────────────────────────────────────────


def compute_contacts(
    labels: LabelVolume,
    config: ContactConfig,
    truncated_cell_ids: Iterable[int] = (),
) -> list[ContactRecord]:
    """Pairwise cell-cell contact interface areas for one field, plus the
    orientation, reliability, and topological-stability metadata needed to
    interpret them (see module docstring: area is secondary and
    orientation-conditioned; coordination number is primary but carries its
    own discrete edge-error risk).

    Honours ``config.estimator`` and
    ``config.resample_isotropic_before_contact`` for ``contact_area_um2``
    itself -- see ``ContactEstimator``'s docstring for why marching_cubes +
    isotropic resampling is the default. When resampling, the LABEL array is
    resampled with nearest-neighbour interpolation and every physical
    measurement that follows uses the resulting isotropic spacing. Reported
    areas are always physical um^2, on whichever grid was actually measured.

    A row is emitted for every touching pair regardless of area -- rows are
    never dropped. ``qualifies_as_contact`` is true only when the area is at
    least ``config.minimum_contact_area_um2``; a sub-threshold contact is
    recorded but must not enter a contact graph. ``valid_for_contact_metrics``
    is false whenever either member cell id is in ``truncated_cell_ids`` -- a
    cropped cell's true contact area is unknown, but the row is kept so
    callers can see what was excluded and why.

    ``values`` additionally carries (all computed regardless of which
    ``estimator`` produced ``contact_area_um2``, since the same digitisation
    bias affects both): ``centroid_distance_um``; the orientation descriptor
    fields in ``_ORIENTATION_KEYS`` (area-weighted normal-angle-to-Z mean,
    std, p10/p50/p90, plus ``n_axial_planes_supporting`` and
    ``interface_planarity_residual_um``); ``centroid_vector_angle_to_z_deg``
    (the crude fallback, see ``_crude_centroid_vector_angle_to_z_deg``);
    ``contact_area_reliability`` (categorical, see
    ``_contact_area_reliability``); and the topological-stability pair
    ``contact_survives_erosion`` / ``contact_area_ratio_dilated`` (see
    ``_topology_stability``).

    ``cell_id_a < cell_id_b`` always -- enforced by ``ContactRecord`` itself,
    and every pair here is built through ``_ordered`` so no pair is ever
    emitted twice.
    """
    lab = labels.cells
    spacing = labels.geometry.spacing_um_zyx

    if config.resample_isotropic_before_contact:
        lab, spacing = _resample_isotropic_nn(lab, spacing)

    truncated = {int(i) for i in truncated_cell_ids}

    areas: dict[Pair, float]
    mesh_cache: dict[Pair, dict[str, NDArray[np.float64]] | None] = {}
    if config.estimator is ContactEstimator.FACE_COUNT:
        areas = _face_adjacency_areas(lab, spacing)
    elif config.estimator is ContactEstimator.MARCHING_CUBES:
        areas = {}
        for a, b in sorted(_candidate_pairs_26(lab)):
            mesh = _interface_triangles(lab, spacing, a, b)
            mesh_cache[(a, b)] = mesh
            areas[(a, b)] = float(mesh["total_area_um2"]) if mesh is not None else 0.0
    else:  # pragma: no cover - exhaustive over the frozen enum
        raise ValueError(f"unknown estimator {config.estimator!r}")

    involved_ids = {i for pair in areas for i in pair}
    centroids = _centroids_um(lab, spacing, involved_ids)

    records: list[ContactRecord] = []
    for (a, b), area in sorted(areas.items()):
        mesh = mesh_cache[(a, b)] if (a, b) in mesh_cache else _interface_triangles(lab, spacing, a, b)
        orientation = _orientation_stats_from_mesh(mesh)
        nominal_area_um2 = float(mesh["total_area_um2"]) if mesh is not None else 0.0
        topology = _topology_stability(lab, spacing, a, b, nominal_area_um2=nominal_area_um2)

        values: dict[str, Scalar] = {}
        if a in centroids and b in centroids:
            values["centroid_distance_um"] = float(np.linalg.norm(centroids[a] - centroids[b]))
        values.update(orientation)
        values["centroid_vector_angle_to_z_deg"] = _crude_centroid_vector_angle_to_z_deg(
            centroids.get(a), centroids.get(b)
        )
        values["contact_area_reliability"] = _contact_area_reliability(
            orientation["interface_normal_angle_to_z_mean_deg"]
        )
        values.update(topology)

        records.append(
            ContactRecord(
                dataset_id=labels.identity.dataset_id,
                field_id=labels.identity.field_id,
                segmentation_run_id=labels.provenance.run_id,
                cell_id_a=a,
                cell_id_b=b,
                contact_area_um2=area,
                estimator=config.estimator,
                qualifies_as_contact=area >= config.minimum_contact_area_um2,
                valid_for_contact_metrics=(a not in truncated and b not in truncated),
                values=values,
            )
        )
    return records


def compute_adhesion_aggregates(
    contacts: Sequence[ContactRecord],
    cell_ids: Iterable[int],
    cell_surface_areas_um2: Mapping[int, float | None],
) -> dict[int, dict[str, Scalar]]:
    """Per-cell adhesion summary keyed by cell id.

    Only contacts with both ``qualifies_as_contact`` and
    ``valid_for_contact_metrics`` true count toward a cell's coordination
    number or contact-area totals: a sub-threshold contact is not adhesion,
    and a contact touching a truncated cell is not a trustworthy area
    measurement for either member.

    ``coordination_number_stable`` is the same count restricted further to
    contacts whose ``values["contact_survives_erosion"]`` is ``True`` -- see
    ``_topology_stability``. A contact can be a real, qualifying,
    non-truncated contact and STILL not survive erosion (a single spurious
    bridge voxel), which is exactly the discrete edge-error risk the module
    docstring describes for coordination number: report both counts, do not
    silently prefer one. ``coordination_number`` on its own is the "unstable"
    (unfiltered) count; the gap between the two numbers for a cell is itself
    diagnostic.

    ``cell_surface_areas_um2`` is accepted as an argument rather than computed
    here so this module does not depend on the concurrently-developed
    geometry module; a missing or ``None`` entry makes
    ``contact_surface_fraction`` ``None`` for that cell.

    Fields per cell: ``coordination_number`` (int, count of qualifying
    neighbours), ``coordination_number_stable`` (int, the erosion-surviving
    subset), ``total_contact_area_um2`` (float, 0.0 when isolated),
    ``mean_contact_area_um2`` (``None`` when coordination is 0),
    ``max_contact_area_um2`` (``None`` when coordination is 0),
    ``contact_surface_fraction`` (total contact area / that cell's surface
    area; ``None`` when the surface area is unknown).
    """
    cell_ids = [int(i) for i in cell_ids]
    per_cell_areas: dict[int, list[float]] = {i: [] for i in cell_ids}
    per_cell_stable_areas: dict[int, list[float]] = {i: [] for i in cell_ids}
    for rec in contacts:
        if not (rec.qualifies_as_contact and rec.valid_for_contact_metrics):
            continue
        per_cell_areas.setdefault(rec.cell_id_a, []).append(rec.contact_area_um2)
        per_cell_areas.setdefault(rec.cell_id_b, []).append(rec.contact_area_um2)
        if rec.values.get("contact_survives_erosion") is True:
            per_cell_stable_areas.setdefault(rec.cell_id_a, []).append(rec.contact_area_um2)
            per_cell_stable_areas.setdefault(rec.cell_id_b, []).append(rec.contact_area_um2)

    out: dict[int, dict[str, Scalar]] = {}
    for cid, areas in per_cell_areas.items():
        coordination = len(areas)
        total = float(sum(areas))
        surface = cell_surface_areas_um2.get(cid)
        out[cid] = {
            "coordination_number": coordination,
            "coordination_number_stable": len(per_cell_stable_areas.get(cid, [])),
            "total_contact_area_um2": total,
            "mean_contact_area_um2": (total / coordination) if coordination > 0 else None,
            "max_contact_area_um2": max(areas) if coordination > 0 else None,
            "contact_surface_fraction": (
                total / surface if (surface is not None and surface > 0) else None
            ),
        }
    return out
