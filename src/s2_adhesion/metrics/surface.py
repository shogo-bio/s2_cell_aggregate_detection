"""Free versus contact surface, and which of it the microscope can actually see.

Why this module exists. The readout we want is whether a transmembrane protein
covers the whole cell surface or only part of it. Measuring that naively over
the whole surface is wrong in a dense aggregate, because where two cells touch:

  * the fluorescence physically belongs to BOTH membranes,
  * the two membranes sit closer together than the axial PSF, so the image
    cannot separate them even in principle, and
  * the segmentation boundary drawn between them carries no real information.

Signal there gets split between neighbours by an arbitrary surface. So an
apparent GAP in a cell's shell may be genuine absence of the protein (the
biology), or the neighbour's label having taken the signal, or two apposed
membranes blurred into one blob. Those are not distinguishable from the image.

The honest treatment is to measure coverage over the FREE surface only and
report the contact-apposed surface as missing data, with explicit bounds. That
is what this module produces: a per-cell breakdown of surface into free,
contact, and a PSF-width guard band between them, plus flags marking contacts
where signal ownership is not identifiable.

The unidentifiability is not a limitation of our implementation. For two
membranes at separation d along an interface normal, the observed intensity is
``I(s) = a_A h(s + d/2) + a_B h(s - d/2)``; as d falls below the PSF width the
individual amplitudes a_A and a_B stop being estimable and only their sum is.
No amount of processing recovers them, so the pipeline flags the condition
rather than deconvolving it.

Areas are computed from a marching-cubes mesh of the cell, with each triangle
classified by what lies on its far side. Note that absolute surface areas
inherit the orientation-dependent error documented in
docs/metric_interpretation.md -- the FRACTIONS here are more trustworthy than
the absolute areas, because numerator and denominator share much of the bias.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage as ndi
from skimage.measure import marching_cubes

from ..config import OpticsConfig
from ..contracts import Scalar, VoxelGeometry


class SurfaceClass:
    """What lies immediately outside a piece of a cell's boundary."""

    FREE = "free"
    CONTACT = "contact"
    GUARD = "guard"


def _triangle_areas_and_normals(
    verts: NDArray[np.floating], faces: NDArray[np.integer]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    tri = verts[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norms = np.linalg.norm(cross, axis=1)
    areas = norms / 2.0
    with np.errstate(invalid="ignore", divide="ignore"):
        normals = cross / norms[:, None]
    normals = np.nan_to_num(normals)
    return areas, normals


def classify_cell_surface(
    labels: NDArray[np.uint32],
    cell_id: int,
    geometry: VoxelGeometry,
    optics: OpticsConfig,
) -> dict[str, Scalar]:
    """Split one cell's surface into free / contact / guard and measure each.

    The guard band is the portion of surface within one PSF width of a
    neighbouring cell but not directly apposed to it. It is separated out rather
    than merged into either side because signal there is contaminated by the
    neighbour without the surface itself being a contact -- counting it as free
    would import the neighbour's fluorescence into a "clean" measurement.
    """
    cell = labels == cell_id
    if not cell.any():
        return {"surface_qc": "cell_absent"}

    others = (labels > 0) & ~cell

    padded = np.pad(cell.astype(np.float32), 1)
    try:
        verts, faces, _, _ = marching_cubes(
            padded, level=0.5, spacing=geometry.spacing_um_zyx
        )
    except (RuntimeError, ValueError):
        return {"surface_qc": "mesh_failed"}
    if len(faces) == 0:
        return {"surface_qc": "mesh_empty"}

    areas, normals = _triangle_areas_and_normals(verts, faces)

    # Map each triangle centroid back to a voxel index (undo the pad).
    spacing = np.asarray(geometry.spacing_um_zyx)
    centroids = verts[faces].mean(axis=1)
    idx = np.rint(centroids / spacing).astype(int) - 1
    np.clip(idx, 0, np.asarray(labels.shape) - 1, out=idx)

    # Directly apposed: another label within one voxel.
    contact_zone = ndi.binary_dilation(others, ndi.generate_binary_structure(3, 3))
    is_contact = contact_zone[idx[:, 0], idx[:, 1], idx[:, 2]]

    # Guard band: within a PSF width of a neighbour, measured PHYSICALLY so the
    # band is the same micrometre distance along Z as along X.
    if others.any():
        distance_to_other = ndi.distance_transform_edt(
            ~others, sampling=geometry.spacing_um_zyx
        )
        per_triangle_resolution = np.array(
            [optics.resolution_along_um(tuple(n)) for n in normals]
        )
        near = (
            distance_to_other[idx[:, 0], idx[:, 1], idx[:, 2]]
            <= per_triangle_resolution
        )
        is_guard = near & ~is_contact
    else:
        is_guard = np.zeros(len(areas), dtype=bool)

    is_free = ~is_contact & ~is_guard

    total = float(areas.sum())
    free_area = float(areas[is_free].sum())
    contact_area = float(areas[is_contact].sum())
    guard_area = float(areas[is_guard].sum())

    # Mean effective resolution over the contact surface: how badly the optics
    # are compromised where this cell touches its neighbours.
    if is_contact.any():
        contact_resolution = float(
            np.average(
                [optics.resolution_along_um(tuple(n)) for n in normals[is_contact]],
                weights=areas[is_contact],
            )
        )
    else:
        contact_resolution = None

    return {
        "surface_area_total_um2": total,
        "surface_area_free_um2": free_area,
        "surface_area_contact_um2": contact_area,
        "surface_area_guard_um2": guard_area,
        "free_surface_fraction": free_area / total if total > 0 else None,
        "contact_surface_fraction_meshed": contact_area / total if total > 0 else None,
        "guard_surface_fraction": guard_area / total if total > 0 else None,
        # The denominator for any coverage measurement: surface where signal
        # ownership is unambiguous. Everything else is missing data.
        "observable_surface_fraction": free_area / total if total > 0 else None,
        "mean_contact_resolution_um": contact_resolution,
        "surface_qc": None,
    }


def compute_surface_partition(
    labels: NDArray[np.uint32],
    geometry: VoxelGeometry,
    optics: OpticsConfig,
    *,
    min_observable_fraction: float = 0.1,
) -> dict[int, dict[str, Scalar]]:
    """Free/contact/guard surface breakdown for every cell.

    ``min_observable_fraction`` gates the coverage readout: a cell buried inside
    an aggregate may have almost no free surface left, and a coverage figure
    computed from a handful of triangles is noise. Such cells are flagged rather
    than given a confident-looking number.

    A caution for downstream analysis, and the reason the raw fractions are all
    reported rather than only the coverage: how much free surface a cell has is
    itself a function of how adhesive it is. Conditioning on it can therefore
    absorb some of the biological effect being studied. Report the free-surface
    fraction alongside any coverage figure so that confound stays visible.
    """
    out: dict[int, dict[str, Scalar]] = {}

    # Crop to each cell's bounding box (plus a margin covering the guard band,
    # which reaches out to one PSF width toward neighbours) so marching cubes and
    # the distance transform run on a small volume, not the whole field per cell.
    # Only areas are computed, so working in crop-local coordinates is exact.
    slices = ndi.find_objects(labels)
    psf = max(optics.axial_fwhm_um, optics.lateral_fwhm_um)
    pad = int(np.ceil(psf / min(geometry.spacing_um_zyx))) + 2

    for label_index, sl in enumerate(slices):
        if sl is None:
            continue
        cell_id = label_index + 1
        padded = tuple(
            slice(max(0, s.start - pad), min(dim, s.stop + pad))
            for s, dim in zip(sl, labels.shape)
        )
        row = classify_cell_surface(labels[padded], cell_id, geometry, optics)
        observable = row.get("observable_surface_fraction")
        if observable is not None and observable < min_observable_fraction:
            row["surface_qc"] = "insufficient_free_surface"
            row["coverage_measurable"] = False
        else:
            row["coverage_measurable"] = row.get("surface_qc") is None
        out[int(cell_id)] = row

    return out


def flag_unresolved_contacts(
    contact_normal_angle_to_z_deg: Mapping[tuple[int, int], float | None],
    optics: OpticsConfig,
    *,
    separation_um: Mapping[tuple[int, int], float] | None = None,
) -> dict[tuple[int, int], dict[str, Scalar]]:
    """Mark contacts whose fluorescence cannot be attributed to either cell.

    Labels that directly abut have no resolved cleft between them, so the
    membrane separation is taken as zero unless a caller measured otherwise. A
    contact is unresolved whenever that separation is below the effective
    resolution along its own normal -- which is why the angle matters: an
    interface facing the optical axis is judged against the axial PSF (~0.7 um)
    while one lying in the imaging plane is judged against the lateral one
    (~0.25 um), so axially-facing contacts are compromised first.

    ``optical_severity`` is continuous rather than binary because the
    degradation is: it is the effective resolution along the normal, in
    micrometres, so downstream analysis can weight instead of filter.
    """
    separation_um = separation_um or {}
    out: dict[tuple[int, int], dict[str, Scalar]] = {}

    for pair, angle_deg in contact_normal_angle_to_z_deg.items():
        if angle_deg is None:
            out[pair] = {
                "optically_unresolved_contact": True,
                "optical_severity_um": None,
                "contact_optics_qc": "no_orientation_evidence",
            }
            continue
        theta = np.deg2rad(angle_deg)
        # angle is measured between the normal and Z, so nz = cos(theta).
        normal = (abs(np.cos(theta)), abs(np.sin(theta)), 0.0)
        resolution = optics.resolution_along_um(normal)
        separation = float(separation_um.get(pair, 0.0))
        out[pair] = {
            "optically_unresolved_contact": bool(separation < resolution),
            "optical_severity_um": resolution,
            "membrane_separation_um": separation,
            "contact_optics_qc": None,
        }
    return out
