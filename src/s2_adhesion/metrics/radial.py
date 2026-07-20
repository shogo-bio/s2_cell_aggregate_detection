"""Radial signal distribution: is a protein at the membrane or inside the cell?

The biology this serves: each channel is a protein tagged with an antibody or
GFP. Most carry a transmembrane domain, so their signal forms a shell around the
cell — a ring in any single plane, often an INCOMPLETE one, because the protein
is not expressed over the whole surface. Constructs that fail to localise
instead light up the cell interior. Telling those two cases apart, per cell, is
the readout.

Why a radial profile rather than only the inner/core shell ratio already in
``localization.py``: the shell ratio compares two absolute physical bands, so it
is sensitive to where the segmentation put the boundary and to cell size. The
radial coordinate here is normalised per cell — 0 at the deepest interior point,
1 at the boundary — which makes cells of different sizes directly comparable and
makes the shape of the distribution, not its absolute position, carry the
answer.

The normalised coordinate is derived from the interior euclidean distance
transform rather than by casting rays from the centroid. EDT is anisotropy-aware
(``sampling=spacing``), needs no ray/surface intersection, and degrades
gracefully for non-convex cells, where a ray from the centroid can exit and
re-enter the mask.

Everything here is a VOLUME integral. That matters at this project's 5x axial
anisotropy: volumetric quantities measure accurately (~0.4% on volume) while
surface reconstructions do not (see docs/metric_interpretation.md). A radial
profile inherits the good behaviour, not the bad.

WHICH FIELD TO ACTUALLY USE, and one trap
-----------------------------------------
``radial_shell_score`` is the discriminator. It compares bin MEANS, so it is not
volume weighted: a uniform signal scores ~0, a shell scores well above 1, an
interior blob scores below -1.

``peripheral_signal_fraction`` integrates, and a sphere keeps most of its volume
near its surface — the outer 30% of the radius already contains
1 - 0.7^3 = 66% of it. So a perfectly uniform signal scores ~0.66, which is close
to a genuine shell, and even signal confined to the inner 80% of the radius still
scores ~0.30. It is useful context but a WEAK discriminator on its own, and
reading it as "fraction of protein at the membrane" would overcall membrane
localisation for every diffuse cytoplasmic protein.

Radial position and surface COVERAGE are also different questions. A protein
expressed over only part of the cell surface is still membrane-localised, and
scores as such here; how much of the surface it covers is measured elsewhere.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage as ndi

from ..contracts import Scalar, VoxelGeometry

# Fraction of the normalised radius counted as "peripheral" when reducing the
# profile to a single number. 0.7 puts the outer 30% of the depth range in the
# periphery, which is where a membrane shell sits for a roughly spherical cell.
PERIPHERAL_RADIUS_FRACTION = 0.7

# Guard for log ratios. Callers should pass a background-scaled value; this is
# only a fallback so a zero core cannot produce an infinity.
DEFAULT_EPS = 1e-6


def normalised_radius(
    cell_mask: NDArray[np.bool_], geometry: VoxelGeometry
) -> NDArray[np.float64]:
    """Per-voxel depth coordinate: 0 at the deepest interior point, 1 at the edge.

    ``sampling`` makes the transform physical, so a voxel one step away in Z
    (0.5 um) is correctly further from the boundary than one step in X (0.1 um).
    Using voxel counts here would tilt the whole profile along the optical axis.
    """
    if not cell_mask.any():
        return np.zeros(cell_mask.shape, dtype=np.float64)
    depth = ndi.distance_transform_edt(cell_mask, sampling=geometry.spacing_um_zyx)
    deepest = float(depth.max())
    if deepest <= 0.0:
        # A single-voxel-thick cell has no interior to speak of; call it all edge.
        return np.where(cell_mask, 1.0, 0.0)
    return np.where(cell_mask, 1.0 - depth / deepest, 0.0)


def radial_profile(
    intensity: NDArray[np.floating],
    cell_mask: NDArray[np.bool_],
    geometry: VoxelGeometry,
    n_bins: int = 10,
) -> dict[str, list[float]]:
    """Bin a cell's signal by normalised radius.

    Returns per-bin voxel counts, sampled volumes, mean intensity and integrated
    intensity. Empty bins are kept with a mean of ``nan`` so the bin axis is the
    same length for every cell and every channel.
    """
    r = normalised_radius(cell_mask, geometry)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    voxel_volume = geometry.voxel_volume_um3

    counts: list[float] = []
    volumes: list[float] = []
    means: list[float] = []
    integrated: list[float] = []

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # Include the top edge in the outermost bin so boundary voxels are not lost.
        in_bin = cell_mask & (r >= lo) & (r <= hi if i == n_bins - 1 else r < hi)
        n = int(in_bin.sum())
        counts.append(float(n))
        volumes.append(n * voxel_volume)
        if n == 0:
            means.append(float("nan"))
            integrated.append(0.0)
        else:
            values = intensity[in_bin]
            means.append(float(values.mean()))
            integrated.append(float(values.sum() * voxel_volume))

    return {
        "bin_edges": edges.tolist(),
        "voxel_count": counts,
        "sampled_volume_um3": volumes,
        "mean_intensity": means,
        "integrated_intensity": integrated,
    }


def _profile_scalars(profile: Mapping[str, Sequence[float]], eps: float) -> dict[str, Scalar]:
    """Reduce a radial profile to the numbers a biologist would actually read."""
    means = np.asarray(profile["mean_intensity"], dtype=np.float64)
    integrated = np.asarray(profile["integrated_intensity"], dtype=np.float64)
    edges = np.asarray(profile["bin_edges"], dtype=np.float64)
    centres = (edges[:-1] + edges[1:]) / 2.0

    populated = ~np.isnan(means)
    if not populated.any():
        return {
            "radial_peak_position": None,
            "peripheral_signal_fraction": None,
            "radial_shell_score": None,
            "radial_profile_qc": "no_populated_bins",
        }

    peak_index = int(np.nanargmax(np.where(populated, means, -np.inf)))

    total = float(integrated.sum())
    peripheral_bins = centres >= PERIPHERAL_RADIUS_FRACTION
    peripheral_fraction = (
        float(integrated[peripheral_bins].sum() / total) if total > 0 else None
    )

    # Ratio of the outer band to the innermost band. Positive means shell-like,
    # near zero means uniform, negative means interior-concentrated.
    outer = means[peripheral_bins & populated]
    core = means[(centres < 1.0 - PERIPHERAL_RADIUS_FRACTION) & populated]
    if outer.size and core.size:
        shell_score = float(np.log2((outer.mean() + eps) / (core.mean() + eps)))
    else:
        shell_score = None

    return {
        "radial_peak_position": float(centres[peak_index]),
        "peripheral_signal_fraction": peripheral_fraction,
        "radial_shell_score": shell_score,
        "radial_profile_qc": None,
    }


def compute_radial_metrics(
    labels: NDArray[np.uint32],
    channel_intensities: Mapping[str, NDArray[np.floating]],
    geometry: VoxelGeometry,
    *,
    background_by_channel: Mapping[str, float] | None = None,
    eps_by_channel: Mapping[str, float] | None = None,
    n_bins: int = 10,
) -> dict[int, dict[str, Scalar]]:
    """Radial metrics for every cell and every supplied channel.

    Keys are namespaced ``ch.<channel_id>.<metric>`` to match ``intensity.py``.

    Like every other module in ``metrics``, this is role-agnostic: it computes
    the same numbers for whatever channels it is handed and never asks what a
    channel stains. Deciding that a high ``radial_shell_score`` means "correctly
    membrane-localised" is interpretation, and interpretation needs the channel
    roles that only the config can supply.
    """
    background_by_channel = background_by_channel or {}
    eps_by_channel = eps_by_channel or {}

    out: dict[int, dict[str, Scalar]] = {}
    cell_ids = np.unique(labels)
    cell_ids = cell_ids[cell_ids > 0]

    for cell_id in cell_ids:
        mask = labels == cell_id
        values: dict[str, Scalar] = {}
        for channel_id, raw in channel_intensities.items():
            background = float(background_by_channel.get(channel_id, 0.0))
            corrected = np.clip(
                np.asarray(raw, dtype=np.float64) - background, 0.0, None
            )
            profile = radial_profile(corrected, mask, geometry, n_bins=n_bins)
            eps = float(eps_by_channel.get(channel_id, DEFAULT_EPS))
            for name, value in _profile_scalars(profile, eps).items():
                values[f"ch.{channel_id}.{name}"] = value
        out[int(cell_id)] = values

    return out
