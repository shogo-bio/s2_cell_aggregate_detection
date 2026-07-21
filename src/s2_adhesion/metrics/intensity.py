"""Per-cell, per-channel intensity metrics.

Pure numpy/scipy -- no ML imports, so this module works with no ML package
installed at all (and importing it must never pull ``torch`` or ``cellpose``
into ``sys.modules``).

ROLE-AGNOSTICISM (the one rule that matters most)
--------------------------------------------------
Nobody has documented what the real channels stain, so this module never
branches on ``ChannelRole``, on a channel index, or on a channel's name. The
only role this module inspects is ``ChannelRole.IGNORE``, used purely as a
skip filter -- "compute nothing for this channel" -- exactly as the
orchestration contract requests ("per non-ignored channel"). Every other
role is treated identically: the same features, computed the same way, for
every configured channel. The orchestration layer -- not this module --
decides which comparisons across channels are biologically meaningful.

Output shape
------------
``compute_intensity`` returns a plain ``dict[int, dict[str, Scalar]]`` keyed
by cell id. For every non-ignored channel binding ``ch`` it adds keys
namespaced ``ch.<channel_id>.<feature>``:

  raw_mean, raw_std, raw_min, raw_max, raw_p10, raw_median, raw_p90, raw_sum
  background, background_mad
  corrected_mean, corrected_integrated_um3
  positive_fraction
  inner_shell_mean, core_mean, outer_shell_mean
  qc

A value is ``None`` (never ``0``, never ``NaN``) whenever the underlying
statistic is undefined, together with a reason recorded in ``ch.<id>.qc`` --
a semicolon-joined string drawn from a fixed vocabulary, in a fixed order,
same convention as ``metrics.qc.qc_code``:

  cell_region_empty                the cell has zero voxels in this channel
  background_region_empty          "outside_cells_median_mad": no voxel in
                                    the whole field has label 0, so there is
                                    nothing to estimate background from
  missing_fixed_background_value   "fixed" mode but this channel_id has no
                                    entry in fixed_value_by_channel
  mad_undefined_for_fixed_background
                                    "fixed" mode never yields a data-driven
                                    background_mad (the config carries only a
                                    point value), so background_mad is always
                                    None in that mode and this code always
                                    accompanies it
  zero_background_mad              the estimated background has exactly zero
                                    spread (e.g. a fully saturated/constant
                                    outside-cells region). The MAD-based
                                    threshold used by positive_fraction would
                                    then be exactly 0, which cannot separate
                                    signal from noise, so positive_fraction is
                                    reported as None rather than as a
                                    misleadingly precise number
  inner_shell_empty / core_empty / outer_shell_empty
                                    that shell contains no voxels for this
                                    cell at the given shell width(s)

Background modes (``BackgroundConfig.mode``)
---------------------------------------------
Background is estimated once per channel (not per cell) and shared by every
cell, since both supported modes describe a property of the *field*, not of
an individual cell:

  "fixed"                    background = the configured per-channel value.
                              background_mad is always None (see QC code
                              above), so positive_fraction is always None too.
  "outside_cells_median_mad" background = median, background_mad = median
                              absolute deviation from that median (no 1.4826
                              normal-consistency scaling applied -- this is a
                              raw MAD in the same units as the channel), both
                              computed over voxels where the cell label is 0.
                              Voxels inside any cell never contribute.

Physical scaling
-----------------
``corrected_integrated_um3 = voxel_volume_um3 * sum(I+)`` uses
``geometry.voxel_volume_um3``, so the same physical object sampled at two
different voxel spacings yields (approximately) the same integrated
corrected intensity -- unlike a raw voxel count or an unscaled sum, which
would scale with sampling density.

Shells
------
``inner_shell_mean``, ``core_mean`` and ``outer_shell_mean`` partition space
around a cell by *physical* distance (micrometres) from its boundary, using
``scipy.ndimage.distance_transform_edt`` with ``sampling=spacing_um_zyx``.
Passing the per-axis spacing to the EDT is essential: under typical confocal
sampling z is coarser than xy (e.g. 0.5 um vs 0.1 um), so a fixed number of
*voxel* steps is not a fixed physical distance, and would make a shell five
times thicker along z than along xy if voxel-unit distances were used
instead. Shell means are of *raw* (not background-corrected) intensity,
parallel to ``raw_mean``. A voxel belongs to "outer shell" if it lies outside
the cell within ``outer_shell_width_um`` of the boundary, regardless of
whether it belongs to another cell or to background -- this module has no
concept of "another cell" to exclude, by the same role/identity-agnostic
design as everything else here.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt

from ..config import BackgroundConfig
from ..contracts import ChannelRole, ImageVolume, Scalar
from ..errors import ContractViolation

# Fixed QC vocabulary, fixed order -- see module docstring for meanings.
_QC_ORDER: tuple[str, ...] = (
    "cell_region_empty",
    "background_region_empty",
    "missing_fixed_background_value",
    "mad_undefined_for_fixed_background",
    "zero_background_mad",
    "inner_shell_empty",
    "core_empty",
    "outer_shell_empty",
)


def _qc_string(codes: set[str]) -> str:
    return ";".join(c for c in _QC_ORDER if c in codes)


def _region_stats(values: NDArray[np.float64]) -> dict[str, Scalar]:
    """Basic distributional stats of a 1-D value array. Empty -> all None."""
    if values.size == 0:
        return {
            "raw_mean": None,
            "raw_std": None,
            "raw_min": None,
            "raw_max": None,
            "raw_p10": None,
            "raw_median": None,
            "raw_p90": None,
            "raw_sum": None,
        }
    p10, median, p90 = np.percentile(values, [10.0, 50.0, 90.0])
    return {
        "raw_mean": float(values.mean()),
        "raw_std": float(values.std()),
        "raw_min": float(values.min()),
        "raw_max": float(values.max()),
        "raw_p10": float(p10),
        "raw_median": float(median),
        "raw_p90": float(p90),
        "raw_sum": float(values.sum()),
    }


def _channel_background(
    channel_data: NDArray[np.float64],
    labels: NDArray[np.uint32],
    channel_id: str,
    background: BackgroundConfig,
) -> tuple[float | None, float | None, set[str]]:
    """(background, background_mad, qc_codes) for one channel, shared by every cell.

    See the module docstring for what each ``BackgroundConfig.mode`` does and
    why ``background_mad`` is always ``None`` in "fixed" mode.
    """
    codes: set[str] = set()

    if background.mode == "fixed":
        value = background.fixed_value_by_channel.get(channel_id)
        if value is None:
            codes.add("missing_fixed_background_value")
            return None, None, codes
        codes.add("mad_undefined_for_fixed_background")
        return float(value), None, codes

    # "outside_cells_median_mad"
    outside = labels == 0
    if not np.any(outside):
        codes.add("background_region_empty")
        return None, None, codes
    vals = channel_data[outside]
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    if mad == 0.0:
        codes.add("zero_background_mad")
    return med, mad, codes


def _shell_masks(
    cell_mask: NDArray[np.bool_],
    spacing_um_zyx: tuple[float, float, float],
    inner_shell_width_um: float,
    outer_shell_width_um: float,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_]]:
    """(inner_shell, core, outer_shell) boolean masks from physical-distance EDT.

    ``sampling=spacing_um_zyx`` is what makes this respect anisotropic voxels
    -- see the module docstring.
    """
    dist_in = distance_transform_edt(cell_mask, sampling=spacing_um_zyx)
    dist_out = distance_transform_edt(~cell_mask, sampling=spacing_um_zyx)
    inner = cell_mask & (dist_in <= inner_shell_width_um)
    core = cell_mask & (dist_in > inner_shell_width_um)
    outer = (~cell_mask) & (dist_out > 0) & (dist_out <= outer_shell_width_um)
    return inner, core, outer


def compute_intensity(
    image: ImageVolume,
    labels: NDArray[np.uint32],
    background: BackgroundConfig,
    inner_shell_width_um: float,
    outer_shell_width_um: float,
) -> dict[int, dict[str, Scalar]]:
    """Per-cell, per-non-ignored-channel intensity features.

    ``labels`` is a raw ``ZYX`` ``uint32`` instance label array (background
    0), matching ``image.shape_zyx``. See the module docstring for the full
    key list, background modes, shell definitions, and QC vocabulary.
    """
    if labels.shape != image.shape_zyx:
        raise ContractViolation(
            f"labels shape {labels.shape} != image shape {image.shape_zyx}"
        )

    spacing = image.geometry.spacing_um_zyx
    voxel_volume = image.geometry.voxel_volume_um3
    mad_multiplier = background.mad_multiplier

    cell_ids = np.unique(labels)
    cell_ids = cell_ids[cell_ids > 0]

    out: dict[int, dict[str, Scalar]] = {int(cid): {} for cid in cell_ids}

    for binding in image.channels:
        if ChannelRole.IGNORE in binding.roles:
            continue
        cid = binding.channel_id
        chan = np.asarray(image.channel(cid), dtype=np.float64)

        bg_value, bg_mad, bg_codes = _channel_background(chan, labels, cid, background)

        for cell_id in cell_ids:
            row = out[int(cell_id)]
            codes = set(bg_codes)
            mask = labels == cell_id
            voxel_count = int(mask.sum())
            values = chan[mask]

            stats = _region_stats(values)
            if voxel_count == 0:
                codes.add("cell_region_empty")

            corrected_mean: float | None = None
            corrected_integrated: float | None = None
            positive_fraction: float | None = None
            i_plus: NDArray[np.float64] | None = None

            if voxel_count > 0 and bg_value is not None:
                i_plus = np.clip(values - bg_value, 0.0, None)
                corrected_mean = float(i_plus.mean())
                corrected_integrated = float(voxel_volume * i_plus.sum())

            if i_plus is not None and bg_mad is not None and bg_mad > 0.0:
                threshold = mad_multiplier * bg_mad
                positive_fraction = float(np.mean(i_plus > threshold))

            inner_mean: float | None = None
            core_mean: float | None = None
            outer_mean: float | None = None
            if voxel_count > 0:
                inner, core, outer = _shell_masks(
                    mask, spacing, inner_shell_width_um, outer_shell_width_um
                )
                if np.any(inner):
                    inner_mean = float(chan[inner].mean())
                else:
                    codes.add("inner_shell_empty")
                if np.any(core):
                    core_mean = float(chan[core].mean())
                else:
                    codes.add("core_empty")
                if np.any(outer):
                    outer_mean = float(chan[outer].mean())
                else:
                    codes.add("outer_shell_empty")

            row.update(
                {
                    f"ch.{cid}.raw_mean": stats["raw_mean"],
                    f"ch.{cid}.raw_std": stats["raw_std"],
                    f"ch.{cid}.raw_min": stats["raw_min"],
                    f"ch.{cid}.raw_max": stats["raw_max"],
                    f"ch.{cid}.raw_p10": stats["raw_p10"],
                    f"ch.{cid}.raw_median": stats["raw_median"],
                    f"ch.{cid}.raw_p90": stats["raw_p90"],
                    f"ch.{cid}.raw_sum": stats["raw_sum"],
                    f"ch.{cid}.background": bg_value,
                    f"ch.{cid}.background_mad": bg_mad,
                    f"ch.{cid}.corrected_mean": corrected_mean,
                    f"ch.{cid}.corrected_integrated_um3": corrected_integrated,
                    f"ch.{cid}.positive_fraction": positive_fraction,
                    f"ch.{cid}.inner_shell_mean": inner_mean,
                    f"ch.{cid}.core_mean": core_mean,
                    f"ch.{cid}.outer_shell_mean": outer_mean,
                    f"ch.{cid}.qc": _qc_string(codes),
                }
            )

    return out
