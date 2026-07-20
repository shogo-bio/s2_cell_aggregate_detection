"""Detector clipping and dynamic-range checks, per cell and per channel.

Real acquisitions here clip: FITC saturates (~13% of bright voxels at the 12-bit
max), mCherry is under-exposed (~22% of range). Both silently corrupt every
intensity-derived metric, so each cell/channel carries an explicit saturation
account rather than the code pretending the intensities are clean.

Design follows a cross-model review; the load-bearing decisions:

* Saturation is measured on RAW integer voxels, before any background
  subtraction / normalisation / filtering -- those move values off the exact
  ceiling and hide the clipping.
* Per region (whole cell, membrane shell, core), because a clipped SHELL and a
  clipped CORE mean different things: a clipped shell only biases the
  ring/interior score downward (still a valid lower bound), while a clipped core
  turns real interior signal into a false "uniform" reading and must null the
  score.
* A tiered status (none / trace / material / severe), not one boolean, because
  "one hot pixel" and "the whole cell is a flat-topped plateau" are not the same
  problem.
* Saturated cells are FLAGGED, never dropped. The brightest cells are the
  highest-expressing ones; silently excluding them turns a detector limit into
  an expression-dependent selection bias.

This module only DETECTS and CLASSIFIES. Which metric each status nulls is
applied where that metric is computed (radial, localization), keyed off the
status this module assigns.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from ..contracts import ChannelBinding, ChannelRole, Scalar, VoxelGeometry

# Status tiers, ordered. `SEVERE` means intensity-derived localisation is not
# trustworthy for this cell/channel.
NONE = "none"
TRACE = "trace"
MATERIAL = "material"
SEVERE = "severe"

# Saturation topology within the cell.
PATTERN_NONE = "none"
PATTERN_SHELL_ONLY = "shell_only"
PATTERN_CORE = "core_present"
PATTERN_SHELL_AND_CORE = "shell_and_core"

# Common detector ceilings (2^n - 1). Used only to sanity-check / auto-suggest.
_KNOWN_CEILINGS = (255, 1023, 4095, 16383, 65535)


def resolve_detector_max(
    data: NDArray[np.integer], configured: int | None
) -> tuple[int, str]:
    """Return (detector_max, provenance).

    Metadata/config first; auto-detection is only a fallback and only a
    suggestion. A configured value that the data EXCEEDS is a hard error -- it
    means the config is wrong about the detector, and every saturation number
    downstream would be silently miscounted.
    """
    observed_max = int(np.max(data)) if data.size else 0

    if configured is not None:
        if observed_max > configured:
            raise ValueError(
                f"configured detector_max={configured} but the data contains "
                f"{observed_max}; the detector ceiling is set wrong."
            )
        return configured, "config"

    # No config: auto-suggest. Prefer a known ceiling the data actually reaches.
    for ceiling in _KNOWN_CEILINGS:
        if observed_max == ceiling:
            return ceiling, "auto_detected_at_known_ceiling"
    # Data never hits a standard ceiling -- either genuinely unsaturated or a
    # non-standard detector. Fall back to the observed max but flag low
    # confidence; nothing will be counted as saturated unless it equals this.
    return observed_max, "auto_observed_max_uncertain"


def _status(fraction_cell: float, fraction_shell: float) -> str:
    """Tiered severity from the cell and shell saturated fractions."""
    if fraction_cell <= 0.0 and fraction_shell <= 0.0:
        return NONE
    if fraction_cell >= 0.01 or fraction_shell >= 0.02:
        return SEVERE
    if fraction_cell >= 0.001 or fraction_shell >= 0.005:
        return MATERIAL
    return TRACE


def _pattern(shell_sat: bool, core_sat: bool) -> str:
    if shell_sat and core_sat:
        return PATTERN_SHELL_AND_CORE
    if core_sat:
        return PATTERN_CORE
    if shell_sat:
        return PATTERN_SHELL_ONLY
    return PATTERN_NONE


def channel_dynamic_range(
    data: NDArray[np.integer], labels: NDArray[np.uint32], detector_max: int
) -> dict[str, Scalar]:
    """Acquisition-level range check for one channel.

    ``range_occupancy`` = (foreground p99.9 - background median) /
    (detector_max - background median). Below ~0.25 the channel is close to the
    noise floor. This flags the ACQUISITION; it does not by itself invalidate a
    cell (that needs a per-cell contrast check), so it is reported, not enforced.
    """
    outside = labels == 0
    bg = data[outside]
    bg_median = float(np.median(bg)) if bg.size else 0.0
    fg_p999 = float(np.percentile(data, 99.9))
    denom = detector_max - bg_median
    occupancy = (fg_p999 - bg_median) / denom if denom > 0 else 0.0
    return {
        "background_median": bg_median,
        "foreground_p99_9": fg_p999,
        "range_occupancy": round(occupancy, 4),
    }


def compute_saturation(
    labels: NDArray[np.uint32],
    channels: Mapping[str, NDArray[np.integer]],
    channel_bindings: Sequence[ChannelBinding],
    geometry: VoxelGeometry,
    detector_max: int,
    *,
    inner_shell_width_um: float = 1.0,
) -> dict[int, dict[str, Scalar]]:
    """Per-cell, per-channel saturation account, keyed ``ch.<id>.saturation_*``.

    ``channels`` must be RAW integer intensities (pre background-subtraction).
    The shell/core split reuses a physical distance from the boundary so it is
    anisotropy-aware, matching the localisation metrics.
    """
    from scipy import ndimage as ndi

    role_ok = {
        b.channel_id
        for b in channel_bindings
        if ChannelRole.IGNORE not in b.roles
    }
    active = {cid: data for cid, data in channels.items() if cid in role_ok}

    out: dict[int, dict[str, Scalar]] = {}
    # Bounding box per label, so the (expensive) distance transform runs on a
    # small crop rather than the whole volume once per cell -- the difference
    # between seconds and many minutes on a real 512x512 field.
    slices = ndi.find_objects(labels)

    for label_index, sl in enumerate(slices):
        if sl is None:
            continue
        cell_id = label_index + 1
        sub_labels = labels[sl]
        mask = sub_labels == cell_id
        if not mask.any():
            continue

        depth = ndi.distance_transform_edt(mask, sampling=geometry.spacing_um_zyx)
        shell = mask & (depth <= inner_shell_width_um)
        core = mask & (depth > inner_shell_width_um)
        n_cell = int(mask.sum())
        n_shell = int(shell.sum())
        n_core = int(core.sum())

        row: dict[str, Scalar] = {}
        for cid, data in active.items():
            at_max = data[sl] == detector_max

            f_cell = float((at_max & mask).sum() / n_cell) if n_cell else 0.0
            f_shell = float((at_max & shell).sum() / n_shell) if n_shell else 0.0
            f_core = float((at_max & core).sum() / n_core) if n_core else 0.0

            shell_sat = f_shell >= 0.005
            core_sat = f_core >= 0.005
            status = _status(f_cell, f_shell)

            row[f"ch.{cid}.saturated_fraction_cell"] = round(f_cell, 5)
            row[f"ch.{cid}.saturated_fraction_shell"] = round(f_shell, 5)
            row[f"ch.{cid}.saturated_fraction_core"] = round(f_core, 5)
            row[f"ch.{cid}.saturation_pattern"] = _pattern(shell_sat, core_sat)
            row[f"ch.{cid}.saturation_status"] = status
            # A convenience flag for downstream: raw intensity aggregates
            # (mean/sum/integrated) are censored lower bounds once material.
            row[f"ch.{cid}.intensity_is_lower_bound"] = status in (MATERIAL, SEVERE)
        out[int(cell_id)] = row

    return out


# Three-way reliability of the ring/interior readout under saturation.
LOC_QUANTITATIVE = "quantitative"  # value is a valid quantitative measurement
LOC_LOWER_BOUND = "lower_bound"    # shell clipped: score is a downward-biased bound
LOC_INDETERMINATE = "indeterminate"  # core clipped: cannot tell interior from uniform


def localization_reliability(saturation_row: Mapping[str, Scalar], channel_id: str) -> str:
    """How far the ring/interior score can be trusted for one channel.

    * core clipped (a clipped core masquerades as uniform signal) -> indeterminate
    * shell clipped, core clean -> the score is a valid downward-biased LOWER
      bound; a conservative "shell-enriched" call still holds if it clears a
      threshold, but the magnitude is not quantitative
    * otherwise -> quantitative
    """
    pattern = saturation_row.get(f"ch.{channel_id}.saturation_pattern", PATTERN_NONE)
    status = saturation_row.get(f"ch.{channel_id}.saturation_status", NONE)
    if pattern in (PATTERN_CORE, PATTERN_SHELL_AND_CORE):
        return LOC_INDETERMINATE
    if pattern == PATTERN_SHELL_ONLY and status in (MATERIAL, SEVERE):
        return LOC_LOWER_BOUND
    return LOC_QUANTITATIVE


def colocalization_trustworthy(
    saturation_row: Mapping[str, Scalar], channel_a: str, channel_b: str
) -> bool:
    """Pearson/Manders need both channels free of material saturation.

    Clipping censors the linear relationship Pearson assumes and distorts
    Manders' intensity sums, so any material saturation in either channel nulls
    the pair.
    """
    for cid in (channel_a, channel_b):
        status = saturation_row.get(f"ch.{cid}.saturation_status", NONE)
        if status in (MATERIAL, SEVERE):
            return False
    return True
