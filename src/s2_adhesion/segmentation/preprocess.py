"""Turns an :class:`ImageVolume` into cellpose-ready input, and maps cellpose's
output labels back onto the volume's original ZYX grid.

No ML packages are imported here (only numpy/scipy, both core measurement
dependencies) -- this module must import cleanly with no ``cellpose``/``torch``
installed, same as the rest of the segmentation-facing (non-adapter) modules.

Pipeline, per TASK 3:

1. Select the configured channels, by logical id, in the configured order.
2. Percentile-normalise each selected channel independently.
3. Resample X and Y to ``min(dy, dx)`` so the in-plane grid is square; leave Z
   at its acquired spacing. The resulting anisotropy (``dz / min(dy, dx)``,
   unchanged by step 3) is what gets passed to cellpose's ``anisotropy=``.
3b. If ``channel_combination`` is ``"max"`` or ``"sum"``, merge the normalised,
   resampled channels into ONE grayscale channel (see :func:`combine_channels`).
   Two cell populations marked by different channels then both have outlines
   in the single image cellpose is given.
4. (elsewhere: engine.evaluate on the resampled grid)
5. Map labels back to the EXACT original ZYX grid by nearest-neighbour --
   never interpolated, so label ids survive exactly.

All physical measurement downstream happens on the original grid; the
resampled/model grid never escapes this module.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from ..config import ChannelCombination, DirectInstanceConfig, NormalizationConfig
from ..contracts import ImageVolume
from ..errors import ContractViolation
from .protocol import PreparedCellposeInput


def _normalize_percentile(
    channel: NDArray[np.generic], config: NormalizationConfig
) -> NDArray[np.float32]:
    """Percentile-normalise one channel to roughly [0, 1].

    A degenerate channel (upper percentile <= lower percentile, e.g. a flat
    background-only crop) maps to all zeros rather than dividing by zero.
    """
    data = np.asarray(channel, dtype=np.float64)
    lo = float(np.percentile(data, config.lower_percentile))
    hi = float(np.percentile(data, config.upper_percentile))
    if hi <= lo:
        return np.zeros_like(data, dtype=np.float32)
    normalized = (data - lo) / (hi - lo)
    if config.clip:
        normalized = np.clip(normalized, 0.0, 1.0)
    return normalized.astype(np.float32)


def _target_inplane_spacing_um(
    spacing_um_zyx: tuple[float, float, float],
) -> tuple[float, float, float]:
    dz, dy, dx = spacing_um_zyx
    target = min(dy, dx)
    return (dz, target, target)


def _resample_inplane(
    channel: NDArray[np.floating], spacing_um_zyx: tuple[float, float, float]
) -> NDArray[np.float32]:
    """Resample one ZYX channel so its in-plane pixel spacing is square.

    Z is left untouched (zoom factor 1). Uses linear interpolation -- this is
    intensity data feeding a model, not the labels that must survive
    nearest-neighbour round-tripping later.
    """
    dz, dy, dx = spacing_um_zyx
    target = min(dy, dx)
    scale_y = dy / target
    scale_x = dx / target
    if scale_y == 1.0 and scale_x == 1.0:
        return channel.astype(np.float32, copy=False)
    resampled = ndimage.zoom(channel, zoom=(1.0, scale_y, scale_x), order=1)
    return resampled.astype(np.float32)


def combined_channel_id(
    combination: ChannelCombination, channel_ids: tuple[str, ...]
) -> str:
    """Synthetic channel id for a merged image, e.g. ``"max(green,red)"``.

    It records *which* channels went in and *how*, so provenance and
    diagnostics that only see ``PreparedCellposeInput.channel_ids`` still say
    what cellpose was fed. It is never looked up as a real channel.
    """
    return f"{combination}({','.join(channel_ids)})"


def combine_channels(
    channels: NDArray[np.floating],
    combination: ChannelCombination,
    norm_config: NormalizationConfig,
) -> NDArray[np.float32]:
    """Merge already-normalised ``(C, Z, Y, X)`` channels into ``(1, Z, Y, X)``.

    ``"max"`` takes the voxel-wise maximum: a cell bright in any one channel
    keeps its full normalised intensity, and nothing is added where a channel
    is dark. ``"sum"`` adds the channels and percentile-normalises the result
    again with the same settings, so its range is comparable to a single
    normalised channel. ``"stack"`` is not a merge and is rejected here.

    Every input channel must already be normalised on its own (that is what
    makes a dim channel count as much as a bright one) -- this function does
    not normalise its inputs.
    """
    if channels.ndim != 4:
        raise ContractViolation(
            f"combine_channels expects (C, Z, Y, X), got shape {channels.shape}"
        )
    if combination == "max":
        merged = np.max(channels, axis=0)
    elif combination == "sum":
        merged = _normalize_percentile(np.sum(channels, axis=0), norm_config)
    else:
        raise ContractViolation(
            f"channel_combination {combination!r} is not a merge; "
            "expected 'max' or 'sum'"
        )
    return merged.astype(np.float32, copy=False)[np.newaxis, ...]


def prepare_cellpose_input(
    image: ImageVolume, direct_config: DirectInstanceConfig
) -> PreparedCellposeInput:
    """Select, normalise, resample and (optionally) merge ``image`` for cellpose."""
    channel_ids = direct_config.input_channel_ids
    norm_config = direct_config.model.normalization
    combination = direct_config.channel_combination

    resampled_channels = [
        _resample_inplane(
            _normalize_percentile(image.channel(cid), norm_config),
            image.geometry.spacing_um_zyx,
        )
        for cid in channel_ids
    ]
    stacked = np.stack(resampled_channels, axis=0)
    if combination != "stack":
        stacked = combine_channels(stacked, combination, norm_config)
        channel_ids = (combined_channel_id(combination, channel_ids),)

    resampled_spacing = _target_inplane_spacing_um(image.geometry.spacing_um_zyx)
    anisotropy = image.geometry.anisotropy_z_to_xy

    diameter_um = direct_config.model.diameter_um
    diameter_px = diameter_um / resampled_spacing[1] if diameter_um is not None else None

    return PreparedCellposeInput(
        data=stacked,
        channel_ids=channel_ids,
        anisotropy=anisotropy,
        original_shape_zyx=image.shape_zyx,
        resampled_spacing_um_zyx=resampled_spacing,
        diameter_px=diameter_px,
    )


def _nearest_indices(n_target: int, n_source: int) -> NDArray[np.intp]:
    """For each of ``n_target`` output positions, the nearest ``n_source`` index.

    Voxel-centre nearest neighbour: output position ``j`` maps to source
    index ``round((j + 0.5) * n_source / n_target - 0.5)``, clipped to range.
    Used both directions (up- and down-sampling); resampling up then back
    down by the same ratio recovers the original index sequence exactly.
    """
    if n_source == n_target:
        return np.arange(n_target)
    positions = (np.arange(n_target) + 0.5) * (n_source / n_target) - 0.5
    idx = np.rint(positions).astype(np.intp)
    return np.clip(idx, 0, n_source - 1)


def map_labels_to_original_grid(
    labels: NDArray[np.integer], prepared: PreparedCellposeInput
) -> NDArray[np.uint32]:
    """Map model-grid labels back onto ``prepared.original_shape_zyx``.

    Nearest-neighbour only (a plain fancy-index gather) -- interpolating
    integer label ids would invent boundary values that are not real
    instance ids. Z is never resampled, so a mismatched Z extent is a
    contract violation rather than something to silently reshape around.
    """
    oz, oy, ox = prepared.original_shape_zyx
    if labels.ndim != 3:
        raise ContractViolation(
            f"cellpose labels must be 3-D ZYX, got shape {labels.shape}"
        )
    rz, ry, rx = labels.shape
    if rz != oz:
        raise ContractViolation(
            "cellpose engine returned a different Z extent than it was given "
            f"(original Z={oz}, returned Z={rz}); Z is never resampled, so "
            "this means the engine dropped or added planes."
        )
    y_idx = _nearest_indices(oy, ry)
    x_idx = _nearest_indices(ox, rx)
    mapped = labels[:, y_idx, :][:, :, x_idx]
    return mapped.astype(np.uint32)
