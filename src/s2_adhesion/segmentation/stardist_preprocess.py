"""Prepare a volume for StarDist 3D and map its labels back afterwards.

Kept separate from ``preprocess.py`` (which serves cellpose) because the two
libraries need genuinely different treatment. Cellpose accepts ``anisotropy`` at
predict time, so it only needs square in-plane pixels. StarDist bakes anisotropy
into the trained model, so a volume must be resampled to whatever sampling the
model was trained at -- usually isotropic -- and the labels resampled home.

Everything physical is measured on the ORIGINAL grid. The model grid exists only
long enough to run inference on.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage as ndi

from ..config import DirectStarDistConfig
from ..contracts import ImageVolume
from ..errors import SegmentationError
from .protocol import PreparedStarDistInput


def normalise_percentile(
    volume: NDArray[np.floating], lower: float, upper: float, clip: bool = True
) -> NDArray[np.float32]:
    """Percentile normalisation, the convention StarDist's own examples use.

    Percentiles rather than min/max because a single hot pixel or a saturated
    speck would otherwise compress the whole dynamic range.
    """
    lo = float(np.percentile(volume, lower))
    hi = float(np.percentile(volume, upper))
    if hi <= lo:
        return np.zeros(volume.shape, dtype=np.float32)
    out = (volume.astype(np.float32) - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0) if clip else out


def prepare_stardist_input(
    image: ImageVolume, config: DirectStarDistConfig
) -> PreparedStarDistInput:
    """Select the configured channel, normalise, and resample if asked.

    Resampling to isotropic uses linear interpolation on intensities. That is
    appropriate here (unlike for labels, where it would invent values between
    ids) because the model consumes a continuous image.
    """
    if len(config.input_channel_ids) != 1:
        raise SegmentationError(
            "StarDist 3D predicts from a single intensity volume, but "
            f"{len(config.input_channel_ids)} channels were configured: "
            f"{config.input_channel_ids}"
        )
    channel_id = config.input_channel_ids[0]
    raw = np.asarray(image.channel(channel_id), dtype=np.float32)

    norm = config.model.normalization
    volume = normalise_percentile(
        raw, norm.lower_percentile, norm.upper_percentile, norm.clip
    )

    spacing = image.geometry.spacing_um_zyx
    original_shape = image.shape_zyx

    if not config.model.resample_isotropic or image.geometry.is_isotropic:
        return PreparedStarDistInput(
            data=volume,
            channel_id=channel_id,
            spacing_um_zyx=spacing,
            original_shape_zyx=original_shape,
            was_resampled=False,
        )

    target = min(spacing)
    zoom = tuple(s / target for s in spacing)
    resampled = ndi.zoom(volume, zoom, order=1, mode="nearest")

    return PreparedStarDistInput(
        data=resampled.astype(np.float32),
        channel_id=channel_id,
        spacing_um_zyx=(target, target, target),
        original_shape_zyx=original_shape,
        was_resampled=True,
    )


def map_labels_to_original_grid(
    labels: NDArray[np.integer], prepared: PreparedStarDistInput
) -> NDArray[np.uint32]:
    """Bring model-grid labels back to the acquired grid.

    Nearest-neighbour, always: interpolating label ids would synthesise ids that
    were never predicted and silently merge or invent instances. The output is
    forced to the original shape exactly, because every physical measurement
    downstream assumes labels and image share a grid.
    """
    labels = np.asarray(labels)
    target_shape = prepared.original_shape_zyx

    if labels.shape == tuple(target_shape):
        return labels.astype(np.uint32)

    # Index-gather rather than ndi.zoom: exact output shape, no interpolation,
    # and no dependence on zoom's rounding of the output extent.
    indices = np.meshgrid(
        *[
            np.clip(
                np.rint(np.arange(t) * (s / t)).astype(int), 0, s - 1
            )
            for t, s in zip(target_shape, labels.shape)
        ],
        indexing="ij",
    )
    return labels[tuple(indices)].astype(np.uint32)
