"""Adapter for cellpose 3.x (``cyto3``, a 26 MB U-Net).

This is one of exactly two modules in this package permitted to import
``cellpose``/``torch`` (the other is ``cellpose_v4.py``). It is only ever
imported by ``factory.create_cellpose_engine`` after confirming the installed
cellpose major version is 3 -- never at import time of any other module in
this package.

cellpose 3's ``models.Cellpose.eval`` genuinely differs from cellpose 4's
``models.CellposeModel.eval``: 3D volumes are passed with the channel axis
last (or omitted for single-channel), channel selection uses the
``channels=[cytoplasm, nucleus]`` 1-based-index convention, and no
``z_axis=`` argument exists or is needed. Every ``CellposeEvalConfig`` field
is forwarded by name (never through a generic ``**kwargs`` filter): if a
field genuinely has no cellpose-3 equivalent, that must be a deliberate,
documented choice here, not a silent drop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from cellpose import models

from ..config import CellposeModelConfig
from ..errors import SegmentationError
from .protocol import CellposeRawResult, PreparedCellposeInput


def _v3_channels(channel_ids: tuple[str, ...]) -> list[int]:
    """cellpose-3 channel convention: ``[cytoplasm_channel, nucleus_channel]``.

    0 means "not present". With one selected channel we treat it as a plain
    grayscale cytoplasm/membrane channel (``[0, 0]``); with two, the first
    selected channel is cytoplasm (index 1 within the array handed to
    ``eval``) and the second is the nuclear channel (index 2). More than two
    is already rejected by ``validate_config`` for ``package_major == 3``, so
    reaching this function with more is a bug upstream, not user input.
    """
    n = len(channel_ids)
    if n == 1:
        return [0, 0]
    if n == 2:
        return [1, 2]
    raise SegmentationError(
        f"cellpose v3 accepts 1 or 2 input channels, got {n}: {channel_ids}"
    )


def _stack_for_v3(data: np.ndarray) -> np.ndarray:
    """``(C, Z, Y, X)`` -> what cellpose-3's 3-D eval expects.

    Single channel: plain ``(Z, Y, X)``. Two channels: ``(Z, Y, X, C)``, the
    layout cellpose-3 expects so ``channels=[1, 2]`` indexes the last axis.
    """
    if data.shape[0] == 1:
        return data[0]
    return np.moveaxis(data, 0, -1)


@dataclass(frozen=True, slots=True)
class CellposeV3Engine:
    """``CellposeEngine`` backed by a real ``cellpose.models.Cellpose``."""

    model_name: str
    package_version: str
    device: str
    model_checksum: str | None
    _model: Any = field(repr=False)
    package_major: Literal[3] = 3

    def evaluate(
        self, prepared: PreparedCellposeInput, *, config: CellposeModelConfig
    ) -> CellposeRawResult:
        eval_cfg = config.eval
        channels = _v3_channels(prepared.channel_ids)
        volume = _stack_for_v3(prepared.data)

        # Optional tiling arguments are omitted rather than passed as None, so
        # cellpose applies its own defaults.
        optional: dict[str, Any] = {}
        if eval_cfg.bsize is not None:
            optional["bsize"] = eval_cfg.bsize
        if eval_cfg.tile_overlap is not None:
            optional["tile_overlap"] = eval_cfg.tile_overlap

        # 2.5D stitching vs native 3D. Stitching segments each plane in 2D and
        # joins planes by IoU -- better on strongly anisotropic stacks, where a
        # native 3D flow must resolve shape along the coarse Z axis.
        if eval_cfg.stitch_threshold is not None:
            optional["do_3D"] = False
            optional["stitch_threshold"] = eval_cfg.stitch_threshold
        else:
            optional["do_3D"] = True
            optional["anisotropy"] = prepared.anisotropy

        masks, flows, styles, diams = self._model.eval(
            volume,
            channels=channels,
            z_axis=0,
            diameter=prepared.diameter_px,
            flow_threshold=eval_cfg.flow_threshold,
            cellprob_threshold=eval_cfg.cellprob_threshold,
            augment=eval_cfg.augment,
            batch_size=eval_cfg.batch_size,
            **optional,
        )

        diameter_value = diams if isinstance(diams, (int, float)) else None
        return CellposeRawResult(
            labels=np.asarray(masks, dtype=np.uint32),
            diameters_px=float(diameter_value) if diameter_value is not None else None,
            extra={"raw_diameter": repr(diams)},
        )


def build_engine(
    config: CellposeModelConfig, *, package_version: str, model_checksum: str | None
) -> CellposeV3Engine:
    gpu = config.device in ("cuda", "auto")
    model_type = (
        str(config.pretrained_model_path)
        if config.pretrained_model_path is not None
        else config.model_name
    )
    model = models.Cellpose(gpu=gpu, model_type=model_type)
    return CellposeV3Engine(
        model_name=config.model_name,
        package_version=package_version,
        device=config.device,
        model_checksum=model_checksum,
        _model=model,
    )
