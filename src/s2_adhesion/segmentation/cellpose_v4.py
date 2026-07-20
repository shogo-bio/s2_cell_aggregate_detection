"""Adapter for cellpose 4.x (``cpsam``, a 1.23 GB SAM/transformer model).

This is one of exactly two modules in this package permitted to import
``cellpose``/``torch`` (the other is ``cellpose_v3.py``). It is only ever
imported by ``factory.create_cellpose_engine`` after confirming the installed
cellpose major version is 4 -- never at import time of any other module in
this package.

cellpose 4's ``models.CellposeModel.eval`` genuinely differs from cellpose
3's ``models.Cellpose.eval``:

* 3-D input REQUIRES an explicit ``z_axis=`` (raises ``ValueError`` without
  it); cellpose 3 has no such argument.
* There is no ``channels=[cytoplasm, nucleus]`` 1-based convention -- cpsam
  auto-detects channel content. Multi-channel volumes are passed with
  ``channel_axis=`` instead.
* There is no boolean ``tile=`` switch. Neither 3.1.1.3 nor 4.2.1.1 has one --
  tiling is unconditional and is tuned through ``bsize``/``tile_overlap``.
  ``CellposeEvalConfig`` therefore exposes those two directly.

Every other ``CellposeEvalConfig`` field maps directly onto a real ``eval``
keyword and is forwarded by name, never through a generic ``**kwargs``
filter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from cellpose import models

from ..config import CellposeModelConfig
from ..errors import SegmentationError
from .protocol import CellposeRawResult, PreparedCellposeInput


def _stack_for_v4(data: np.ndarray) -> tuple[np.ndarray, int | None]:
    """``(C, Z, Y, X)`` -> ``(Z, Y, X[, C])`` plus the ``channel_axis`` to pass.

    Single channel: plain ``(Z, Y, X)``, ``channel_axis=None``. Multiple
    channels: ``(Z, Y, X, C)`` with ``channel_axis=-1``, matching ``z_axis=0``
    still referring to the (unchanged) first axis.
    """
    if data.shape[0] == 1:
        return data[0], None
    return np.moveaxis(data, 0, -1), -1


@dataclass(frozen=True, slots=True)
class CellposeV4Engine:
    """``CellposeEngine`` backed by a real ``cellpose.models.CellposeModel``."""

    model_name: str
    package_version: str
    device: str
    model_checksum: str | None
    _model: Any = field(repr=False)
    package_major: Literal[4] = 4

    def evaluate(
        self, prepared: PreparedCellposeInput, *, config: CellposeModelConfig
    ) -> CellposeRawResult:
        eval_cfg = config.eval

        volume, channel_axis = _stack_for_v4(prepared.data)

        # Optional tiling arguments are omitted rather than passed as None, so
        # cellpose applies its own defaults.
        optional: dict[str, Any] = {}
        if eval_cfg.bsize is not None:
            optional["bsize"] = eval_cfg.bsize
        if eval_cfg.tile_overlap is not None:
            optional["tile_overlap"] = eval_cfg.tile_overlap

        masks, flows, styles = self._model.eval(
            volume,
            z_axis=0,
            channel_axis=channel_axis,
            do_3D=True,
            anisotropy=prepared.anisotropy,
            diameter=prepared.diameter_px,
            flow_threshold=eval_cfg.flow_threshold,
            cellprob_threshold=eval_cfg.cellprob_threshold,
            augment=eval_cfg.augment,
            batch_size=eval_cfg.batch_size,
            **optional,
        )

        return CellposeRawResult(
            labels=np.asarray(masks, dtype=np.uint32),
            diameters_px=None,
            extra={},
        )


def build_engine(
    config: CellposeModelConfig, *, package_version: str, model_checksum: str | None
) -> CellposeV4Engine:
    gpu = config.device in ("cuda", "auto")
    if config.pretrained_model_path is not None:
        model = models.CellposeModel(gpu=gpu, pretrained_model=str(config.pretrained_model_path))
    else:
        model = models.CellposeModel(gpu=gpu)
    return CellposeV4Engine(
        model_name=config.model_name,
        package_version=package_version,
        device=config.device,
        model_checksum=model_checksum,
        _model=model,
    )
