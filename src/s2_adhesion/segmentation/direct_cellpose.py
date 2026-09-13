"""``direct_cellpose`` strategy: one cellpose model, no nuclear seeding.

``DirectCellposeBackend`` receives its :class:`~.protocol.CellposeEngine` by
dependency injection (a constructor argument) rather than building one
itself via ``factory.create_cellpose_engine``. That keeps this module fully
unit-testable with a fake engine and no ML package installed -- wiring a real
engine (which does need cellpose installed) is the caller's job, one layer up.
"""

from __future__ import annotations

import hashlib
import platform
import time
from dataclasses import asdict, dataclass, is_dataclass

import numpy as np

from ..config import DirectInstanceConfig
from ..contracts import LabelVolume, SegmentationProvenance
from .postprocess import (
    fill_internal_holes,
    filter_by_physical_volume,
    filter_by_z_extent,
    split_disconnected_labels,
)
from .preprocess import map_labels_to_original_grid, prepare_cellpose_input
from .protocol import (
    CellposeEngine,
    SegmentationDiagnostics,
    SegmentationRequest,
    SegmentationResult,
)

_BACKEND_ID = "direct_cellpose"


def _count_instances(labels: np.ndarray) -> int:
    ids = np.unique(labels)
    return int((ids > 0).sum())


def _config_sha256(config: DirectInstanceConfig) -> str:
    """Deterministic hash of a frozen config for provenance.

    ``str(asdict(...))`` is stable across calls for the same values (dict
    insertion order follows field declaration order, which is fixed) and
    good enough to detect "labels came from a different config" -- it is not
    meant to be a canonical serialisation format.
    """

    def _to_plain(value: object) -> object:
        if is_dataclass(value) and not isinstance(value, type):
            return {f: _to_plain(v) for f, v in asdict(value).items()}  # type: ignore[arg-type]
        if isinstance(value, (list, tuple)):
            return [_to_plain(v) for v in value]
        if isinstance(value, dict):
            return {k: _to_plain(v) for k, v in value.items()}
        return repr(value)

    payload = repr(_to_plain(config)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class DirectCellposeBackend:
    """Segments cells directly with one injected cellpose engine.

    ``segment`` never leaves a partial result behind: it only constructs a
    ``LabelVolume``/``SegmentationResult`` after ``engine.evaluate`` returns
    successfully. If the engine raises, the exception propagates unchanged
    and nothing downstream is touched.
    """

    config: DirectInstanceConfig
    engine: CellposeEngine
    backend_id: str = _BACKEND_ID

    def segment(self, request: SegmentationRequest) -> SegmentationResult:
        image = request.image

        t0 = time.perf_counter()
        prepared = prepare_cellpose_input(image, self.config)
        prep_seconds = time.perf_counter() - t0

        t1 = time.perf_counter()
        raw = self.engine.evaluate(prepared, config=self.config.model)
        eval_seconds = time.perf_counter() - t1

        cells = map_labels_to_original_grid(raw.labels, prepared)

        # Shared post-processing in physical units (same order as the StarDist
        # strategy). Until 2026-09-13 this strategy silently ignored these
        # config fields: min_cell_volume_um3 was never applied, so 142 of
        # 2408 objects in the 25-field run were below the configured 50 um^3.
        n_raw = _count_instances(cells)
        if self.config.split_disconnected_labels:
            cells = split_disconnected_labels(cells)
        if self.config.fill_internal_holes:
            cells = fill_internal_holes(cells)
        n_split = _count_instances(cells)
        cells = filter_by_physical_volume(
            cells, image.geometry.spacing_um_zyx, self.config.min_cell_volume_um3
        )
        n_after_volume = _count_instances(cells)
        if self.config.min_z_extent_planes is not None:
            cells = filter_by_z_extent(cells, self.config.min_z_extent_planes)
        n_final = _count_instances(cells)
        warnings: list[str] = []
        if n_split - n_after_volume:
            warnings.append(
                f"cellpose_small_objects_removed: {n_split - n_after_volume} instance(s) "
                f"below {self.config.min_cell_volume_um3} um^3"
            )
        if n_after_volume - n_final:
            warnings.append(
                f"cellpose_thin_objects_removed: {n_after_volume - n_final} instance(s) "
                f"thinner than {self.config.min_z_extent_planes} Z planes (not on the Z border)"
            )

        provenance = SegmentationProvenance(
            run_id=request.run_id,
            backend_id=self.backend_id,
            strategy=_BACKEND_ID,
            config_sha256=_config_sha256(self.config),
            input_image_sha256=image.identity.image_content_sha256,
            device=self.engine.device,
            host_platform=platform.platform(),
            package_name="cellpose",
            package_version=self.engine.package_version,
            model_name=self.engine.model_name,
            model_sha256=self.engine.model_checksum,
        )

        labels = LabelVolume(
            cells=cells,
            geometry=image.geometry,
            identity=image.identity,
            provenance=provenance,
        )

        diagnostics = SegmentationDiagnostics(
            warnings=tuple(warnings),
            timing_seconds={"preprocess": prep_seconds, "evaluate": eval_seconds},
            extra={
                "n_input_channels": len(prepared.channel_ids),
                "input_channel_ids": ",".join(prepared.channel_ids),
                "channel_combination": self.config.channel_combination,
                "anisotropy": prepared.anisotropy,
                "n_instances_raw": n_raw,
                "n_instances_after_split": n_split,
                "n_instances_after_volume_filter": n_after_volume,
                "n_instances_final": n_final,
            },
        )
        return SegmentationResult(labels=labels, diagnostics=diagnostics)
