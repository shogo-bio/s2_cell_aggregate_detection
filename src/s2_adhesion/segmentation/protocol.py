"""Shared types for the segmentation stage.

This module is pure data + structural typing (``typing.Protocol``) and MUST
import cleanly with no ML packages installed -- it is imported by the
measurement/orchestration layer even on machines that never run a model, and
by every unit test that exercises segmentation logic against a fake engine.

Only ``dataclasses``, ``typing``, ``numpy`` and sibling modules in this
package that make the same promise (``..contracts``, ``..config``) may be
imported here. Never import ``cellpose`` or ``torch`` from this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from ..config import CellposeModelConfig
from ..contracts import ImageVolume, LabelVolume, Scalar

# ─── Backend-facing request/result types ──────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SegmentationRequest:
    """What any :class:`SegmentationBackend` needs to produce labels for one field.

    Strategy-specific parameters (channel ids, model config, thresholds, ...)
    live on the backend instance itself, not here -- a backend is constructed
    once per configured strategy and then handed one request per field. This
    keeps the request identical across strategies (``direct_cellpose``,
    ``nucleus_seeded_watershed``, ...).
    """

    image: ImageVolume
    run_id: str


@dataclass(frozen=True, slots=True)
class SegmentationDiagnostics:
    """Non-canonical information about how a segmentation run went.

    Nothing here is required for correctness of the returned labels; it exists
    for debugging and QC dashboards. ``warnings`` are human-readable strings,
    never exceptions -- a warning means "the result is present but might be
    suspect", not "the result is missing".
    """

    warnings: tuple[str, ...] = ()
    timing_seconds: Mapping[str, float] = field(default_factory=dict)
    extra: Mapping[str, Scalar] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SegmentationResult:
    """What a :class:`SegmentationBackend` hands back for one field."""

    labels: LabelVolume
    diagnostics: SegmentationDiagnostics


@runtime_checkable
class SegmentationBackend(Protocol):
    """Anything that turns one field's image into instance labels.

    Implementations: ``DirectCellposeBackend`` (this package) and a
    nucleus-seeded watershed backend owned elsewhere. Both are constructed
    once from a validated ``SegmentationConfig`` and then called per field.
    """

    def segment(self, request: SegmentationRequest) -> SegmentationResult: ...


# ─── Cellpose-engine-facing types ──────────────────────────────────────────────
#
# These sit between preprocessing (grid resampling, normalisation) and the
# version-specific cellpose adapters. Everything on these types lives on the
# MODEL grid (square in-plane pixels); only ``original_shape_zyx`` reaches back
# to the physical grid so labels can be mapped home after inference.


@dataclass(frozen=True, slots=True)
class PreparedCellposeInput:
    """Model-ready input, plus everything needed to map results back home.

    ``data`` is ``(n_channels, Z, Y', X')``, float32, normalised per channel,
    with Y' and X' resampled so the in-plane pixel is square (spacing
    ``min(dy, dx)`` of the original geometry). Z is never resampled.
    """

    data: NDArray[np.floating]
    channel_ids: tuple[str, ...]
    anisotropy: float
    original_shape_zyx: tuple[int, int, int]
    resampled_spacing_um_zyx: tuple[float, float, float]
    diameter_px: float | None

    def __post_init__(self) -> None:
        if self.data.ndim != 4:
            raise ValueError(
                f"PreparedCellposeInput.data must be (C, Z, Y, X), got shape "
                f"{self.data.shape}"
            )
        if self.data.shape[0] != len(self.channel_ids):
            raise ValueError(
                f"data has {self.data.shape[0]} channels but channel_ids has "
                f"{len(self.channel_ids)}: {self.channel_ids}"
            )


@dataclass(frozen=True, slots=True)
class CellposeRawResult:
    """Raw output of one cellpose ``eval`` call, still on the model grid."""

    labels: NDArray[np.integer]
    diameters_px: float | None = None
    extra: Mapping[str, Scalar] = field(default_factory=dict)


@runtime_checkable
class CellposeEngine(Protocol):
    """One concrete, version-pinned way to run cellpose inference.

    Implemented by ``CellposeV3Engine`` and ``CellposeV4Engine`` (each in
    their own module, the only two modules in this package allowed to import
    ``cellpose``/``torch``). Built exclusively by
    ``factory.create_cellpose_engine`` so nothing outside that factory ever
    imports the wrong major version.

    The provenance-shaped attributes (``package_version``, ``device``,
    ``model_checksum``) are populated at construction time so
    ``DirectCellposeBackend`` can build a ``SegmentationProvenance`` without
    reaching back into cellpose or the filesystem.
    """

    package_major: Literal[3, 4]
    model_name: str
    package_version: str
    device: str
    model_checksum: str | None

    def evaluate(
        self, prepared: PreparedCellposeInput, *, config: CellposeModelConfig
    ) -> CellposeRawResult: ...


# ─── StarDist-engine-facing types ─────────────────────────────────────────────
#
# Deliberately separate from the cellpose types rather than a shared
# "MLEngine", because the two libraries differ in a way that is not cosmetic:
# cellpose takes `anisotropy` at predict time, while StarDist bakes anisotropy
# into the trained model. Forcing both behind one interface would hide that,
# and hiding it is how a volume gets fed to a model at the wrong sampling.


@dataclass(frozen=True, slots=True)
class PreparedStarDistInput:
    """Model-ready input for StarDist 3D, plus how to get labels home again.

    ``data`` is a single ``(Z, Y, X)`` float32 volume -- StarDist 3D predicts
    from one intensity channel, not a channel stack.

    ``spacing_um_zyx`` is the sampling of ``data`` AS GIVEN TO THE MODEL, which
    differs from the acquired spacing whenever ``resample_isotropic`` is on.
    ``original_shape_zyx`` is what the labels must be mapped back to; every
    physical measurement happens on that grid, never on the model grid.
    """

    data: NDArray[np.floating]
    channel_id: str
    spacing_um_zyx: tuple[float, float, float]
    original_shape_zyx: tuple[int, int, int]
    was_resampled: bool

    def __post_init__(self) -> None:
        if self.data.ndim != 3:
            raise ValueError(
                f"PreparedStarDistInput.data must be (Z, Y, X), got shape "
                f"{self.data.shape}"
            )


@dataclass(frozen=True, slots=True)
class StarDistRawResult:
    """Raw output of one ``predict_instances`` call, still on the model grid."""

    labels: NDArray[np.integer]
    n_instances: int
    extra: Mapping[str, Scalar] = field(default_factory=dict)


@runtime_checkable
class StarDistEngine(Protocol):
    """One concrete, loaded StarDist model.

    Implemented by ``StarDist3DEngine`` (the only module in this package allowed
    to import ``stardist``/``tensorflow``) and built by
    ``factory.create_stardist_engine``. Backends receive one by dependency
    injection, so every backend test runs against a fake with no ML installed.

    ``model_anisotropy`` is the anisotropy the loaded model was TRAINED with, so
    callers can check it against the data's own sampling rather than assuming.
    ``None`` means the model did not record one.
    """

    model_name: str
    package_version: str
    model_anisotropy: tuple[float, float, float] | None
    is_pretrained: bool

    def predict(self, prepared: PreparedStarDistInput) -> StarDistRawResult: ...
