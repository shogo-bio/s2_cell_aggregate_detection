"""Nucleus-seeded watershed segmentation.

Dense S2 aggregates -- the whole point of this pipeline -- are exactly where
direct cell-body instance segmentation fails: touching membranes merge and
cells split wrongly. Nuclei are well separated and near-convex, so they seed
reliably even inside a clump. This backend segments nuclei first, then grows
a watershed from those seeds, constrained to a "this is cell, not background"
extent, using the membrane/cytoplasm channels (smoothed) as the flooding
cost surface.

The ML model itself is injected (``CellposeEngine``), never imported here, so
this module -- and its tests -- run with no cellpose/torch installed. See the
guarded import block below: ``segmentation.protocol`` is owned by a parallel
workstream. The real names are imported when present; an equivalent local
fallback (kept in lockstep with the real shapes) is defined otherwise. Either
way, ``SegmentationBackend``, ``SegmentationRequest``, ``SegmentationResult``,
``SegmentationDiagnostics``, ``PreparedCellposeInput``, ``CellposeEngine`` and
``CellposeRawResult`` are available from this module for tests to import.

``segmentation.preprocess`` (also owned elsewhere) is imported directly, not
guarded: ``map_labels_to_original_grid`` is public, generic infrastructure --
it maps any model-grid label array back onto ``PreparedCellposeInput``'s
original ZYX grid by nearest neighbour -- and reusing it is exactly the point
of that function existing outside ``DirectCellposeBackend``. Building the
*forward* direction (normalise + resample to a square in-plane pixel) is
duplicated locally in ``_prepare_cellpose_input`` because
``preprocess.prepare_cellpose_input`` is typed specifically to
``DirectInstanceConfig``'s channel-selection shape (``input_channel_ids`` /
``model``), which ``NucleusSeededWatershedConfig`` does not share
(``nucleus_channel_id`` / ``boundary_channel_ids`` / ``seed_model`` /
``extent_model``).
"""

from __future__ import annotations

import hashlib
import platform
import time
from dataclasses import dataclass, field
from typing import Literal, Mapping, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed

from ..config import CellposeModelConfig, NormalizationConfig, NucleusSeededWatershedConfig
from ..contracts import ImageVolume, LabelVolume, Scalar, SegmentationProvenance
from ..errors import SegmentationError
from . import postprocess
from .preprocess import map_labels_to_original_grid

try:  # pragma: no cover - exercised whichever branch is live
    from .protocol import (  # type: ignore[import-not-found]
        CellposeEngine,
        CellposeRawResult,
        PreparedCellposeInput,
        SegmentationBackend,
        SegmentationDiagnostics,
        SegmentationRequest,
        SegmentationResult,
    )
except ImportError:  # pragma: no cover - protocol.py not landed by the parallel agent yet
    @dataclass(frozen=True, slots=True)
    class SegmentationRequest:  # type: ignore[no-redef]
        """What any backend needs for one field. Strategy params live on the
        backend instance (constructed once per configured strategy), not here."""

        image: ImageVolume
        run_id: str

    @dataclass(frozen=True, slots=True)
    class SegmentationDiagnostics:  # type: ignore[no-redef]
        warnings: tuple[str, ...] = ()
        timing_seconds: Mapping[str, float] = field(default_factory=dict)
        extra: Mapping[str, Scalar] = field(default_factory=dict)

    @dataclass(frozen=True, slots=True)
    class SegmentationResult:  # type: ignore[no-redef]
        labels: LabelVolume
        diagnostics: SegmentationDiagnostics

    @runtime_checkable
    class SegmentationBackend(Protocol):  # type: ignore[no-redef]
        def segment(self, request: SegmentationRequest) -> SegmentationResult: ...

    @dataclass(frozen=True, slots=True)
    class PreparedCellposeInput:  # type: ignore[no-redef]
        """Model-ready input, plus everything needed to map results back home.

        ``data`` is ``(n_channels, Z, Y', X')``, float32, with Y'/X' resampled
        to a square in-plane pixel (``min(dy, dx)``); Z is never resampled.
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
                    "PreparedCellposeInput.data must be (C, Z, Y, X), got shape "
                    f"{self.data.shape}"
                )
            if self.data.shape[0] != len(self.channel_ids):
                raise ValueError(
                    f"data has {self.data.shape[0]} channels but channel_ids has "
                    f"{len(self.channel_ids)}: {self.channel_ids}"
                )

    @dataclass(frozen=True, slots=True)
    class CellposeRawResult:  # type: ignore[no-redef]
        """Raw output of one cellpose ``eval`` call, still on the model grid."""

        labels: NDArray[np.integer]
        diameters_px: float | None = None
        extra: Mapping[str, Scalar] = field(default_factory=dict)

    @runtime_checkable
    class CellposeEngine(Protocol):  # type: ignore[no-redef]
        """A bound, ready-to-run cellpose model, injected rather than imported."""

        package_major: Literal[3, 4]
        model_name: str
        package_version: str
        device: str
        model_checksum: str | None

        def evaluate(
            self, prepared: PreparedCellposeInput, *, config: CellposeModelConfig
        ) -> CellposeRawResult: ...


_BACKEND_ID = "nucleus_seeded_watershed"


def compute_gaussian_sigma_voxels(
    sigma_um: float, spacing_um_zyx: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Convert one physical Gaussian sigma (um) to a per-axis voxel sigma.

    ``sigma_vox[axis] = sigma_um / spacing_um_zyx[axis]``. With Z spacing 5x
    coarser than XY (a typical confocal stack), the same physical blur needs a
    *smaller* voxel-sigma in Z and a *larger* one in XY -- using one scalar
    sigma in voxel units would smear Z five times too far past what the
    config actually asked for.
    """
    if sigma_um < 0:
        raise SegmentationError(f"gaussian_sigma_um must be >= 0, got {sigma_um}")
    if any(s <= 0 for s in spacing_um_zyx):
        raise SegmentationError(f"spacing_um_zyx must be strictly positive, got {spacing_um_zyx}")
    dz, dy, dx = spacing_um_zyx
    return (sigma_um / dz, sigma_um / dy, sigma_um / dx)


# ─── local mirror of preprocess.py's forward (image -> model grid) step ────
#
# See the module docstring for why this is not imported from ``.preprocess``.


def _normalize_percentile(
    channel: NDArray[np.generic], norm_config: NormalizationConfig
) -> NDArray[np.float32]:
    """Percentile-normalise one channel to roughly [0, 1].

    A degenerate channel (upper percentile <= lower percentile) maps to all
    zeros rather than dividing by zero.
    """
    data = np.asarray(channel, dtype=np.float64)
    lo = float(np.percentile(data, norm_config.lower_percentile))
    hi = float(np.percentile(data, norm_config.upper_percentile))
    if hi <= lo:
        return np.zeros_like(data, dtype=np.float32)
    normalized = (data - lo) / (hi - lo)
    if norm_config.clip:
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
    """Resample one ZYX channel so its in-plane pixel spacing is square. Z untouched."""
    dz, dy, dx = spacing_um_zyx
    target = min(dy, dx)
    scale_y = dy / target
    scale_x = dx / target
    if scale_y == 1.0 and scale_x == 1.0:
        return channel.astype(np.float32, copy=False)
    resampled = ndi.zoom(channel, zoom=(1.0, scale_y, scale_x), order=1)
    return resampled.astype(np.float32)


def _prepare_cellpose_input(
    image: ImageVolume,
    channel_ids: tuple[str, ...],
    model_config: CellposeModelConfig,
) -> PreparedCellposeInput:
    """Select, normalise and resample ``channel_ids`` of ``image`` for cellpose."""
    resampled_channels = [
        _resample_inplane(
            _normalize_percentile(image.channel(cid), model_config.normalization),
            image.geometry.spacing_um_zyx,
        )
        for cid in channel_ids
    ]
    stacked = np.stack(resampled_channels, axis=0)
    resampled_spacing = _target_inplane_spacing_um(image.geometry.spacing_um_zyx)
    anisotropy = image.geometry.anisotropy_z_to_xy
    diameter_um = model_config.diameter_um
    diameter_px = diameter_um / resampled_spacing[1] if diameter_um is not None else None
    return PreparedCellposeInput(
        data=stacked,
        channel_ids=channel_ids,
        anisotropy=anisotropy,
        original_shape_zyx=image.shape_zyx,
        resampled_spacing_um_zyx=resampled_spacing,
        diameter_px=diameter_px,
    )


# ─── extent + cost surface (operate on the ORIGINAL grid, no cellpose needed
# for adaptive_intensity) ────────────────────────────────────────────────────


def _boundary_signal(
    image: ImageVolume, boundary_channel_ids: tuple[str, ...]
) -> NDArray[np.float64]:
    if not boundary_channel_ids:
        raise SegmentationError("at least one boundary_channel_id is required")
    stacked = np.stack(
        [np.asarray(image.channel(cid), dtype=np.float64) for cid in boundary_channel_ids],
        axis=0,
    )
    return stacked.mean(axis=0)


def _adaptive_intensity_extent(
    image: ImageVolume, boundary_channel_ids: tuple[str, ...]
) -> NDArray[np.bool_]:
    """Cell extent = the (Otsu-thresholded) complement of the boundary signal.

    Membrane/cytoplasm boundary channels are bright at cell borders; the
    segmentable interior is the darker complement. A boundary signal with no
    contrast at all (e.g. an empty synthetic channel) has no threshold to
    find -- ``threshold_otsu`` requires a non-constant image -- so that
    degenerate case falls back to "everything is extent" rather than raising.
    """
    signal = _boundary_signal(image, boundary_channel_ids)
    if signal.max() <= signal.min():
        extent = np.ones(signal.shape, dtype=bool)
    else:
        threshold = threshold_otsu(signal)
        extent = signal <= threshold
    return ndi.binary_fill_holes(extent)


def _config_fingerprint(config: NucleusSeededWatershedConfig) -> str:
    """Deterministic hash of the config used for one run, for provenance."""
    return hashlib.sha256(repr(config).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class NucleusSeededWatershedBackend:
    """Seeds a watershed from nucleus instances, injected model(s) only.

    ``config`` and both engines are bound once at construction (matching
    ``DirectCellposeBackend``'s shape): ``SegmentationRequest`` only ever
    carries the image and a run id. ``seed_engine`` runs ``config.seed_model``
    on the nucleus channel to produce seed instances -- this is the only
    engine required, and the only one exercised by unit tests with no ML
    installed. ``extent_engine`` is only consulted when
    ``config.extent_mode == "cellpose_union"``.
    """

    config: NucleusSeededWatershedConfig
    seed_engine: CellposeEngine
    extent_engine: CellposeEngine | None = None
    backend_id: str = _BACKEND_ID

    def segment(self, request: SegmentationRequest) -> SegmentationResult:
        image = request.image
        config = self.config
        spacing = image.geometry.spacing_um_zyx
        warnings: list[str] = []
        timings: dict[str, float] = {}

        # ── 1. seed instances from the nucleus channel ─────────────────────
        t0 = time.perf_counter()
        seed_prepared = _prepare_cellpose_input(
            image, (config.nucleus_channel_id,), config.seed_model
        )
        timings["prepare_seed"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        raw_seed = self.seed_engine.evaluate(seed_prepared, config=config.seed_model)
        timings["evaluate_seed"] = time.perf_counter() - t0

        raw_nucleus_labels = map_labels_to_original_grid(raw_seed.labels, seed_prepared)

        nucleus_labels = postprocess.split_disconnected_labels(raw_nucleus_labels)
        n_seeds_before = len(np.unique(nucleus_labels)) - (1 if (nucleus_labels == 0).any() else 0)
        nucleus_labels = postprocess.filter_by_physical_volume(
            nucleus_labels, spacing, config.min_nucleus_volume_um3
        )
        n_seeds_after = len(np.unique(nucleus_labels)) - (1 if (nucleus_labels == 0).any() else 0)
        n_nuclei_below_min_volume = n_seeds_before - n_seeds_after
        if n_nuclei_below_min_volume > 0:
            warnings.append(
                f"nucleus_below_min_volume: removed {n_nuclei_below_min_volume} nucleus "
                f"seed(s) below min_nucleus_volume_um3={config.min_nucleus_volume_um3}"
            )
        nucleus_labels = postprocess.relabel_sequential(nucleus_labels)

        # ── 2. cell extent ──────────────────────────────────────────────────
        t0 = time.perf_counter()
        extent_mask = self._build_extent(image)
        timings["build_extent"] = time.perf_counter() - t0

        structure = ndi.generate_binary_structure(3, 3)
        extent_components, n_components = ndi.label(extent_mask, structure=structure)

        # ── 3. flag zero-seed / over-full extents and orphan nuclei ────────
        nucleus_ids = sorted(int(i) for i in np.unique(nucleus_labels) if i > 0)
        dominant_component: dict[int, int | None] = {}
        for nucleus_id in nucleus_ids:
            overlap = extent_components[nucleus_labels == nucleus_id]
            overlap = overlap[overlap > 0]
            if overlap.size == 0:
                dominant_component[nucleus_id] = None
                warnings.append(
                    f"orphan_nucleus: nucleus seed {nucleus_id} does not overlap any "
                    "segmentation extent and will not seed a cell"
                )
                continue
            values, counts = np.unique(overlap, return_counts=True)
            dominant_component[nucleus_id] = int(values[np.argmax(counts)])

        nuclei_per_component: dict[int, int] = {}
        for component_id in dominant_component.values():
            if component_id is not None:
                nuclei_per_component[component_id] = nuclei_per_component.get(component_id, 0) + 1

        n_zero_seed_extents = 0
        n_oversegmented_extents = 0
        for component_id in range(1, n_components + 1):
            count = nuclei_per_component.get(component_id, 0)
            if count == 0:
                n_zero_seed_extents += 1
                warnings.append(
                    f"zero_seed_extent: extent component {component_id} contains no "
                    "nucleus seeds and will produce no cell"
                )
            elif count > config.max_nuclei_per_cell:
                n_oversegmented_extents += 1
                warnings.append(
                    f"oversegmented_extent: extent component {component_id} contains "
                    f"{count} nucleus seeds, exceeding max_nuclei_per_cell="
                    f"{config.max_nuclei_per_cell}"
                )

        # ── 4. watershed cost surface (anisotropic Gaussian, in um) + flood ──
        t0 = time.perf_counter()
        boundary_signal = _boundary_signal(image, config.boundary_channel_ids)
        sigma_vox = compute_gaussian_sigma_voxels(config.gaussian_sigma_um, spacing)
        cost_surface = ndi.gaussian_filter(boundary_signal, sigma=sigma_vox)

        raw_cells = watershed(
            cost_surface,
            markers=nucleus_labels.astype(np.int64),
            mask=extent_mask,
            compactness=config.watershed_compactness,
        ).astype(np.uint32)
        timings["watershed"] = time.perf_counter() - t0

        # ── 5. flag + drop cells below the physical volume floor ───────────
        before_ids = set(int(i) for i in np.unique(raw_cells) if i > 0)
        cells = postprocess.filter_by_physical_volume(raw_cells, spacing, config.min_cell_volume_um3)
        after_ids = set(int(i) for i in np.unique(cells) if i > 0)
        removed_ids = sorted(before_ids - after_ids)
        voxel_volume_um3 = spacing[0] * spacing[1] * spacing[2]
        for removed_id in removed_ids:
            voxel_count = int(np.sum(raw_cells == removed_id))
            volume_um3 = voxel_count * voxel_volume_um3
            warnings.append(
                f"cell_below_min_volume: cell {removed_id} volume {volume_um3:.4f} um3 < "
                f"min_cell_volume_um3={config.min_cell_volume_um3}, removed"
            )
        n_cells_below_min_volume = len(removed_ids)

        # ── 6. shared cleanup, stable renumbering ───────────────────────────
        cells = postprocess.split_disconnected_labels(cells)
        cells = postprocess.fill_internal_holes(cells)
        cells = postprocess.relabel_sequential(cells)

        provenance = SegmentationProvenance(
            run_id=request.run_id,
            backend_id=self.backend_id,
            strategy=_BACKEND_ID,
            config_sha256=_config_fingerprint(config),
            input_image_sha256=image.identity.image_content_sha256,
            device=self.seed_engine.device,
            host_platform=platform.platform(),
            package_name="cellpose",
            package_version=self.seed_engine.package_version,
            model_name=self.seed_engine.model_name,
            model_sha256=self.seed_engine.model_checksum,
        )
        labels = LabelVolume(
            cells=cells.astype(np.uint32),
            geometry=image.geometry,
            identity=image.identity,
            provenance=provenance,
            nuclei=nucleus_labels.astype(np.uint32),
        )
        diagnostics = SegmentationDiagnostics(
            warnings=tuple(warnings),
            timing_seconds=timings,
            extra={
                "n_zero_seed_extents": n_zero_seed_extents,
                "n_oversegmented_extents": n_oversegmented_extents,
                "n_orphan_nuclei": sum(1 for v in dominant_component.values() if v is None),
                "n_cells_below_min_volume": n_cells_below_min_volume,
                "n_nuclei_below_min_volume": n_nuclei_below_min_volume,
            },
        )
        return SegmentationResult(labels=labels, diagnostics=diagnostics)

    def _build_extent(self, image: ImageVolume) -> NDArray[np.bool_]:
        config = self.config
        if config.extent_mode == "cellpose_union":
            return self._cellpose_union_extent(image)
        if config.extent_mode == "adaptive_intensity":
            return _adaptive_intensity_extent(image, config.boundary_channel_ids)
        raise SegmentationError(f"unknown extent_mode {config.extent_mode!r}")  # pragma: no cover

    def _cellpose_union_extent(self, image: ImageVolume) -> NDArray[np.bool_]:
        config = self.config
        if self.extent_engine is None:
            raise SegmentationError(
                "extent_mode 'cellpose_union' requires an extent_engine to be injected"
            )
        if config.extent_model is None:
            raise SegmentationError(
                "extent_mode 'cellpose_union' requires config.extent_model (validate_config "
                "should have caught this already)"
            )
        prepared = _prepare_cellpose_input(image, config.boundary_channel_ids, config.extent_model)
        raw = self.extent_engine.evaluate(prepared, config=config.extent_model)
        mapped = map_labels_to_original_grid(raw.labels, prepared)
        return mapped > 0
