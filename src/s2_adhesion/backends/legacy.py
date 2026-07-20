"""``LegacyThreshold2DBackend``: the original 2D threshold pipeline, exposed
as an analysis backend so its results sit in the same ``objects.csv`` table
format as the new 3D pipeline and can be compared directly.

Shape note on ``AnalysisBackend``
----------------------------------
``s2_adhesion.backends.protocol`` does not exist and, per the person
coordinating this refactor, will not. ``s2_adhesion.backends.ml3d`` (owned by
a concurrent agent) is expected to define the canonical backend shape: a
``backend_id`` attribute plus ``run(source, *, config, output_dir) ->
RunArtifacts``. At the time this module was written, ``ml3d.py`` did not yet
exist in this checkout, so :class:`RunArtifacts` below is an independent,
best-effort match of that coordinator-specified shape. If ``ml3d.py``'s
actual ``RunArtifacts`` ends up different, only the dataclass below needs to
change -- ``LegacyThreshold2DBackend.run()``'s body does not, since it only
depends on ``RunArtifacts`` having ``backend_id``/``bundle``/``written_paths``.

All pixel-level computation is delegated to ``s2_adhesion.legacy.algorithm``,
extracted verbatim (in behaviour) from the original ``detect_aggregations.py``.
This module's only job is adaptation: turning ``ImageVolume``\\ s from a
``VolumeSource`` into the arrays/dicts those functions expect, and turning
the original ``RegionRecord`` output into ``ObjectRecord`` rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from ..config import LegacyConfig, PipelineConfig
from ..contracts import ImageVolume, MeasurementBundle, VolumeSource
from ..errors import ConfigError, ContractViolation
from ..io.tables import write_measurement_bundle
from ..legacy import algorithm
from ..metrics.records import build_object_record, merge_bundles

_BACKEND_ID = "legacy_threshold_2d"
_OBJECT_KIND = "legacy_aggregate_2d"


@dataclass(frozen=True, slots=True)
class RunArtifacts:
    """What one backend run produced. See module docstring for the
    provenance of this shape."""

    backend_id: str
    bundle: MeasurementBundle
    written_paths: Mapping[str, Path]


def _image_sizes(vol: ImageVolume) -> dict[str, int]:
    """``sizes`` dict for ``vol.data``, in the same key order as its axes.

    ``ImageVolume.data`` is always CZYX (frozen in ``contracts.py``), so this
    dict's key order always matches the array's actual dimension order --
    exactly what every ``s2_adhesion.legacy.algorithm`` function assumes when
    it does ``dim_keys.index(k)`` to find an axis by name.
    """
    n_c, n_z, n_y, n_x = vol.data.shape
    return {"C": n_c, "Z": n_z, "Y": n_y, "X": n_x}


def _pixel_size_um(vol: ImageVolume) -> float:
    """X spacing only -- matches the original script's ``voxel_size().x`` and
    its implicit assumption of square, isotropic-in-plane pixels. Preserved
    deliberately for comparability with results already collected using the
    original algorithm; see ``s2_adhesion.io.nd2_source`` for the documented
    fix this repo does NOT backport into the legacy path.
    """
    _dz, _dy, dx = vol.geometry.spacing_um_zyx
    return float(dx)


def _algorithm_config(legacy_cfg: LegacyConfig) -> algorithm.Config:
    """Adapt the frozen ``config.LegacyConfig`` into the original script's
    ``Config`` NamedTuple. Field values are copied straight across --
    ``minimum_cell_equivalents`` is deliberately NOT cast to ``int`` even
    though the original ``Config.aggregation_min_cells`` is typed ``int``,
    because ``compute_area_threshold_px`` only ever multiplies by it; casting
    would silently truncate a fractional configured value and change the
    area threshold.
    """
    return algorithm.Config(
        s2_diameter_um=legacy_cfg.s2_diameter_um,
        aggregation_min_cells=legacy_cfg.minimum_cell_equivalents,  # type: ignore[arg-type]
        morph_close_radius_um=legacy_cfg.closing_radius_um,
        binary_threshold=legacy_cfg.threshold_uint8,
        min_active_channels=legacy_cfg.minimum_active_channels,
    )


class LegacyThreshold2DBackend:
    """Analysis backend wrapping the original 2D threshold pipeline.

    Emits ``ObjectRecord`` rows with ``backend_id="legacy_threshold_2d"`` and
    ``object_kind="legacy_aggregate_2d"`` so legacy and 3D results sit in one
    table and can be compared directly.
    """

    backend_id: str = _BACKEND_ID

    def run(
        self, source: VolumeSource, *, config: PipelineConfig, output_dir: Path
    ) -> RunArtifacts:
        output_dir = Path(output_dir)
        legacy_cfg = config.legacy
        if legacy_cfg is None:
            raise ConfigError(
                "LegacyThreshold2DBackend.run requires config.legacy to be "
                "set (analysis_backend='legacy_threshold_2d' should already "
                "enforce this via config.validate_config, but this backend "
                "checks independently rather than trusting the caller)"
            )
        alg_cfg = _algorithm_config(legacy_cfg)

        field_ids = list(source.field_ids())
        images = [source.read_field(fid) for fid in field_ids]

        global_minmax = _global_channel_minmax(images)

        per_field_bundles = [
            MeasurementBundle(
                objects=tuple(
                    self._process_field(vol, alg_cfg, legacy_cfg, global_minmax)
                )
            )
            for vol in images
        ]
        bundle = merge_bundles(*per_field_bundles)

        written = write_measurement_bundle(bundle, output_dir)
        return RunArtifacts(
            backend_id=self.backend_id, bundle=bundle, written_paths=written
        )

    def _process_field(
        self,
        vol: ImageVolume,
        alg_cfg: algorithm.Config,
        legacy_cfg: LegacyConfig,
        global_minmax: list[tuple[float, float]],
    ) -> list:
        sizes = _image_sizes(vol)
        pixel_size_um = _pixel_size_um(vol)

        median_k = algorithm.compute_median_kernel_size(pixel_size_um, alg_cfg.s2_diameter_um)
        channel_projections = algorithm.extract_channel_projections(vol.data, sizes, global_minmax)
        center_slices = algorithm.extract_center_slices(
            vol.data, sizes, pixel_size_um, alg_cfg.s2_diameter_um, global_minmax
        )
        gray_merged = algorithm.merge_channels(channel_projections)
        gray_merged_median = algorithm.median_blur(gray_merged, median_k)

        _mask_binary, _mask_filled, mask_final = algorithm.segment_occupancy(
            gray_merged_median, alg_cfg, pixel_size_um
        )
        binary_center_slices = algorithm.binarize_slices(center_slices, alg_cfg.binary_threshold)

        area_threshold_px = algorithm.compute_area_threshold_px(
            pixel_size_um, alg_cfg.s2_diameter_um, legacy_cfg.minimum_cell_equivalents
        )

        # field_id passed here is bookkeeping internal to RegionRecord only
        # (unused beyond this function); the real field identity comes from
        # vol.identity below.
        records, valid_mask, _rejected_mask = algorithm.detect_aggregations(
            mask_final, binary_center_slices, pixel_size_um, 0,
            area_threshold_px, alg_cfg.min_active_channels,
        )

        bboxes = algorithm.component_bboxes(valid_mask)
        if len(bboxes) != len(records):
            raise ContractViolation(
                "internal invariant violated: valid_mask must contain exactly "
                f"one connected component per accepted RegionRecord (got "
                f"{len(bboxes)} components for {len(records)} records in "
                f"field {vol.identity.field_id!r})"
            )

        h, w = valid_mask.shape
        objects = []
        for record, (x, y, bw, bh, _area) in zip(records, bboxes):
            touches_xy_border = x == 0 or y == 0 or x + bw >= w or y + bh >= h
            objects.append(
                build_object_record(
                    dataset_id=vol.identity.dataset_id,
                    field_id=vol.identity.field_id,
                    backend_id=self.backend_id,
                    object_kind=_OBJECT_KIND,
                    object_id=record.aggregation_id,
                    touches_xy_border=touches_xy_border,
                    # A MIP-based 2D aggregate has no bounded Z extent -- every
                    # pixel already sees the whole Z stack via max-projection
                    # -- so it can never be claimed "untruncated in Z".
                    # Conservatively always True; see class docstring.
                    touches_z_border=True,
                    # 2D projection-based measurement has no canonical 3D
                    # geometry (volume, surface area, hull...); this gate
                    # must never open for legacy aggregates.
                    valid_for_geometry=False,
                    metrics=(
                        {
                            "area_px": record.area_px,
                            "area_um2": record.area_um2,
                            "active_channels": record.active_channels,
                        },
                    ),
                )
            )
        return objects


def _global_channel_minmax(images: list[ImageVolume]) -> list[tuple[float, float]]:
    """Global per-channel (lo, hi) across ALL fields and Z, matching the
    original script's definition (``compute_global_channel_minmax`` called
    once over the whole multi-position nd2 array before iterating fields).

    A ``VolumeSource`` hands out one field's ``ImageVolume`` at a time, so
    this computes the per-channel min/max field-by-field (reusing the
    verbatim ``algorithm.compute_global_channel_minmax``) and reduces with
    min-of-mins / max-of-maxes -- mathematically identical to computing it
    over the concatenation of every field's data.
    """
    if not images:
        return []
    per_field = [
        algorithm.compute_global_channel_minmax(vol.data, _image_sizes(vol))
        for vol in images
    ]
    n_channels = len(per_field[0])
    return [
        (
            min(mm[ch][0] for mm in per_field),
            max(mm[ch][1] for mm in per_field),
        )
        for ch in range(n_channels)
    ]
