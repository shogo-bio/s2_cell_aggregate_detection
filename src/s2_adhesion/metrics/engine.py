"""Orchestrates every ``metrics.*`` module into one :class:`MeasurementBundle`.

This is the one place that knows the full dependency order between the Wave
1/2 metric modules for the ``ml_instance_3d`` backend:

    qc -> geometry -> contacts -> aggregates
    (intensity, nuclei, localization only when an image is present)

Pure numpy/scipy/scikit-image transitively -- this module imports only
sibling ``metrics`` modules plus ``..config``/``..contracts``, so it carries
no ML dependency and is safe to import on a machine with no torch/cellpose
installed (required: ``commands.measure`` imports this module and must never
pull in ML packages, even transitively).

CRITICAL CONTRACT: ``image=None`` is a legitimate, expected input (the
"segmentation ran on a GPU box, but nobody has an image artifact handy, or a
labels-only geometry pass is all that's wanted" case), never an error.
``compute_measurements`` must still produce geometry/contact/aggregate
metrics in that case and must never raise merely because ``image`` is
``None`` -- every intensity/nuclei/localization field that would have needed
an image is instead recorded as missing, with the reason
``"missing_image_artifact"``, both as a bundle-level warning and as a
per-object marker key (there is no way to emit real per-channel columns like
``ch.<id>.raw_mean`` without an image to name channels from, so those
columns simply never appear in that run's CSVs -- which is what "empty"
means for a data-driven column set, see ``io.tables``).
"""

from __future__ import annotations

from typing import Mapping

from ..config import MeasurementConfig
from ..contracts import (
    ChannelRole,
    ImageVolume,
    LabelVolume,
    MeasurementBundle,
    Scalar,
)
from . import aggregates as aggregates_metrics
from . import contacts as contacts_metrics
from . import geometry as geometry_metrics
from . import intensity as intensity_metrics
from . import localization as localization_metrics
from . import populations as populations_metrics
from . import radial as radial_metrics
from . import saturation as saturation_metrics
from . import surface as surface_metrics
from . import nuclei as nuclei_metrics
from . import qc as qc_metrics
from . import records

__all__ = ["compute_measurements", "MISSING_IMAGE_REASON"]

# Recorded whenever intensity/nuclei/localization metrics are skipped for
# lack of an image artifact -- the one reason this orchestrator ever nulls
# those fields rather than raising. Fixed string so callers can match on it.
MISSING_IMAGE_REASON = "missing_image_artifact"

_DEFAULT_BACKEND_ID = "ml_instance_3d"


def _truncated_ids(
    touches_xy: Mapping[int, bool], touches_z: Mapping[int, bool]
) -> set[int]:
    """Cell ids cropped by the field of view on any axis.

    One shared definition used both for ``metrics.contacts.compute_contacts``
    (``truncated_cell_ids``) and ``metrics.aggregates.compute_aggregates``
    (``truncated_cell_ids``), matching
    ``contracts.ObjectRecord.is_truncated``.
    """
    return {cid for cid in touches_xy if touches_xy[cid] or touches_z[cid]}



# Radial-score keys whose quantitative value is destroyed by a clipped core.
_RADIAL_SCORE_KEYS = (
    "radial_shell_score",
    "peripheral_signal_fraction",
    "radial_peak_position",
)


def _apply_saturation_gates(
    radial_rows: dict[int, dict[str, Scalar]],
    coloc_rows: dict[int, dict[str, Scalar]],
    saturation_rows: dict[int, dict[str, Scalar]],
    channel_ids,
    coloc_pairs,
) -> None:
    """Null or flag intensity-derived metrics per each cell's saturation status.

    Follows the reviewed policy: a clipped CORE makes the ring/interior score
    indeterminate (nulled, but a clipped core is itself positive evidence of
    interior signal, recorded as a reason); a clipped SHELL leaves the score as a
    valid downward-biased LOWER bound (kept, flagged); material saturation in
    either channel of a colocalization pair nulls Pearson/Manders. Mutates the
    row dicts in place.
    """
    for cid, sat_row in saturation_rows.items():
        radial = radial_rows.get(cid)
        if radial is not None:
            for channel_id in channel_ids:
                reliability = saturation_metrics.localization_reliability(
                    sat_row, channel_id
                )
                radial[f"ch.{channel_id}.localization_reliability"] = reliability
                if reliability == saturation_metrics.LOC_INDETERMINATE:
                    for key in _RADIAL_SCORE_KEYS:
                        full = f"ch.{channel_id}.{key}"
                        if full in radial:
                            radial[full] = None
                    radial[f"ch.{channel_id}.radial_profile_qc"] = (
                        "saturated_core_localization_indeterminate"
                    )

        coloc = coloc_rows.get(cid)
        if coloc is not None:
            for pair in coloc_pairs:
                if not saturation_metrics.colocalization_trustworthy(
                    sat_row, pair.signal_channel_id, pair.reference_channel_id
                ):
                    prefix = f"{pair.signal_channel_id}_vs_{pair.reference_channel_id}"
                    for key in list(coloc):
                        if key.startswith(prefix + "__"):
                            coloc[key] = None
                    coloc[f"{prefix}__colocalization_qc"] = "saturated_channel"


def compute_measurements(
    labels: LabelVolume,
    image: ImageVolume | None,
    config: MeasurementConfig,
    *,
    backend_id: str = _DEFAULT_BACKEND_ID,
) -> MeasurementBundle:
    """Compute every metric this pipeline knows for one field's labels.

    ``labels`` and ``image`` are already-loaded, in-memory objects -- artifact
    reading/binding validation is the caller's job
    (``commands.measure``/``commands.segment`` + ``io.zarr_store``); this
    function only does the in-memory contract guard
    (``labels.validate_against(image)``, a no-op when ``image`` is ``None``)
    as defence in depth.

    ``object_kind`` is always ``"cell_3d"`` -- this engine exists for the
    ``ml_instance_3d`` analysis backend; a 2D legacy pipeline builds its own
    records elsewhere. ``backend_id`` is the *analysis* backend id (e.g.
    ``"ml_instance_3d"``, ``config.analysis_backend`` in
    ``PipelineConfig``) recorded on every ``ObjectRecord`` -- distinct from
    ``labels.provenance.backend_id``, which names the *segmentation*
    strategy (e.g. ``"direct_cellpose"``) that produced these labels.

    Never raises merely because ``image`` is ``None`` -- see the module
    docstring for what happens to intensity/nuclei/localization fields in
    that case. Real failures (a genuinely corrupt/inconsistent label array,
    a duplicate metric key, ...) propagate unchanged; this function does not
    swallow them.
    """
    labels.validate_against(image)

    dataset_id = labels.identity.dataset_id
    field_id = labels.identity.field_id
    run_id = labels.provenance.run_id
    cells = labels.cells
    geometry = labels.geometry

    cell_ids = [int(i) for i in labels.cell_ids()]

    warnings: list[str] = []

    # ── qc ───────────────────────────────────────────────────────────────
    touches_xy = qc_metrics.touches_xy_border(cells)
    touches_z = qc_metrics.touches_z_border(cells)
    valid_geometry = qc_metrics.valid_for_geometry(cells)
    cc_count = qc_metrics.connected_component_count(cells)
    qc_codes = qc_metrics.qc_code(cells)
    truncated_ids = _truncated_ids(touches_xy, touches_z)

    # ── geometry ─────────────────────────────────────────────────────────
    geom_rows = geometry_metrics.compute_geometry(cells, geometry)

    # Free/contact/guard surface split. Geometry only -- no image needed -- and
    # computed for every cell because it supplies the denominator any future
    # surface-coverage measurement has to use: signal on contact surface belongs
    # to both cells and is missing data, not measurable coverage.
    surface_rows = surface_metrics.compute_surface_partition(
        cells, geometry, config.optics
    )

    # ── contacts ─────────────────────────────────────────────────────────
    contact_records = contacts_metrics.compute_contacts(
        labels, config.contact, truncated_cell_ids=truncated_ids
    )
    surface_areas: dict[int, float | None] = {
        cid: geom_rows.get(cid, {}).get("surface_area_um2") for cid in cell_ids
    }
    adhesion_rows = contacts_metrics.compute_adhesion_aggregates(
        contact_records, cell_ids, surface_areas
    )

    # ── aggregates ───────────────────────────────────────────────────────
    cell_volumes: dict[int, float | None] = {
        cid: geom_rows.get(cid, {}).get("volume_um3") for cid in cell_ids
    }
    aggregate_records = aggregates_metrics.compute_aggregates(
        cells,
        geometry,
        contact_records,
        cell_volumes,
        truncated_ids,
        dataset_id=dataset_id,
        field_id=field_id,
        segmentation_run_id=run_id,
    )
    cell_to_aggregate: dict[int, int] = {
        member: agg.aggregate_id
        for agg in aggregate_records
        for member in agg.member_cell_ids
    }

    # ── intensity / nuclei / localization: only with an image ──────────────
    intensity_rows: dict[int, dict[str, Scalar]] = {}
    radial_rows: dict[int, dict[str, Scalar]] = {}
    population_rows: dict[int, dict[str, Scalar]] = {}
    saturation_rows: dict[int, dict[str, Scalar]] = {}
    nuclei_rows: dict[int, dict[str, Scalar]] = {}
    shell_rows: dict[int, dict[str, Scalar]] = {}
    coloc_rows: dict[int, dict[str, Scalar]] = {}
    localization_profiles: tuple = ()

    if image is None:
        warnings.append(
            f"{MISSING_IMAGE_REASON}: intensity/nuclei/localization metrics "
            f"skipped for field {field_id!r} because no image artifact was "
            "supplied; geometry/contact/aggregate metrics are unaffected"
        )
    else:
        channels: dict[str, object] = {
            c.channel_id: image.channel(c.channel_id)
            for c in image.channels
            if not c.has_role(ChannelRole.IGNORE)
        }

        # Assign each cell to a population when the config marks any channel with
        # one. Background-relative so FITC's much higher raw brightness does not
        # sweep every cell into the green group.
        if any(c.population for c in image.channels):
            pop_cfg = config.population
            pop_channels = {
                c.channel_id: image.channel(c.channel_id) for c in image.channels
                if c.population
            }
            if pop_cfg.method == "intensity_ratio":
                population_rows = populations_metrics.assign_populations_by_intensity_ratio(
                    cells,
                    pop_channels,
                    image.channels,
                    floors=pop_cfg.floors_for(field_id),
                    ratio_low=pop_cfg.ratio_low,
                    ratio_high=pop_cfg.ratio_high,
                    double_label=pop_cfg.double_label,
                )
            else:
                population_rows = populations_metrics.assign_populations(
                    cells,
                    pop_channels,
                    image.channels,
                    min_score=pop_cfg.min_score_mad,
                    dominance_ratio=pop_cfg.dominance_ratio,
                )

        intensity_rows = intensity_metrics.compute_intensity(
            image,
            cells,
            config.background,
            config.localization.inner_shell_width_um,
            config.localization.outer_shell_width_um,
        )

        nuclei_rows = nuclei_metrics.compute_nuclei(cells, labels.nuclei, geometry)

        localization_profiles = localization_metrics.compute_signed_distance_profiles(
            cells, geometry, dataset_id, field_id, channels, config.localization
        )

        # Per-channel (median, mad) background, reused as both the
        # colocalization threshold source and the shell-enrichment epsilon
        # (mad approximates the "caller's background noise estimate" that
        # module asks for). Background is field-level, shared by every cell
        # (see metrics.intensity's docstring), so any one cell's row carries
        # the same value -- read it back from compute_intensity's own public
        # output rather than reaching into that module's private helpers.
        background_stats: dict[str, tuple[float, float]] = {}
        noise_std: dict[str, float] = {}
        if cell_ids:
            sample_row = intensity_rows.get(cell_ids[0], {})
            for cid in channels:
                bg = sample_row.get(f"ch.{cid}.background")
                mad = sample_row.get(f"ch.{cid}.background_mad")
                if bg is not None:
                    background_stats[cid] = (float(bg), float(mad) if mad is not None else 0.0)
                if mad is not None:
                    noise_std[cid] = float(mad)

        shell_rows = localization_metrics.compute_shell_enrichment_scores(
            cells, geometry, channels, config.localization, background_noise_std=noise_std
        )

        # Normalised radial profiles: the membrane-vs-interior readout. Reuses
        # the same per-channel background and noise estimates so its scores are
        # on the same footing as the shell-enrichment ones.
        radial_rows = radial_metrics.compute_radial_metrics(
            cells,
            channels,
            geometry,
            background_by_channel={c: bg for c, (bg, _) in background_stats.items()},
            eps_by_channel={c: max(m, radial_metrics.DEFAULT_EPS)
                            for c, m in noise_std.items()},
        )
        if config.localization.colocalization_pairs:
            coloc_rows = localization_metrics.compute_colocalization(
                cells, geometry, channels, config.localization, background_stats=background_stats
            )

        # ── saturation: detect clipping and gate the intensity-derived metrics ──
        # Measured on RAW integer channels (pre background-subtraction), because
        # clipping only shows as exact-ceiling values. Real acquisitions here
        # clip (FITC) and under-expose (mCherry), silently corrupting the ring/
        # interior score and colocalization -- so those are nulled or flagged per
        # the saturation status rather than reported as clean.
        detector_max, dmax_provenance = saturation_metrics.resolve_detector_max(
            image.data, config.saturation.detector_max
        )
        if "uncertain" in dmax_provenance:
            warnings.append(
                f"detector_max auto-detected as {detector_max} with low "
                "confidence (data never reaches a standard ceiling); set "
                "measurement.saturation.detector_max explicitly to be sure"
            )
        raw_channels = {
            c.channel_id: image.channel(c.channel_id)
            for c in image.channels
            if ChannelRole.IGNORE not in c.roles
        }
        saturation_rows = saturation_metrics.compute_saturation(
            cells, raw_channels, image.channels, geometry, detector_max,
            inner_shell_width_um=config.localization.inner_shell_width_um,
        )
        _apply_saturation_gates(
            radial_rows, coloc_rows, saturation_rows, raw_channels.keys(),
            config.localization.colocalization_pairs,
        )

        if config.localization.decision.enabled:
            warnings.append(
                "localization_decision_skipped: "
                "LocalizationDecisionConfig.enabled=True, but "
                "metrics.engine does not compute categorical localization "
                "calls (metrics.localization.classify_localization) -- "
                "continuous shell-enrichment and colocalization scores are "
                "still recorded on each cell"
            )

    # ── flatten into ObjectRecords ──────────────────────────────────────────
    objects = []
    for cid in cell_ids:
        metrics_sources: list[Mapping[str, Scalar]] = [
            {"connected_component_count": cc_count.get(cid), "qc_code": qc_codes.get(cid, "")},
            geom_rows.get(cid, {}),
            surface_rows.get(cid, {}),
            adhesion_rows.get(cid, {}),
        ]
        if image is None:
            metrics_sources.append({"intensity_missing_reason": MISSING_IMAGE_REASON})
            metrics_sources.append({"nuclei_missing_reason": MISSING_IMAGE_REASON})
            metrics_sources.append({"localization_missing_reason": MISSING_IMAGE_REASON})
        else:
            metrics_sources.append(intensity_rows.get(cid, {}))
            metrics_sources.append(nuclei_rows.get(cid, {}))
            metrics_sources.append(shell_rows.get(cid, {}))
            metrics_sources.append(radial_rows.get(cid, {}))
            metrics_sources.append(coloc_rows.get(cid, {}))
            metrics_sources.append(population_rows.get(cid, {}))
            metrics_sources.append(saturation_rows.get(cid, {}))

        objects.append(
            records.build_object_record(
                dataset_id=dataset_id,
                field_id=field_id,
                backend_id=backend_id,
                object_kind="cell_3d",
                object_id=cid,
                touches_xy_border=touches_xy.get(cid, False),
                touches_z_border=touches_z.get(cid, False),
                valid_for_geometry=valid_geometry.get(cid, False),
                metrics=metrics_sources,
                aggregate_id=cell_to_aggregate.get(cid),
                segmentation_run_id=run_id,
            )
        )

    # ── field-level summary: the adhesion mixing index ─────────────────────
    # A single number per field (how much the cell populations touch across
    # groups vs within), so it belongs here rather than in objects.csv. Computed
    # only when populations were assigned -- i.e. a channel carried a population
    # label -- otherwise there is nothing to mix.
    field_values: dict[str, Scalar] = {}
    if population_rows:
        cell_population = {
            cid: str(row.get("population"))
            for cid, row in population_rows.items()
            if row.get("population") is not None
        }
        field_values.update(
            populations_metrics.compute_mixing(
                contact_records, cell_population,
                non_population_labels=frozenset({config.population.double_label}),
            )
        )
    field_summaries: tuple = ()
    if field_values:
        field_summaries = (
            records.build_field_summary_record(
                dataset_id=dataset_id, field_id=field_id, metrics=[field_values]
            ),
        )

    return MeasurementBundle(
        objects=tuple(objects),
        contacts=tuple(contact_records),
        aggregates=tuple(aggregate_records),
        localization_profiles=tuple(localization_profiles),
        field_summaries=field_summaries,
        warnings=tuple(warnings),
    )
