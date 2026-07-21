"""Flatten computed metric dicts into the frozen output records.

``ObjectRecord``, ``ContactRecord`` and ``AggregateRecord`` all carry an
open-ended ``values: Mapping[str, Scalar]`` (frozen in ``contracts.py``) so
that adding a metric becomes a new CSV column with no change to any writer.
That only works if merging metric dicts is strict: two metric functions that
(accidentally) compute a key of the same name must never silently overwrite
one another, and every value handed to a record must be a genuine
serialisable ``Scalar`` -- in particular never ``float('nan')`` or an
infinity, which would not serialise as the unambiguous empty field that
``io.tables`` relies on for ``None``.

Identity and QC fields are explicit, required keyword arguments on every
``build_*`` function here -- they must never be smuggled through a metric
dict, since that would make a field that must never be optional look
optional by accident.

This module only knows about ``dict[str, Scalar]`` and the frozen record
dataclasses in ``contracts.py``. It never imports a metric-computing module
(``metrics.geometry``, ``metrics.qc``, a future ``metrics.contact``, ...),
which is why record fixtures can be built directly, with no metrics
implementation needing to exist yet.
"""

from __future__ import annotations

import math
from typing import Literal, Mapping, Sequence

from ..contracts import (
    AggregateRecord,
    ContactEstimator,
    ContactRecord,
    FieldSummaryRecord,
    LocalizationProfileRecord,
    MeasurementBundle,
    ObjectRecord,
    Scalar,
)
from ..errors import ContractViolation

__all__ = [
    "build_field_summary_record",
    "DuplicateMetricKeyError",
    "InvalidMetricValueError",
    "merge_metric_values",
    "build_object_record",
    "build_contact_record",
    "build_aggregate_record",
    "build_localization_profile_record",
    "merge_bundles",
]


class DuplicateMetricKeyError(ContractViolation):
    """Two metric sources computed the same key for one record.

    Letting the second source silently overwrite the first is exactly how a
    bug in one metric function corrupts a column nobody suspects, so this is
    a hard error rather than a last-write-wins merge.
    """


class InvalidMetricValueError(ContractViolation):
    """A metric value is not a genuine, CSV-serialisable ``Scalar``."""


def _validate_scalar(key: str, value: Scalar) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        return
    if isinstance(value, float):
        if math.isnan(value):
            raise InvalidMetricValueError(
                f"metric {key!r} is float('nan'); use None for a missing "
                "value instead -- NaN would not serialise as the "
                "unambiguous empty field the None convention relies on"
            )
        if math.isinf(value):
            raise InvalidMetricValueError(
                f"metric {key!r} is {value!r}, which has no CSV-safe "
                "representation; compute None instead of an infinite value"
            )
        return
    if isinstance(value, (int, str)):
        return
    raise InvalidMetricValueError(
        f"metric {key!r} has value {value!r} of type {type(value).__name__}, "
        "which is not a str | int | float | bool | None Scalar"
    )


def merge_metric_values(*sources: Mapping[str, Scalar]) -> dict[str, Scalar]:
    """Union several metric dicts into one, validating and rejecting overlap.

    Each ``source`` is normally the per-object/per-pair row returned by one
    metric-computing function (``metrics.qc``'s per-label dicts,
    ``metrics.geometry.compute_geometry``'s per-label row, a future
    ``metrics.contact``, ...). A key present in more than one source almost
    certainly means two functions independently computed "the same"
    quantity under the same name -- silently keeping only the last one would
    hide that divergence, so this raises :class:`DuplicateMetricKeyError`
    instead of merging it away.
    """
    merged: dict[str, Scalar] = {}
    origin: dict[str, int] = {}
    for source_index, source in enumerate(sources):
        for key, value in source.items():
            _validate_scalar(key, value)
            if key in merged:
                raise DuplicateMetricKeyError(
                    f"metric key {key!r} is produced by more than one source "
                    f"(source #{origin[key]} and source #{source_index}); "
                    "each metric name must be computed by exactly one "
                    "function -- rename one of them or drop the duplicate"
                )
            merged[key] = value
            origin[key] = source_index
    return merged


def build_object_record(
    *,
    dataset_id: str,
    field_id: str,
    backend_id: str,
    object_kind: Literal["cell_3d", "legacy_aggregate_2d"],
    object_id: int,
    touches_xy_border: bool,
    touches_z_border: bool,
    valid_for_geometry: bool,
    metrics: Sequence[Mapping[str, Scalar]] = (),
    aggregate_id: int | None = None,
    segmentation_run_id: str | None = None,
) -> ObjectRecord:
    """Build one ``ObjectRecord``, merging every metric dict in ``metrics``."""
    return ObjectRecord(
        dataset_id=dataset_id,
        field_id=field_id,
        backend_id=backend_id,
        object_kind=object_kind,
        object_id=object_id,
        touches_xy_border=touches_xy_border,
        touches_z_border=touches_z_border,
        valid_for_geometry=valid_for_geometry,
        values=merge_metric_values(*metrics),
        aggregate_id=aggregate_id,
        segmentation_run_id=segmentation_run_id,
    )



def build_field_summary_record(
    *,
    dataset_id: str,
    field_id: str,
    metrics: Sequence[Mapping[str, Scalar]] = (),
) -> FieldSummaryRecord:
    """Assemble one field-level summary row (e.g. the adhesion mixing index)."""
    return FieldSummaryRecord(
        dataset_id=dataset_id,
        field_id=field_id,
        values=merge_metric_values(*metrics),
    )


def build_contact_record(
    *,
    dataset_id: str,
    field_id: str,
    segmentation_run_id: str,
    cell_id_a: int,
    cell_id_b: int,
    contact_area_um2: float,
    estimator: ContactEstimator,
    qualifies_as_contact: bool,
    valid_for_contact_metrics: bool,
    metrics: Sequence[Mapping[str, Scalar]] = (),
) -> ContactRecord:
    """Build one ``ContactRecord``, merging every metric dict in ``metrics``.

    ``ContactRecord.__post_init__`` (frozen in ``contracts.py``) already
    enforces ``cell_id_a < cell_id_b``; this function does not duplicate
    that check.
    """
    return ContactRecord(
        dataset_id=dataset_id,
        field_id=field_id,
        segmentation_run_id=segmentation_run_id,
        cell_id_a=cell_id_a,
        cell_id_b=cell_id_b,
        contact_area_um2=contact_area_um2,
        estimator=estimator,
        qualifies_as_contact=qualifies_as_contact,
        valid_for_contact_metrics=valid_for_contact_metrics,
        values=merge_metric_values(*metrics),
    )


def build_aggregate_record(
    *,
    dataset_id: str,
    field_id: str,
    segmentation_run_id: str,
    aggregate_id: int,
    member_cell_ids: Sequence[int],
    contains_truncated_cell: bool,
    metrics: Sequence[Mapping[str, Scalar]] = (),
) -> AggregateRecord:
    """Build one ``AggregateRecord``, merging every metric dict in ``metrics``."""
    return AggregateRecord(
        dataset_id=dataset_id,
        field_id=field_id,
        segmentation_run_id=segmentation_run_id,
        aggregate_id=aggregate_id,
        member_cell_ids=tuple(member_cell_ids),
        contains_truncated_cell=contains_truncated_cell,
        values=merge_metric_values(*metrics),
    )


def build_localization_profile_record(
    *,
    dataset_id: str,
    field_id: str,
    cell_id: int,
    channel_id: str,
    reference: Literal["cell_boundary", "nucleus_boundary", "organelle_boundary"],
    bin_start_um: float,
    bin_end_um: float,
    voxel_count: int,
    sampled_volume_um3: float,
    mean_intensity: float | None = None,
    integrated_corrected_intensity: float | None = None,
) -> LocalizationProfileRecord:
    """Validated constructor for ``LocalizationProfileRecord``.

    This record has no open-ended ``values`` mapping to merge -- every field
    is explicit in ``contracts.py`` -- so this is a thin pass-through kept
    for symmetry with the other ``build_*`` functions, plus the one ordering
    check the frozen dataclass itself does not make.
    """
    if bin_end_um <= bin_start_um:
        raise ContractViolation(
            "localization bin must have bin_end_um > bin_start_um, got "
            f"[{bin_start_um}, {bin_end_um}]"
        )
    return LocalizationProfileRecord(
        dataset_id=dataset_id,
        field_id=field_id,
        cell_id=cell_id,
        channel_id=channel_id,
        reference=reference,
        bin_start_um=bin_start_um,
        bin_end_um=bin_end_um,
        voxel_count=voxel_count,
        sampled_volume_um3=sampled_volume_um3,
        mean_intensity=mean_intensity,
        integrated_corrected_intensity=integrated_corrected_intensity,
    )


def merge_bundles(*bundles: MeasurementBundle) -> MeasurementBundle:
    """Concatenate several per-field bundles into one for a whole-run write.

    ``io.tables`` writes one set of four CSVs per call; a full analysis run
    spans many fields of view, so callers combine their per-field bundles
    with this before writing.
    """
    return MeasurementBundle(
        objects=tuple(o for b in bundles for o in b.objects),
        contacts=tuple(c for b in bundles for c in b.contacts),
        aggregates=tuple(a for b in bundles for a in b.aggregates),
        localization_profiles=tuple(
            p for b in bundles for p in b.localization_profiles
        ),
        field_summaries=tuple(
            fs for b in bundles for fs in b.field_summaries
        ),
        warnings=tuple(w for b in bundles for w in b.warnings),
    )
