"""Write a :class:`~s2_adhesion.contracts.MeasurementBundle` to CSV plus a manifest.

Four tables, one manifest, written next to each other in an output directory::

    objects.csv                  one row per ObjectRecord
    contacts.csv                 one row per ContactRecord
                                  key (field_id, cell_id_a, cell_id_b)
    aggregates.csv               one row per AggregateRecord
    localization_profiles.csv    one row per LocalizationProfileRecord
                                  key (field_id, cell_id, channel_id, reference,
                                       bin_start_um)
    metrics_manifest.json        name / unit / dtype / nullable / definition
                                  for every column in every CSV above

"Long format" means one row per OBJECT in ``objects.csv``. Pairwise data
(``contacts.csv``) and per-bin profile data (``localization_profiles.csv``)
live in their own tables and are never duplicated into the object table.

Serialisation rules enforced here, not negotiable by callers:

* ``None`` is an EMPTY CSV field -- never the string ``"None"``, ``"nan"``,
  ``"null"``, and never the number ``0``. A truncated cell's volume must be
  readably absent, because a ``0`` there would silently bias any downstream
  average.
* booleans always serialise as the literal strings ``"True"`` / ``"False"``.
* floats always serialise with Python's shortest round-tripping ``repr``, so
  the same in-memory value always produces the same bytes on any run.
* ``objects.csv`` (and ``contacts.csv`` / ``aggregates.csv``) column order is
  every explicit record field (everything except ``values``) in the order
  ``contracts.py`` declares it, then every metric key seen anywhere in the
  batch, sorted alphabetically -- stable across runs no matter which record
  happened to be built first, and no matter which records carry which keys.
  A metric key present on some records and absent on others still gets one
  shared column; "absent" serialises exactly like ``None``: an empty field.

Unit inference for metric columns follows the convention frozen in
``contracts.py``: "Distances are micrometres, areas um^2, volumes um^3. No
pixel units escape into a physical field." A metric name ending in
``_um``/``_um2``/``_um3`` is documented with that unit; anything else is
documented "unitless" -- a metric author who wants a recorded physical unit
must name the column accordingly. This module never imports a metric module,
so the name is the only signal it has.

Deliberately dependency-light: only the standard library ``csv``/``json`` do
the writing (no pandas import here), so measurement-side callers can write
tables with no ML stack, and no pandas, on the path at all. Tests read the
files back with pandas to verify round-tripping from the consumer's side.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..contracts import (
    SCHEMA_VERSION_RECORDS,
    AggregateRecord,
    ContactRecord,
    LocalizationProfileRecord,
    MeasurementBundle,
    ObjectRecord,
    Scalar,
)
from ..errors import ContractViolation

__all__ = [
    "write_objects_csv",
    "write_contacts_csv",
    "write_aggregates_csv",
    "write_localization_profiles_csv",
    "build_metrics_manifest",
    "write_metrics_manifest",
    "write_measurement_bundle",
]

_MEMBER_ID_SEP = ";"

# (name, unit, dtype, nullable, one-line definition)
_ColumnSpec = tuple[str, str, str, bool, str]

_OBJECT_FIXED_COLUMNS: tuple[_ColumnSpec, ...] = (
    ("dataset_id", "unitless", "str", False, "Dataset this field belongs to."),
    ("field_id", "unitless", "str", False, "Field of view this object was measured in."),
    ("backend_id", "unitless", "str", False, "Segmentation backend that produced this object."),
    ("object_kind", "unitless", "str", False, "'cell_3d' or legacy 'legacy_aggregate_2d'."),
    ("object_id", "unitless", "int", False, "Label id; not necessarily consecutive, never renumbered."),
    ("touches_xy_border", "unitless", "bool", False, "True if the object touches the X or Y edge of the volume."),
    ("touches_z_border", "unitless", "bool", False, "True if the object touches the Z edge of the volume."),
    ("valid_for_geometry", "unitless", "bool", False, "True iff untruncated and single-component; gates canonical geometry fields."),
    ("aggregate_id", "unitless", "int", True, "Id of the aggregate this object is a member of, if any."),
    ("segmentation_run_id", "unitless", "str", True, "Id of the segmentation run that produced this object, if known."),
    ("schema_version", "unitless", "str", False, "Schema version this record was written under."),
)

_CONTACT_FIXED_COLUMNS: tuple[_ColumnSpec, ...] = (
    ("dataset_id", "unitless", "str", False, "Dataset this field belongs to."),
    ("field_id", "unitless", "str", False, "Field of view this contact was measured in."),
    ("segmentation_run_id", "unitless", "str", False, "Segmentation run that produced both cells in this pair."),
    ("cell_id_a", "unitless", "int", False, "Smaller of the two touching cell ids (cell_id_a < cell_id_b)."),
    ("cell_id_b", "unitless", "int", False, "Larger of the two touching cell ids (cell_id_a < cell_id_b)."),
    ("contact_area_um2", "um^2", "float", False, "Estimated cell-cell interface area."),
    ("estimator", "unitless", "str", False, "Which interface-area estimator computed contact_area_um2."),
    ("qualifies_as_contact", "unitless", "bool", False, "True if the pair passes the configured contact threshold."),
    ("valid_for_contact_metrics", "unitless", "bool", False, "True iff both cells pass QC for contact measurement."),
)

_AGGREGATE_FIXED_COLUMNS: tuple[_ColumnSpec, ...] = (
    ("dataset_id", "unitless", "str", False, "Dataset this field belongs to."),
    ("field_id", "unitless", "str", False, "Field of view this aggregate was found in."),
    ("segmentation_run_id", "unitless", "str", False, "Segmentation run that produced the member cells."),
    ("aggregate_id", "unitless", "int", False, "Id of this connected component within its field."),
    ("member_cell_ids", "unitless", "str", False, "';'-joined member cell ids, unambiguous even for non-consecutive ids, e.g. '1;5;900'."),
    ("contains_truncated_cell", "unitless", "bool", False, "True if any member cell touches a volume border."),
)

_LOCALIZATION_COLUMNS: tuple[_ColumnSpec, ...] = (
    ("dataset_id", "unitless", "str", False, "Dataset this field belongs to."),
    ("field_id", "unitless", "str", False, "Field of view this profile was measured in."),
    ("cell_id", "unitless", "int", False, "Cell this bin belongs to."),
    ("channel_id", "unitless", "str", False, "Logical channel this intensity profile was sampled from."),
    ("reference", "unitless", "str", False, "Boundary this profile's signed distance is measured from."),
    ("bin_start_um", "um", "float", False, "Signed-distance bin lower edge (inclusive)."),
    ("bin_end_um", "um", "float", False, "Signed-distance bin upper edge (exclusive)."),
    ("voxel_count", "unitless", "int", False, "Number of voxels sampled in this bin."),
    ("sampled_volume_um3", "um^3", "float", False, "Physical volume sampled in this bin."),
    ("mean_intensity", "unitless", "float", True, "Mean raw intensity in this bin (arbitrary units), if computed."),
    ("integrated_corrected_intensity", "unitless", "float", True, "Background/bleed-corrected integrated intensity in this bin, if computed."),
)

_UNIT_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("_um3", "um^3"),
    ("_um2", "um^2"),
    ("_um", "um"),
)


# ─── scalar <-> CSV field ───────────────────────────────────────────────────


def _serialize_scalar(value: Scalar) -> str:
    """The one place a Python value becomes a CSV field. See module docstring."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ContractViolation(
                f"cannot serialise non-finite float {value!r} to CSV; "
                "producers must use None for a missing value"
            )
        return repr(value)
    if isinstance(value, int):
        return str(value)
    return str(value)


def _scalar_dtype(value: Scalar) -> str | None:
    """Manifest dtype label for one value, or ``None`` if ``value`` is null."""
    if value is None:
        return None
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, float):
        return "float"
    if isinstance(value, int):
        return "int"
    return "str"


def _infer_metric_unit(name: str) -> str:
    """Unit implied by a metric's name, per the suffix convention in ``contracts.py``."""
    for suffix, unit in _UNIT_SUFFIXES:
        if name.endswith(suffix):
            return unit
    return "unitless"


# ─── generic CSV writer ─────────────────────────────────────────────────────


def _write_csv(path: Path | str, columns: Sequence[str], rows: Iterable[Sequence[Scalar]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(columns)
        for row in rows:
            writer.writerow([_serialize_scalar(v) for v in row])


def _compound_key_duplicates(keys: Sequence[tuple]) -> list[tuple]:
    seen: set[tuple] = set()
    dupes: list[tuple] = []
    for k in keys:
        if k in seen:
            dupes.append(k)
        seen.add(k)
    return dupes


def _sorted_metric_keys(values_list: Iterable[Mapping[str, Scalar]]) -> list[str]:
    return sorted({key for values in values_list for key in values})


# ─── objects.csv ────────────────────────────────────────────────────────────


def write_objects_csv(objects: Sequence[ObjectRecord], path: Path | str) -> list[str]:
    """Write ``objects.csv``: exactly one row per object. Returns the header."""
    metric_keys = _sorted_metric_keys(o.values for o in objects)
    columns = [c[0] for c in _OBJECT_FIXED_COLUMNS] + metric_keys

    def rows() -> Iterable[list[Scalar]]:
        for o in objects:
            yield [
                o.dataset_id,
                o.field_id,
                o.backend_id,
                o.object_kind,
                o.object_id,
                o.touches_xy_border,
                o.touches_z_border,
                o.valid_for_geometry,
                o.aggregate_id,
                o.segmentation_run_id,
                o.schema_version,
            ] + [o.values.get(key) for key in metric_keys]

    _write_csv(path, columns, rows())
    return columns


# ─── contacts.csv ───────────────────────────────────────────────────────────


def write_contacts_csv(contacts: Sequence[ContactRecord], path: Path | str) -> list[str]:
    """Write ``contacts.csv``: one row per unordered pair, keyed on
    ``(field_id, cell_id_a, cell_id_b)``. Raises :class:`ContractViolation`
    if that compound key is not unique. Returns the header.
    """
    keys = [(c.field_id, c.cell_id_a, c.cell_id_b) for c in contacts]
    dupes = _compound_key_duplicates(keys)
    if dupes:
        raise ContractViolation(
            "contacts.csv compound key (field_id, cell_id_a, cell_id_b) is "
            f"not unique: duplicated {dupes[0]!r}"
        )

    metric_keys = _sorted_metric_keys(c.values for c in contacts)
    columns = [c[0] for c in _CONTACT_FIXED_COLUMNS] + metric_keys

    def rows() -> Iterable[list[Scalar]]:
        for c in contacts:
            yield [
                c.dataset_id,
                c.field_id,
                c.segmentation_run_id,
                c.cell_id_a,
                c.cell_id_b,
                c.contact_area_um2,
                str(c.estimator),
                c.qualifies_as_contact,
                c.valid_for_contact_metrics,
            ] + [c.values.get(key) for key in metric_keys]

    _write_csv(path, columns, rows())
    return columns


# ─── aggregates.csv ─────────────────────────────────────────────────────────


def write_aggregates_csv(aggregates: Sequence[AggregateRecord], path: Path | str) -> list[str]:
    """Write ``aggregates.csv``: one row per aggregate. Returns the header.

    ``member_cell_ids`` is serialised ``;``-joined (e.g. ``"1;5;900"``) so
    non-consecutive ids survive round-tripping unambiguously.
    """
    metric_keys = _sorted_metric_keys(a.values for a in aggregates)
    columns = [c[0] for c in _AGGREGATE_FIXED_COLUMNS] + metric_keys

    def rows() -> Iterable[list[Scalar]]:
        for a in aggregates:
            yield [
                a.dataset_id,
                a.field_id,
                a.segmentation_run_id,
                a.aggregate_id,
                _MEMBER_ID_SEP.join(str(i) for i in a.member_cell_ids),
                a.contains_truncated_cell,
            ] + [a.values.get(key) for key in metric_keys]

    _write_csv(path, columns, rows())
    return columns


# ─── localization_profiles.csv ──────────────────────────────────────────────


def write_localization_profiles_csv(
    profiles: Sequence[LocalizationProfileRecord], path: Path | str
) -> list[str]:
    """Write ``localization_profiles.csv``, keyed on ``(field_id, cell_id,
    channel_id, reference, bin_start_um)``. Raises :class:`ContractViolation`
    if that compound key is not unique. Returns the header.
    """
    keys = [
        (p.field_id, p.cell_id, p.channel_id, p.reference, p.bin_start_um)
        for p in profiles
    ]
    dupes = _compound_key_duplicates(keys)
    if dupes:
        raise ContractViolation(
            "localization_profiles.csv compound key (field_id, cell_id, "
            f"channel_id, reference, bin_start_um) is not unique: "
            f"duplicated {dupes[0]!r}"
        )

    columns = [c[0] for c in _LOCALIZATION_COLUMNS]

    def rows() -> Iterable[list[Scalar]]:
        for p in profiles:
            yield [
                p.dataset_id,
                p.field_id,
                p.cell_id,
                p.channel_id,
                p.reference,
                p.bin_start_um,
                p.bin_end_um,
                p.voxel_count,
                p.sampled_volume_um3,
                p.mean_intensity,
                p.integrated_corrected_intensity,
            ]

    _write_csv(path, columns, rows())
    return columns


# ─── metrics_manifest.json ──────────────────────────────────────────────────


def _metric_column_entries(values_list: Sequence[Mapping[str, Scalar]]) -> list[dict[str, Any]]:
    """One manifest entry per metric key seen across ``values_list``.

    Enforces that a key holds one consistent dtype everywhere it is
    non-null: a key that is an int in one row and a str in another means two
    different metrics collided under one name, which is exactly the
    silent-corruption failure mode ``metrics.records.merge_metric_values``
    also guards against within a single record.
    """
    keys = sorted({key for values in values_list for key in values})
    entries: list[dict[str, Any]] = []
    for key in keys:
        seen_dtype: str | None = None
        nullable = False
        for values in values_list:
            if key not in values:
                nullable = True
                continue
            dtype = _scalar_dtype(values[key])
            if dtype is None:
                nullable = True
                continue
            if seen_dtype is None:
                seen_dtype = dtype
            elif seen_dtype != dtype:
                raise ContractViolation(
                    f"metric column {key!r} has both {seen_dtype!r} and "
                    f"{dtype!r} values across records; a metric name must "
                    "mean the same thing -- and hold the same type -- "
                    "everywhere it appears"
                )
        entries.append(
            {
                "name": key,
                "unit": _infer_metric_unit(key),
                "dtype": seen_dtype or "str",
                "nullable": nullable,
                "definition": f"Metric column {key!r} (see the metric "
                "function that computed it for its definition).",
            }
        )
    return entries


def _table_entries(fixed: Sequence[_ColumnSpec], metric_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"name": n, "unit": u, "dtype": d, "nullable": nb, "definition": defn}
        for n, u, d, nb, defn in fixed
    ] + metric_entries


def build_metrics_manifest(bundle: MeasurementBundle) -> dict[str, Any]:
    """Column documentation for every CSV :func:`write_measurement_bundle` emits.

    Every column name appearing in any of the four CSVs appears here exactly
    once, per table, and vice versa: callers may rely on set equality
    between this manifest's per-table column names and the actual CSV
    header written for that table.
    """
    return {
        "schema_version": SCHEMA_VERSION_RECORDS,
        "tables": {
            "objects.csv": _table_entries(
                _OBJECT_FIXED_COLUMNS,
                _metric_column_entries([o.values for o in bundle.objects]),
            ),
            "contacts.csv": _table_entries(
                _CONTACT_FIXED_COLUMNS,
                _metric_column_entries([c.values for c in bundle.contacts]),
            ),
            "aggregates.csv": _table_entries(
                _AGGREGATE_FIXED_COLUMNS,
                _metric_column_entries([a.values for a in bundle.aggregates]),
            ),
            "localization_profiles.csv": _table_entries(_LOCALIZATION_COLUMNS, []),
        },
    }


def write_metrics_manifest(bundle: MeasurementBundle, path: Path | str) -> Path:
    """Write ``metrics_manifest.json``: sorted keys, fixed formatting, so two
    writers given the same bundle produce byte-identical files.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_metrics_manifest(bundle)
    path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return path


# ─── everything at once ─────────────────────────────────────────────────────


def write_measurement_bundle(bundle: MeasurementBundle, output_dir: Path | str) -> dict[str, Path]:
    """Write all four CSVs plus ``metrics_manifest.json`` into ``output_dir``.

    Returns a dict keyed ``"objects"``, ``"contacts"``, ``"aggregates"``,
    ``"localization_profiles"``, ``"metrics_manifest"`` -> the path written.
    An empty bundle still writes valid headers-only CSVs and a manifest
    listing only the fixed columns.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "objects": output_dir / "objects.csv",
        "contacts": output_dir / "contacts.csv",
        "aggregates": output_dir / "aggregates.csv",
        "localization_profiles": output_dir / "localization_profiles.csv",
        "metrics_manifest": output_dir / "metrics_manifest.json",
    }
    write_objects_csv(bundle.objects, paths["objects"])
    write_contacts_csv(bundle.contacts, paths["contacts"])
    write_aggregates_csv(bundle.aggregates, paths["aggregates"])
    write_localization_profiles_csv(bundle.localization_profiles, paths["localization_profiles"])
    write_metrics_manifest(bundle, paths["metrics_manifest"])
    return paths
