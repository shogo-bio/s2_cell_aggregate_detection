"""Tests for ``metrics.records`` (flattening metric dicts into records) and
``io.tables`` (writing those records to CSV + a manifest).

Record fixtures are built directly from the frozen dataclasses in
``contracts.py`` / via ``metrics.records.build_*`` -- no metrics
implementation needs to exist for any of this.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from s2_adhesion.contracts import (
    AggregateRecord,
    ContactEstimator,
    ContactRecord,
    ContractViolation,
    LocalizationProfileRecord,
    MeasurementBundle,
    ObjectRecord,
)
from s2_adhesion.io.tables import (
    build_metrics_manifest,
    write_aggregates_csv,
    write_contacts_csv,
    write_localization_profiles_csv,
    write_measurement_bundle,
    write_metrics_manifest,
    write_objects_csv,
)
from s2_adhesion.metrics.records import (
    DuplicateMetricKeyError,
    InvalidMetricValueError,
    build_aggregate_record,
    build_contact_record,
    build_localization_profile_record,
    build_object_record,
    merge_bundles,
    merge_metric_values,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


# ─── helpers / fixtures ─────────────────────────────────────────────────────


def _object(
    object_id: int,
    *,
    metrics=(),
    touches_xy_border: bool = False,
    touches_z_border: bool = False,
    valid_for_geometry: bool = True,
    aggregate_id: int | None = None,
) -> ObjectRecord:
    return build_object_record(
        dataset_id="ds0",
        field_id="field00",
        backend_id="ml_instance_3d",
        object_kind="cell_3d",
        object_id=object_id,
        touches_xy_border=touches_xy_border,
        touches_z_border=touches_z_border,
        valid_for_geometry=valid_for_geometry,
        metrics=metrics,
        aggregate_id=aggregate_id,
        segmentation_run_id="run0",
    )


def _contact(cell_id_a: int, cell_id_b: int, *, metrics=()) -> ContactRecord:
    return build_contact_record(
        dataset_id="ds0",
        field_id="field00",
        segmentation_run_id="run0",
        cell_id_a=cell_id_a,
        cell_id_b=cell_id_b,
        contact_area_um2=12.5,
        estimator=ContactEstimator.MARCHING_CUBES,
        qualifies_as_contact=True,
        valid_for_contact_metrics=True,
        metrics=metrics,
    )


def _aggregate(aggregate_id: int, member_cell_ids, *, metrics=()) -> AggregateRecord:
    return build_aggregate_record(
        dataset_id="ds0",
        field_id="field00",
        segmentation_run_id="run0",
        aggregate_id=aggregate_id,
        member_cell_ids=member_cell_ids,
        contains_truncated_cell=False,
        metrics=metrics,
    )


def _profile(cell_id: int, bin_start_um: float, **kw) -> LocalizationProfileRecord:
    return build_localization_profile_record(
        dataset_id="ds0",
        field_id="field00",
        cell_id=cell_id,
        channel_id="signal",
        reference="cell_boundary",
        bin_start_um=bin_start_um,
        bin_end_um=bin_start_um + 0.5,
        voxel_count=10,
        sampled_volume_um3=1.0,
        **kw,
    )


def _sample_bundle() -> MeasurementBundle:
    """Objects with non-consecutive ids, one truncated (nulled metrics)."""
    objects = (
        _object(1, metrics=[{"volume_um3": 120.5, "sphericity": 0.9}]),
        _object(
            5,
            metrics=[{"volume_um3": None, "sphericity": None, "voxel_count_observed": 42}],
            touches_xy_border=True,
            valid_for_geometry=False,
        ),
        _object(900, metrics=[{"volume_um3": 88.0, "sphericity": 0.75}], aggregate_id=1),
    )
    contacts = (_contact(1, 5, metrics=[{"contact_solidity": 0.42}]),)
    aggregates = (_aggregate(1, (1, 5, 900), metrics=[{"aggregate_area_um2": 500.25}]),)
    profiles = (
        _profile(1, -1.0, mean_intensity=100.0, integrated_corrected_intensity=90.0),
        _profile(1, -0.5, mean_intensity=None, integrated_corrected_intensity=None),
    )
    return MeasurementBundle(
        objects=objects, contacts=contacts, aggregates=aggregates, localization_profiles=profiles
    )


# ─── merge_metric_values ────────────────────────────────────────────────────


class TestMergeMetricValues:
    def test_merges_disjoint_dicts(self):
        merged = merge_metric_values({"a": 1}, {"b": 2.0}, {"c": "x"}, {"d": None})
        assert merged == {"a": 1, "b": 2.0, "c": "x", "d": None}

    def test_no_sources_gives_empty_dict(self):
        assert merge_metric_values() == {}

    def test_duplicate_key_across_sources_raises(self):
        with pytest.raises(DuplicateMetricKeyError, match="volume_um3"):
            merge_metric_values({"volume_um3": 1.0}, {"volume_um3": 2.0})

    def test_duplicate_key_error_names_both_source_indices(self):
        with pytest.raises(DuplicateMetricKeyError, match=r"source #0.*source #2"):
            merge_metric_values({"x": 1}, {"y": 2}, {"x": 3})

    def test_does_not_silently_overwrite(self):
        """The failure mode this exists to prevent: last-write-wins merging."""
        try:
            merge_metric_values({"volume_um3": 1.0}, {"volume_um3": 999.0})
        except DuplicateMetricKeyError:
            pass
        else:
            pytest.fail("expected DuplicateMetricKeyError, got a silently merged dict")

    def test_nan_float_is_rejected(self):
        with pytest.raises(InvalidMetricValueError, match="nan"):
            merge_metric_values({"bad": float("nan")})

    def test_infinite_float_is_rejected(self):
        with pytest.raises(InvalidMetricValueError):
            merge_metric_values({"bad": float("inf")})

    def test_non_scalar_value_is_rejected(self):
        with pytest.raises(InvalidMetricValueError):
            merge_metric_values({"bad": [1, 2, 3]})

    def test_none_and_bool_and_int_and_str_are_all_valid(self):
        merged = merge_metric_values({"a": None, "b": True, "c": 1, "d": "s"})
        assert merged == {"a": None, "b": True, "c": 1, "d": "s"}


# ─── build_* record constructors ────────────────────────────────────────────


class TestBuildRecords:
    def test_build_object_record_merges_metrics(self):
        obj = _object(1, metrics=[{"volume_um3": 1.0}, {"sphericity": 0.5}])
        assert obj.values == {"volume_um3": 1.0, "sphericity": 0.5}

    def test_build_object_record_duplicate_metric_raises(self):
        with pytest.raises(DuplicateMetricKeyError):
            _object(1, metrics=[{"volume_um3": 1.0}, {"volume_um3": 2.0}])

    def test_build_contact_record_merges_metrics(self):
        c = _contact(1, 5, metrics=[{"x": 1}, {"y": 2}])
        assert c.values == {"x": 1, "y": 2}

    def test_build_contact_record_still_enforces_ordered_pair(self):
        """contracts.ContactRecord.__post_init__ rejects a >= b; build_* must not swallow it."""
        with pytest.raises(ContractViolation, match="ordered"):
            build_contact_record(
                dataset_id="ds0",
                field_id="field00",
                segmentation_run_id="run0",
                cell_id_a=5,
                cell_id_b=1,
                contact_area_um2=1.0,
                estimator=ContactEstimator.MARCHING_CUBES,
                qualifies_as_contact=True,
                valid_for_contact_metrics=True,
            )

    def test_build_aggregate_record_merges_metrics_and_keeps_member_order(self):
        agg = _aggregate(1, (900, 1, 5), metrics=[{"a": 1}])
        assert agg.member_cell_ids == (900, 1, 5)
        assert agg.values == {"a": 1}

    def test_build_localization_profile_record_rejects_bad_bin_order(self):
        with pytest.raises(ContractViolation, match="bin_end_um"):
            build_localization_profile_record(
                dataset_id="ds0",
                field_id="field00",
                cell_id=1,
                channel_id="signal",
                reference="cell_boundary",
                bin_start_um=1.0,
                bin_end_um=0.5,
                voxel_count=1,
                sampled_volume_um3=1.0,
            )

    def test_merge_bundles_concatenates_every_table(self):
        b1 = MeasurementBundle(objects=(_object(1),), warnings=("w1",))
        b2 = MeasurementBundle(objects=(_object(5),), warnings=("w2",))
        merged = merge_bundles(b1, b2)
        assert [o.object_id for o in merged.objects] == [1, 5]
        assert merged.warnings == ("w1", "w2")


# ─── io.tables: round trip ──────────────────────────────────────────────────


class TestRoundTrip:
    def test_objects_round_trip_through_pandas(self, tmp_path):
        bundle = _sample_bundle()
        path = tmp_path / "objects.csv"
        write_objects_csv(bundle.objects, path)
        df = pd.read_csv(path)

        row1 = df[df["object_id"] == 1].iloc[0]
        assert row1["volume_um3"] == pytest.approx(120.5)
        assert row1["sphericity"] == pytest.approx(0.9)
        assert bool(row1["touches_xy_border"]) is False

        row900 = df[df["object_id"] == 900].iloc[0]
        assert row900["aggregate_id"] == 1

    def test_none_metric_is_empty_field_and_comes_back_as_nan(self, tmp_path):
        path = tmp_path / "objects.csv"
        write_objects_csv(_sample_bundle().objects, path)

        # Raw bytes: the truncated cell's volume must be a genuinely empty
        # field, never "None", "nan", "null", or "0".
        raw = path.read_text(encoding="utf-8")
        lines = raw.splitlines()
        header = lines[0].split(",")
        vol_idx = header.index("volume_um3")
        row5_fields = next(l for l in lines[1:] if l.split(",")[header.index("object_id")] == "5").split(",")
        assert row5_fields[vol_idx] == ""

        df = pd.read_csv(path)
        row5 = df[df["object_id"] == 5].iloc[0]
        assert pd.isna(row5["volume_um3"])
        assert row5["voxel_count_observed"] == 42

    def test_none_metric_never_serialises_as_zero(self, tmp_path):
        path = tmp_path / "objects.csv"
        write_objects_csv(_sample_bundle().objects, path)
        df = pd.read_csv(path)
        row5 = df[df["object_id"] == 5].iloc[0]
        # A biased-low 0.0 would be silently indistinguishable from a real
        # measurement; it must come back as missing, not as 0.
        assert row5["volume_um3"] != 0
        assert pd.isna(row5["volume_um3"])

    def test_booleans_serialise_as_true_false_literals(self, tmp_path):
        path = tmp_path / "objects.csv"
        write_objects_csv(_sample_bundle().objects, path)

        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
        header, data_rows = rows[0], rows[1:]
        xy_idx = header.index("touches_xy_border")
        z_idx = header.index("touches_z_border")
        oid_idx = header.index("object_id")

        by_id = {row[oid_idx]: row for row in data_rows}
        # object 1: neither border touched -> both False
        assert by_id["1"][xy_idx] == "False"
        assert by_id["1"][z_idx] == "False"
        # object 5: built with touches_xy_border=True
        assert by_id["5"][xy_idx] == "True"
        # never 0/1 or upper/lower-case variants
        for row in data_rows:
            assert row[xy_idx] in ("True", "False")
            assert row[z_idx] in ("True", "False")

        df = pd.read_csv(path)
        row1 = df[df["object_id"] == 1]
        assert row1["touches_xy_border"].iloc[0] in (False, "False")

    def test_non_consecutive_object_ids_survive(self, tmp_path):
        path = tmp_path / "objects.csv"
        write_objects_csv(_sample_bundle().objects, path)
        df = pd.read_csv(path)
        assert sorted(df["object_id"].tolist()) == [1, 5, 900]

    def test_aggregate_member_cell_ids_serialise_unambiguously(self, tmp_path):
        path = tmp_path / "aggregates.csv"
        write_aggregates_csv(_sample_bundle().aggregates, path)
        df = pd.read_csv(path)
        assert df.loc[0, "member_cell_ids"] == "1;5;900"

    def test_localization_profiles_round_trip(self, tmp_path):
        path = tmp_path / "localization_profiles.csv"
        write_localization_profiles_csv(_sample_bundle().localization_profiles, path)
        df = pd.read_csv(path)
        assert len(df) == 2
        row_missing = df[df["bin_start_um"] == -0.5].iloc[0]
        assert pd.isna(row_missing["mean_intensity"])
        assert pd.isna(row_missing["integrated_corrected_intensity"])
        row_present = df[df["bin_start_um"] == -1.0].iloc[0]
        assert row_present["mean_intensity"] == pytest.approx(100.0)


# ─── io.tables: schema / determinism ────────────────────────────────────────


class TestColumnOrderAndDeterminism:
    def test_identity_qc_columns_come_before_metric_columns_alphabetically(self, tmp_path):
        path = tmp_path / "objects.csv"
        columns = write_objects_csv(_sample_bundle().objects, path)
        metric_start = columns.index("sphericity")
        assert columns[:metric_start] == [
            "dataset_id", "field_id", "backend_id", "object_kind", "object_id",
            "touches_xy_border", "touches_z_border", "valid_for_geometry",
            "aggregate_id", "segmentation_run_id", "schema_version",
        ]
        metric_cols = columns[metric_start:]
        assert metric_cols == sorted(metric_cols)

    def test_column_order_identical_across_two_runs_regardless_of_row_order(self, tmp_path):
        bundle = _sample_bundle()
        forward = write_objects_csv(bundle.objects, tmp_path / "a.csv")
        reversed_objects = tuple(reversed(bundle.objects))
        backward = write_objects_csv(reversed_objects, tmp_path / "b.csv")
        assert forward == backward

    def test_writing_same_bundle_twice_is_byte_identical(self, tmp_path):
        bundle = _sample_bundle()
        dir1, dir2 = tmp_path / "run1", tmp_path / "run2"
        write_measurement_bundle(bundle, dir1)
        write_measurement_bundle(bundle, dir2)
        for name in (
            "objects.csv", "contacts.csv", "aggregates.csv",
            "localization_profiles.csv", "metrics_manifest.json",
        ):
            assert (dir1 / name).read_bytes() == (dir2 / name).read_bytes(), name

    def test_writing_same_bundle_twice_to_same_path_is_byte_identical(self, tmp_path):
        bundle = _sample_bundle()
        path = tmp_path / "objects.csv"
        write_objects_csv(bundle.objects, path)
        first = path.read_bytes()
        write_objects_csv(bundle.objects, path)
        second = path.read_bytes()
        assert first == second

    def test_different_metric_key_sets_produce_union_header_no_column_dropped(self, tmp_path):
        objects = (
            _object(1, metrics=[{"only_on_1": 1.0}]),
            _object(2, metrics=[{"only_on_2": "x"}]),
        )
        path = tmp_path / "objects.csv"
        columns = write_objects_csv(objects, path)
        assert "only_on_1" in columns
        assert "only_on_2" in columns

        df = pd.read_csv(path)
        row1 = df[df["object_id"] == 1].iloc[0]
        row2 = df[df["object_id"] == 2].iloc[0]
        assert row1["only_on_1"] == pytest.approx(1.0)
        assert pd.isna(row1["only_on_2"])
        assert row2["only_on_2"] == "x"
        assert pd.isna(row2["only_on_1"])

    def test_floats_use_a_fixed_repr(self, tmp_path):
        objects = (_object(1, metrics=[{"v": 1.0 / 3.0}]),)
        path = tmp_path / "objects.csv"
        write_objects_csv(objects, path)
        raw = path.read_text(encoding="utf-8")
        assert repr(1.0 / 3.0) in raw


# ─── io.tables: compound keys ───────────────────────────────────────────────


class TestCompoundKeys:
    def test_duplicate_contact_key_raises(self, tmp_path):
        contacts = (_contact(1, 5), _contact(1, 5))
        with pytest.raises(ContractViolation, match="contacts.csv"):
            write_contacts_csv(contacts, tmp_path / "contacts.csv")

    def test_unique_contact_keys_write_fine(self, tmp_path):
        contacts = (_contact(1, 5), _contact(1, 6), _contact(5, 6))
        columns = write_contacts_csv(contacts, tmp_path / "contacts.csv")
        assert "cell_id_a" in columns and "cell_id_b" in columns

    def test_duplicate_localization_key_raises(self, tmp_path):
        profiles = (_profile(1, -1.0), _profile(1, -1.0))
        with pytest.raises(ContractViolation, match="localization_profiles.csv"):
            write_localization_profiles_csv(profiles, tmp_path / "loc.csv")

    def test_same_cell_different_channel_is_not_a_duplicate(self, tmp_path):
        p1 = build_localization_profile_record(
            dataset_id="ds0", field_id="field00", cell_id=1, channel_id="signal",
            reference="cell_boundary", bin_start_um=-1.0, bin_end_um=-0.5,
            voxel_count=1, sampled_volume_um3=1.0,
        )
        p2 = build_localization_profile_record(
            dataset_id="ds0", field_id="field00", cell_id=1, channel_id="other",
            reference="cell_boundary", bin_start_um=-1.0, bin_end_um=-0.5,
            voxel_count=1, sampled_volume_um3=1.0,
        )
        columns = write_localization_profiles_csv((p1, p2), tmp_path / "loc.csv")
        assert columns  # writes without raising


# ─── io.tables: manifest ────────────────────────────────────────────────────


class TestMetricsManifest:
    def test_manifest_documents_every_objects_column(self, tmp_path):
        bundle = _sample_bundle()
        columns = write_objects_csv(bundle.objects, tmp_path / "objects.csv")
        manifest = build_metrics_manifest(bundle)
        manifest_names = {e["name"] for e in manifest["tables"]["objects.csv"]}
        assert manifest_names == set(columns)

    def test_manifest_documents_every_column_in_every_table(self, tmp_path):
        bundle = _sample_bundle()
        paths = write_measurement_bundle(bundle, tmp_path)
        manifest = json.loads(paths["metrics_manifest"].read_text(encoding="utf-8"))

        for table_name, csv_key in (
            ("objects.csv", "objects"),
            ("contacts.csv", "contacts"),
            ("aggregates.csv", "aggregates"),
            ("localization_profiles.csv", "localization_profiles"),
        ):
            actual_header = pd.read_csv(paths[csv_key]).columns.tolist()
            manifest_names = {e["name"] for e in manifest["tables"][table_name]}
            assert manifest_names == set(actual_header), table_name

    def test_every_manifest_entry_has_required_fields(self, tmp_path):
        bundle = _sample_bundle()
        manifest = build_metrics_manifest(bundle)
        for entries in manifest["tables"].values():
            for entry in entries:
                assert set(entry) == {"name", "unit", "dtype", "nullable", "definition"}
                assert isinstance(entry["unit"], str) and entry["unit"] != ""
                assert entry["dtype"] in {"str", "int", "float", "bool"}
                assert isinstance(entry["nullable"], bool)
                assert isinstance(entry["definition"], str) and entry["definition"]

    def test_physical_unit_columns_are_not_unitless(self, tmp_path):
        bundle = _sample_bundle()
        manifest = build_metrics_manifest(bundle)
        by_name = {e["name"]: e for e in manifest["tables"]["objects.csv"]}
        assert by_name["volume_um3"]["unit"] == "um^3"

        loc_by_name = {e["name"]: e for e in manifest["tables"]["localization_profiles.csv"]}
        assert loc_by_name["bin_start_um"]["unit"] == "um"
        assert loc_by_name["sampled_volume_um3"]["unit"] == "um^3"

    def test_manifest_json_is_byte_identical_across_two_writes(self, tmp_path):
        bundle = _sample_bundle()
        p1 = tmp_path / "m1.json"
        p2 = tmp_path / "m2.json"
        write_metrics_manifest(bundle, p1)
        write_metrics_manifest(bundle, p2)
        assert p1.read_bytes() == p2.read_bytes()

    def test_inconsistent_metric_dtype_across_records_raises(self, tmp_path):
        objects = (
            _object(1, metrics=[{"weird": 1}]),
            _object(2, metrics=[{"weird": "not a number"}]),
        )
        bundle = MeasurementBundle(objects=objects)
        with pytest.raises(ContractViolation, match="weird"):
            build_metrics_manifest(bundle)


# ─── io.tables: empty bundle ────────────────────────────────────────────────


class TestEmptyBundle:
    def test_empty_bundle_writes_headers_only_csvs(self, tmp_path):
        bundle = MeasurementBundle()
        paths = write_measurement_bundle(bundle, tmp_path)

        for key in ("objects", "contacts", "aggregates", "localization_profiles"):
            df = pd.read_csv(paths[key])
            assert len(df) == 0
            assert len(df.columns) > 0

    def test_empty_bundle_manifest_is_valid_json_with_no_metric_columns(self, tmp_path):
        bundle = MeasurementBundle()
        paths = write_measurement_bundle(bundle, tmp_path)
        manifest = json.loads(paths["metrics_manifest"].read_text(encoding="utf-8"))
        objects_names = [e["name"] for e in manifest["tables"]["objects.csv"]]
        assert "dataset_id" in objects_names
        assert len(objects_names) == 11  # only the fixed identity/QC columns


# ─── no heavy/ML imports ────────────────────────────────────────────────────


def test_importing_records_and_tables_does_not_import_torch_or_cellpose():
    code = (
        "import sys\n"
        "import s2_adhesion.metrics.records\n"
        "import s2_adhesion.io.tables\n"
        "assert 'torch' not in sys.modules, sys.modules.keys()\n"
        "assert 'cellpose' not in sys.modules, sys.modules.keys()\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


# ─── field_summary.csv (mixing index) ────────────────────────────────────────


def test_field_summary_csv_writes_mixing_columns(tmp_path):
    import pandas as pd

    from s2_adhesion.contracts import FieldSummaryRecord
    from s2_adhesion.io.tables import write_field_summary_csv

    rec = FieldSummaryRecord(
        dataset_id="d",
        field_id="f0",
        values={
            "mixing_index": 1.8,
            "heterotypic_fraction": 0.6,
            "n_qualifying_contacts": 10,
            "mixing_qc": None,
        },
    )
    path = tmp_path / "field_summary.csv"
    columns = write_field_summary_csv(rec, path)
    assert columns[:2] == ["dataset_id", "field_id"]
    assert "mixing_index" in columns

    df = pd.read_csv(path)
    assert len(df) == 1
    assert df.iloc[0]["mixing_index"] == 1.8
    assert df.iloc[0]["n_qualifying_contacts"] == 10
    # None serialises as an empty field, read back as NaN, never "None"/0.
    assert pd.isna(df.iloc[0]["mixing_qc"])


def test_field_summary_csv_none_writes_headers_only(tmp_path):
    import pandas as pd

    from s2_adhesion.io.tables import write_field_summary_csv

    path = tmp_path / "field_summary.csv"
    write_field_summary_csv(None, path)  # no populations configured
    df = pd.read_csv(path)
    assert len(df) == 0
    assert list(df.columns) == ["dataset_id", "field_id"]
