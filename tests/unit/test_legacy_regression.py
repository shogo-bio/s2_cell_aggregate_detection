"""Regression tests: the refactored legacy pipeline must be BEHAVIOURALLY
IDENTICAL to the pre-refactor ``detect_aggregations.py`` -- not merely
similar. That is the entire point of extracting it into
``s2_adhesion.legacy.algorithm``: the new 3D pipeline is only comparable
against the old one if the old one's numbers didn't quietly change underfoot.

The pre-refactor script is loaded straight out of git (``git show
HEAD:detect_aggregations.py``) into a throwaway module, so this test compares
against the actual committed original, not a hand-copied approximation of it.
"""

from __future__ import annotations

import csv
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np
import pytest

from s2_adhesion.backends.legacy import LegacyThreshold2DBackend
from s2_adhesion.commands import run_legacy as run_legacy_module
from s2_adhesion.config import LegacyConfig, PipelineConfig
from s2_adhesion.contracts import ImageVolume
from s2_adhesion.legacy import algorithm

from tests.conftest import make_image_volume

# ─── synthetic data ─────────────────────────────────────────────────────────
#
# One field, 3 channels, 5 Z-planes, 100x100 XY. Four disjoint disk-shaped
# "blobs" designed to exercise every preserved rule at once:
#
#   B (60, 60) r=10  ch0+ch1 active at every Z, incl. centre (z=2) -> ACCEPTED
#   S (20, 80) r=3   ch0+ch1 active at every Z, but area too small -> DROPPED
#                     (fails the area filter before the channel check even runs)
#   A (30, 30) r=10  ch0 active at every Z; ch1 active only at z=0 and z=4,
#                     i.e. NOT at the centre slice (z=2) -> REJECTED, because
#                     the preserved "centre-Z-slice-only" rule never sees ch1
#                     active here even though it genuinely lights up elsewhere
#                     in the stack.
#   E (0, 50)  r=10  ch0+ch1 active at every Z, clipped by the Y=0 edge of the
#                     volume -> ACCEPTED (the legacy algorithm has no concept
#                     of border truncation at all).
#
# ch2 is never touched (always background) in every blob, so it is always
# inactive -- a deliberate "always-off" channel alongside the two "on" ones.

_N_C, _N_Z, _N_Y, _N_X = 3, 5, 100, 100
_BACKGROUND = 10.0
_FOREGROUND = 220.0
_PIXEL_SIZE_UM = 1.0


def _disk(data: np.ndarray, ch: int, z: int, cy: int, cx: int, r: int) -> None:
    yy, xx = np.ogrid[:_N_Y, :_N_X]
    m = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
    data[ch, z][m] = _FOREGROUND


def _build_synthetic_stack() -> tuple[np.ndarray, dict[str, int], float]:
    data = np.full((_N_C, _N_Z, _N_Y, _N_X), _BACKGROUND, dtype=np.float32)

    for z in range(_N_Z):
        _disk(data, 0, z, 60, 60, 10)  # B
        _disk(data, 1, z, 60, 60, 10)

        _disk(data, 0, z, 20, 80, 3)  # S
        _disk(data, 1, z, 20, 80, 3)

        _disk(data, 0, z, 30, 30, 10)  # A (ch0 only, every z)

        _disk(data, 0, z, 0, 50, 10)  # E
        _disk(data, 1, z, 0, 50, 10)

    _disk(data, 1, 0, 30, 30, 10)  # A ch1, z=0 (not centre)
    _disk(data, 1, _N_Z - 1, 30, 30, 10)  # A ch1, z=4 (not centre)

    sizes = {"C": _N_C, "Z": _N_Z, "Y": _N_Y, "X": _N_X}
    return data, sizes, _PIXEL_SIZE_UM


def _run_pipeline(mod: ModuleType, data: np.ndarray, sizes: dict[str, int], pixel_size_um: float, cfg) -> dict:
    """Drive one module's (original or refactored) exposed functions through
    the same sequence ``process_field`` uses, keeping every intermediate so
    the test can assert on masks and the component table, not just the
    final CSV-facing records.
    """
    median_k = mod.compute_median_kernel_size(pixel_size_um, cfg.s2_diameter_um)
    global_minmax = mod.compute_global_channel_minmax(data, sizes)
    channel_projections = mod.extract_channel_projections(data, sizes, global_minmax)
    center_slices = mod.extract_center_slices(
        data, sizes, pixel_size_um, cfg.s2_diameter_um, global_minmax
    )
    gray_merged = mod.merge_channels(channel_projections)
    gray_merged_median = cv2.medianBlur(gray_merged, median_k)
    mask_binary, mask_filled, mask_final = mod.segment_occupancy(gray_merged_median, cfg, pixel_size_um)
    binary_center_slices = [
        cv2.threshold(slc, cfg.binary_threshold, 255, cv2.THRESH_BINARY)[1]
        for slc in center_slices
    ]
    area_threshold_px = mod.compute_area_threshold_px(
        pixel_size_um, cfg.s2_diameter_um, cfg.aggregation_min_cells
    )
    records, valid_mask, rejected_mask = mod.detect_aggregations(
        mask_final, binary_center_slices, pixel_size_um, 0,
        area_threshold_px, cfg.min_active_channels,
    )
    return {
        "global_minmax": global_minmax,
        "channel_projections": channel_projections,
        "center_slices": center_slices,
        "gray_merged": gray_merged,
        "gray_merged_median": gray_merged_median,
        "mask_binary": mask_binary,
        "mask_filled": mask_filled,
        "mask_final": mask_final,
        "records": records,
        "valid_mask": valid_mask,
        "rejected_mask": rejected_mask,
        "area_threshold_px": area_threshold_px,
    }


# ─── fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def original_algorithm(tmp_path_factory) -> ModuleType:
    """The pre-refactor ``detect_aggregations.py``, loaded straight from git."""
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "show", "HEAD:detect_aggregations.py"],
        cwd=repo_root, capture_output=True, text=True, encoding="utf-8", timeout=30,
    )
    assert result.returncode == 0, f"git show failed: {result.stderr}"
    assert result.stdout.strip(), "git show HEAD:detect_aggregations.py returned nothing"

    tmp_dir = tmp_path_factory.mktemp("legacy_regression")
    original_path = tmp_dir / "original_detect_aggregations.py"
    original_path.write_text(result.stdout, encoding="utf-8")

    spec = importlib.util.spec_from_file_location(
        "original_detect_aggregations_pre_refactor", original_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def synthetic_stack() -> tuple[np.ndarray, dict[str, int], float]:
    return _build_synthetic_stack()


# ─── the critical regression: masks and component tables are IDENTICAL ─────


def test_masks_and_records_identical_to_pre_refactor(original_algorithm, synthetic_stack):
    data, sizes, pixel_size_um = synthetic_stack

    cfg_orig = original_algorithm.Config(
        s2_diameter_um=10.0, aggregation_min_cells=1, morph_close_radius_um=1.65,
        binary_threshold=50, min_active_channels=2,
    )
    cfg_new = algorithm.Config(
        s2_diameter_um=10.0, aggregation_min_cells=1, morph_close_radius_um=1.65,
        binary_threshold=50, min_active_channels=2,
    )

    orig = _run_pipeline(original_algorithm, data, sizes, pixel_size_um, cfg_orig)
    new = _run_pipeline(algorithm, data, sizes, pixel_size_um, cfg_new)

    assert orig["global_minmax"] == new["global_minmax"]
    assert orig["area_threshold_px"] == new["area_threshold_px"]

    assert len(orig["channel_projections"]) == len(new["channel_projections"])
    for a, b in zip(orig["channel_projections"], new["channel_projections"]):
        assert np.array_equal(a, b)

    assert len(orig["center_slices"]) == len(new["center_slices"])
    for a, b in zip(orig["center_slices"], new["center_slices"]):
        assert np.array_equal(a, b)

    assert np.array_equal(orig["gray_merged"], new["gray_merged"])
    assert np.array_equal(orig["gray_merged_median"], new["gray_merged_median"])
    assert np.array_equal(orig["mask_binary"], new["mask_binary"])
    assert np.array_equal(orig["mask_filled"], new["mask_filled"])
    assert np.array_equal(orig["mask_final"], new["mask_final"])
    assert np.array_equal(orig["valid_mask"], new["valid_mask"])
    assert np.array_equal(orig["rejected_mask"], new["rejected_mask"])

    # RegionRecord is a NamedTuple in both modules; tuple equality compares
    # values only, so this is a genuine identity check across the two
    # independently-defined classes, not a type-blind coincidence.
    assert list(orig["records"]) == list(new["records"])


def test_synthetic_scenario_exercises_area_and_active_channel_rules(synthetic_stack):
    """Sanity check on the fixture itself: confirms the four designed blobs
    (accepted / area-dropped / centre-slice-rejected / border-touching
    accepted) actually produce the outcomes the docstring above claims, using
    only the refactored module. The cross-module identity is asserted
    separately above; this just proves the scenario is not vacuous.
    """
    data, sizes, pixel_size_um = synthetic_stack
    cfg = algorithm.Config(
        s2_diameter_um=10.0, aggregation_min_cells=1, morph_close_radius_um=1.65,
        binary_threshold=50, min_active_channels=2,
    )
    result = _run_pipeline(algorithm, data, sizes, pixel_size_um, cfg)

    # B and E accepted; S dropped by area before any channel check; A
    # rejected by the centre-Z-slice-only active-channel rule.
    assert len(result["records"]) == 2
    for r in result["records"]:
        assert r.area_px >= result["area_threshold_px"]
        assert r.active_channels >= cfg.min_active_channels

    # A passes the area filter (radius 10 disk) but fails the channel check,
    # so it must show up in rejected_mask, not simply vanish.
    n_rejected_components, _ = cv2.connectedComponents(result["rejected_mask"])
    assert n_rejected_components - 1 == 1  # background + exactly one component

    # S is smaller than the area threshold and is dropped via `continue`
    # before ever reaching the active-channel check or rejected_mask --
    # i.e. it appears in neither valid_mask nor rejected_mask.
    combined = np.maximum(result["valid_mask"], result["rejected_mask"])
    yy, xx = np.ogrid[:_N_Y, :_N_X]
    s_region = (yy - 20) ** 2 + (xx - 80) ** 2 <= 3 ** 2
    assert not combined[s_region].any()


def test_area_threshold_formula_matches_min_cells_times_cell_area(original_algorithm):
    pixel_size_um = 0.37
    s2_diameter_um = 8.5
    min_cells = 4
    expected = min_cells * np.pi * (s2_diameter_um / 2.0) ** 2 / (pixel_size_um ** 2)

    assert algorithm.compute_area_threshold_px(
        pixel_size_um, s2_diameter_um, min_cells
    ) == pytest.approx(expected)
    assert original_algorithm.compute_area_threshold_px(
        pixel_size_um, s2_diameter_um, min_cells
    ) == pytest.approx(expected)


# ─── CLI surface preserved ──────────────────────────────────────────────────


def test_cli_flags_and_defaults_match_pre_refactor(original_algorithm, tmp_path):
    common_args = [str(tmp_path / "in.nd2"), str(tmp_path / "out")]
    orig_ns = vars(original_algorithm.build_parser().parse_args(common_args))
    new_ns = vars(algorithm.build_parser().parse_args(common_args))
    assert orig_ns == new_ns


def test_run_legacy_reuses_algorithm_build_parser_exactly():
    # Guarantees flags/defaults can never drift between the two modules --
    # they are, by construction, the exact same argparse.ArgumentParser
    # factory function, not two hand-kept-in-sync copies.
    assert run_legacy_module.build_parser is algorithm.build_parser


def test_run_legacy_defaults_match_config_defaults():
    defaults = algorithm.Config()
    parsed = algorithm.build_parser().parse_args(["in.nd2", "out"])
    assert parsed.s2_diameter == defaults.s2_diameter_um
    assert parsed.min_cells == defaults.aggregation_min_cells
    assert parsed.morph_close_radius == defaults.morph_close_radius_um
    assert parsed.binary_threshold == defaults.binary_threshold
    assert parsed.min_active_channels == defaults.min_active_channels
    assert parsed.debug is False


# ─── CSV columns preserved ──────────────────────────────────────────────────


def test_csv_columns_unchanged(tmp_path):
    records = [
        algorithm.RegionRecord(
            field_id=0, aggregation_id=0, area_px=123, area_um2=45.6, active_channels=2
        ),
    ]
    out_path = tmp_path / "field_aggregations.csv"
    algorithm.save_csv(records, out_path)

    with out_path.open(newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))

    assert header == ["field_id", "aggregation_id", "area_px", "area_um2", "active_channels"]


# ─── no ML stack pulled in ──────────────────────────────────────────────────


def test_legacy_modules_importable_without_torch_or_cellpose():
    """Fresh interpreter: importing the legacy modules must not pull in
    torch/cellpose. Checked in a subprocess with a clean sys.modules, since
    this test-runner process already has both loaded by other tests.
    """
    script = (
        "import sys\n"
        "from s2_adhesion.legacy import algorithm\n"
        "from s2_adhesion.backends import legacy\n"
        "from s2_adhesion.commands import run_legacy\n"
        "assert 'torch' not in sys.modules, sys.modules.keys()\n"
        "assert 'cellpose' not in sys.modules, sys.modules.keys()\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


# ─── console encoding guard ─────────────────────────────────────────────────
#
# Confirmed separately (against `git show HEAD:detect_aggregations.py` run
# directly) that on a cp932 (Japanese Windows) console, even `--help` crashes
# with UnicodeEncodeError, because every help string quoting a physical unit
# contains 'µm'. `s2_adhesion._console.ensure_utf8_output` (owned by nobody
# here, not edited by this module) fixes that; both `main()` entry points
# this file owns call it as their first statement. These tests exercise the
# fix deterministically -- by monkeypatching stdout to a real cp932
# TextIOWrapper inside a subprocess -- rather than depending on the actual
# host machine's code page.

_CP932_HELP_SCRIPT_TEMPLATE = """
import io
import sys

buf = io.BytesIO()
sys.stdout = io.TextIOWrapper(buf, encoding="cp932", errors="strict")
sys.argv = ["prog", "--help"]

from {module} import main

try:
    main()
except SystemExit:
    pass

sys.stdout.flush()
text = buf.getvalue().decode("utf-8")
assert "\\u00b5m" in text, repr(text)
print("HELP_OK", file=sys.stderr)
"""


@pytest.mark.parametrize(
    "module",
    ["s2_adhesion.legacy.algorithm", "s2_adhesion.commands.run_legacy"],
)
def test_help_survives_cp932_console_via_utf8_guard(module):
    """`--help` must succeed, and its output must still contain 'µm' (not a
    dropped/mangled character), even when stdout cannot natively encode it.
    The guard reconfigures the stream so the character survives; it must not
    just swallow the crash and lose the unit.
    """
    script = _CP932_HELP_SCRIPT_TEMPLATE.format(module=module)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, encoding="utf-8", timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "HELP_OK" in result.stderr


# ─── backends.legacy: ObjectRecord emission and border-truncation isolation ─


class _FakeVolumeSource:
    def __init__(self, fields: dict[str, ImageVolume]) -> None:
        self._fields = fields

    def field_ids(self):
        return list(self._fields.keys())

    def read_field(self, field_id: str) -> ImageVolume:
        return self._fields[field_id]


def test_backend_emits_object_records_matching_direct_pipeline_and_keeps_border_aggregates(
    synthetic_stack, tmp_path
):
    """LegacyThreshold2DBackend must (a) reproduce the same accepted regions
    as calling the algorithm directly, and (b) never let a border-touching
    aggregate (blob E, clipped by the Y=0 edge) be dropped or altered --
    the legacy algorithm has no border-truncation concept at all, and the
    new 3D-pipeline border rules (owned elsewhere) must not leak into it.
    """
    data, sizes, pixel_size_um = synthetic_stack
    cfg = algorithm.Config(
        s2_diameter_um=10.0, aggregation_min_cells=1, morph_close_radius_um=1.65,
        binary_threshold=50, min_active_channels=2,
    )
    direct = _run_pipeline(algorithm, data, sizes, pixel_size_um, cfg)

    vol = make_image_volume(
        data.astype(np.float32),
        spacing=(1.0, pixel_size_um, pixel_size_um),
        channel_ids=("membrane", "signal", "nucleus"),
    )
    source = _FakeVolumeSource({vol.identity.field_id: vol})

    pipeline_config = PipelineConfig(
        channels=vol.channels,
        analysis_backend="legacy_threshold_2d",
        legacy=LegacyConfig(
            threshold_uint8=cfg.binary_threshold,
            s2_diameter_um=cfg.s2_diameter_um,
            minimum_cell_equivalents=float(cfg.aggregation_min_cells),
            closing_radius_um=cfg.morph_close_radius_um,
            minimum_active_channels=cfg.min_active_channels,
        ),
    )

    backend = LegacyThreshold2DBackend()
    assert backend.backend_id == "legacy_threshold_2d"

    artifacts = backend.run(source, config=pipeline_config, output_dir=tmp_path / "legacy_run")

    objects = artifacts.bundle.objects
    assert len(objects) == len(direct["records"]) == 2

    direct_areas = sorted(r.area_px for r in direct["records"])
    backend_areas = sorted(o.values["area_px"] for o in objects)
    assert direct_areas == backend_areas

    for o in objects:
        assert o.backend_id == "legacy_threshold_2d"
        assert o.object_kind == "legacy_aggregate_2d"
        assert o.valid_for_geometry is False
        assert o.touches_z_border is True

    # Exactly one of the two accepted objects is blob E, clipped at the Y=0
    # edge -- it must be present AND flagged as touching the XY border, not
    # excluded because of it.
    border_flags = sorted(o.touches_xy_border for o in objects)
    assert border_flags == [False, True]

    assert artifacts.backend_id == "legacy_threshold_2d"
    assert set(artifacts.written_paths) == {
        "objects", "contacts", "aggregates", "localization_profiles",
        "field_summary", "metrics_manifest",
    }
    objects_csv = artifacts.written_paths["objects"]
    assert objects_csv.exists()
    with objects_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert all(row["object_kind"] == "legacy_aggregate_2d" for row in rows)
