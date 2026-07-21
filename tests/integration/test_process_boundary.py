"""Integration tests for the segment/measure process-boundary architecture.

``commands.measure`` has to work as a genuinely separate program from
segmentation: labels + (optionally) an image are read from on-disk
artifacts, metrics are computed, and CSVs are written -- nothing here may
depend on anything from the same process that produced the labels. These
tests build an image artifact and a synthetic (hand-built, no ML) label
artifact in THIS process, then exercise ``measure`` both in-process and, for
the one criterion that specifically demands it, in a brand-new subprocess.

The complementary "does this actually work with torch/cellpose made
unimportable" guarantee lives in ``test_measure_without_ml.py`` -- that is
the load-bearing test for the whole split-machine architecture and gets its
own file because it needs its own import-blocking hook.
"""

from __future__ import annotations

import hashlib
import inspect
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from s2_adhesion.config import (
    ArtifactConfig,
    ContactConfig,
    DirectInstanceConfig,
    MeasurementConfig,
    PipelineConfig,
)
from s2_adhesion.contracts import (
    ChannelBinding,
    ChannelRole,
    FieldIdentity,
    ImageVolume,
    LabelVolume,
    SegmentationProvenance,
    VoxelGeometry,
)
from s2_adhesion.commands.measure import measure
from s2_adhesion.errors import ArtifactBindingError
from s2_adhesion.io.zarr_store import write_image_volume, write_label_volume

from tests.synthetic_volumes import sphere_pair_bisected

REPO_ROOT = Path(__file__).resolve().parents[2]

# Geometry chosen so both cells sit well clear of the volume border (no
# truncation -- canonical volume/surface fields are populated, not nulled)
# while still touching each other with a real, non-trivial interface.
SPACING = (0.5, 0.25, 0.25)
SHAPE = (24, 40, 60)
CENTRE_UM = (6.0, 5.0, 7.5)
RADIUS_UM = 3.0
SEPARATION_UM = 4.0
# pi * (radius^2 - (separation/2)^2), the flat radical-plane interface area
# of two equal touching balls -- see tests/synthetic_volumes.py.
ANALYTIC_CONTACT_AREA_UM2 = np.pi * (RADIUS_UM**2 - (SEPARATION_UM / 2.0) ** 2)


def _config() -> PipelineConfig:
    channels = (
        ChannelBinding(
            channel_id="membrane", source_index=0, roles=frozenset({ChannelRole.MEMBRANE})
        ),
        ChannelBinding(
            channel_id="nucleus", source_index=1, roles=frozenset({ChannelRole.NUCLEUS})
        ),
    )
    return PipelineConfig(
        channels=channels,
        analysis_backend="ml_instance_3d",
        segmentation=DirectInstanceConfig(
            strategy="direct_cellpose", input_channel_ids=("membrane",)
        ),
        measurement=MeasurementConfig(contact=ContactConfig(minimum_contact_area_um2=0.5)),
        artifacts=ArtifactConfig(),
    )


def _build_pair(
    tmp_path: Path,
    *,
    subdir: str,
    wrong_hash: bool = False,
    wrong_shape: bool = False,
    wrong_spacing: bool = False,
) -> tuple[Path, Path]:
    """Write a matching image artifact + label artifact (two touching cells).

    ``wrong_hash``/``wrong_shape``/``wrong_spacing`` each independently break
    exactly one axis of the label/image binding, leaving the others intact,
    so a test can assert precisely which check fired.
    """
    root = tmp_path / subdir

    cells = sphere_pair_bisected(
        centre_um=CENTRE_UM,
        radius_um=RADIUS_UM,
        separation_um=SEPARATION_UM,
        shape=SHAPE,
        spacing=SPACING,
        orientation="axis_aligned",
    ).astype(np.uint32)

    rng = np.random.default_rng(0)
    data = (rng.random((2, *SHAPE)) * 500.0).astype(np.float32)
    data[0][cells > 0] += 3000.0  # real, non-degenerate membrane signal inside both cells

    content_hash = hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()
    image_identity = FieldIdentity(
        dataset_id="ds0",
        field_id="field00",
        source_uri="memory://synthetic",
        source_field_index=0,
        image_content_sha256=content_hash,
    )
    channels = (
        ChannelBinding(
            channel_id="membrane", source_index=0, roles=frozenset({ChannelRole.MEMBRANE})
        ),
        ChannelBinding(
            channel_id="nucleus", source_index=1, roles=frozenset({ChannelRole.NUCLEUS})
        ),
    )
    image = ImageVolume(
        data=data,
        geometry=VoxelGeometry(spacing_um_zyx=SPACING),
        channels=channels,
        identity=image_identity,
    )

    label_cells = cells[:, :, :-2] if wrong_shape else cells
    label_spacing = (0.5, 0.5, 0.5) if wrong_spacing else SPACING
    input_image_sha256 = ("0" * 64) if wrong_hash else image_identity.image_content_sha256

    label_identity = FieldIdentity(
        dataset_id=image_identity.dataset_id,
        field_id=image_identity.field_id,
        source_uri=image_identity.source_uri,
        source_field_index=image_identity.source_field_index,
        image_content_sha256=image_identity.image_content_sha256,
    )
    provenance = SegmentationProvenance(
        run_id="run-test",
        backend_id="direct_cellpose",
        strategy="direct_cellpose",
        config_sha256=hashlib.sha256(b"cfg").hexdigest(),
        input_image_sha256=input_image_sha256,
        device="cpu",
        host_platform="test",
    )
    labels = LabelVolume(
        cells=label_cells,
        geometry=VoxelGeometry(spacing_um_zyx=label_spacing),
        identity=label_identity,
        provenance=provenance,
    )

    image_dir = write_image_volume(image, root / "image.ome.zarr", chunks=image.data.shape)
    label_dir = write_label_volume(labels, root / "labels.ome.zarr", chunks=label_cells.shape)
    return image_dir, label_dir


def _build_bound_pair(tmp_path: Path, *, subdir: str = "pair") -> tuple[Path, Path]:
    return _build_pair(tmp_path, subdir=subdir)


def _measure_script(label_dir: Path, output_dir: Path, image_dir: Path | None) -> str:
    """A self-contained ``python -c`` script that calls ``measure`` once.

    Embeds ``_config``'s actual source (via ``inspect.getsource``) rather
    than duplicating its logic by hand, so the subprocess config can never
    silently drift from what ``_config()`` really builds in-process.
    """
    image_expr = f"Path({str(image_dir)!r})" if image_dir is not None else "None"
    return textwrap.dedent(
        f"""
        import sys
        from pathlib import Path
        sys.path.insert(0, {str(REPO_ROOT)!r})

        from s2_adhesion.config import (
            ArtifactConfig, ContactConfig, DirectInstanceConfig, MeasurementConfig, PipelineConfig,
        )
        from s2_adhesion.contracts import ChannelBinding, ChannelRole
        from s2_adhesion.commands.measure import measure

        {textwrap.indent(inspect.getsource(_config), "        ").strip()}

        config = _config()
        result = measure(
            Path({str(label_dir)!r}),
            Path({str(output_dir)!r}),
            config,
            image_artifact_dir={image_expr},
        )
        print("MEASURE_OK", result.output_dir)
        """
    )


def _run_python(script: str, *, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=timeout,
    )


# ─── the process boundary, exercised for real ──────────────────────────────


def test_measure_runs_in_separate_subprocess_and_produces_real_numbers(tmp_path):
    """Build artifacts in THIS process; measure them in a brand-new one."""
    image_dir, label_dir = _build_bound_pair(tmp_path)
    output_dir = tmp_path / "measure_out"

    script = _measure_script(label_dir, output_dir, image_dir)
    result = _run_python(script)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "MEASURE_OK" in result.stdout

    objects = pd.read_csv(output_dir / "objects.csv")
    contacts = pd.read_csv(output_dir / "contacts.csv")
    aggregates = pd.read_csv(output_dir / "aggregates.csv")

    assert set(objects["object_id"]) == {1, 2}
    assert (objects["touches_xy_border"] == False).all()  # noqa: E712
    assert (objects["touches_z_border"] == False).all()  # noqa: E712
    assert (objects["valid_for_geometry"] == True).all()  # noqa: E712
    # Real, non-null, sanely-bounded physical numbers -- not zeros, not NaN.
    assert objects["volume_um3"].notna().all()
    assert objects["volume_um3"].between(20.0, 300.0).all()
    assert objects["ch.membrane.raw_mean"].notna().all()
    assert (objects["ch.membrane.raw_mean"] > 1000.0).all()  # signal was bumped by 3000

    assert len(contacts) == 1
    row = contacts.iloc[0]
    assert bool(row["qualifies_as_contact"]) is True
    # MARCHING_CUBES carries a documented, bounded bias vs the analytic flat
    # interface (contracts.ContactEstimator's docstring); a generous bracket
    # around the analytic value confirms a real measurement, not a stub.
    assert ANALYTIC_CONTACT_AREA_UM2 * 0.5 < row["contact_area_um2"] < ANALYTIC_CONTACT_AREA_UM2 * 2.0

    assert len(aggregates) == 1
    agg = aggregates.iloc[0]
    assert agg["member_cell_ids"] == "1;2"
    assert bool(agg["contains_truncated_cell"]) is False


# ─── geometry-only (no image artifact) ─────────────────────────────────────


def test_geometry_only_measure_succeeds_without_image_artifact(tmp_path):
    _, label_dir = _build_bound_pair(tmp_path)
    config = _config()
    output_dir = tmp_path / "geo_out"

    result = measure(label_dir, output_dir, config, image_artifact_dir=None)

    objects = pd.read_csv(result.objects_csv)
    contacts = pd.read_csv(result.contacts_csv)
    aggregates = pd.read_csv(result.aggregates_csv)
    localization = pd.read_csv(result.localization_profiles_csv)

    # volume/contact/aggregate columns: populated, real numbers.
    assert objects["volume_um3"].notna().all()
    assert objects["volume_um3"].between(20.0, 300.0).all()
    assert len(contacts) == 1
    assert bool(contacts.iloc[0]["qualifies_as_contact"]) is True
    assert len(aggregates) == 1

    # intensity/localization: no per-channel columns can even be named
    # without an image, and the localization table has no rows at all --
    # both are "empty" in the sense io.tables' data-driven column set makes
    # available.
    assert not any(c.startswith("ch.") for c in objects.columns)
    assert len(localization) == 0

    # ... with the reason documented on every object row and in the bundle's
    # warnings, per metrics.engine.MISSING_IMAGE_REASON.
    assert (objects["intensity_missing_reason"] == "missing_image_artifact").all()
    assert (objects["nuclei_missing_reason"] == "missing_image_artifact").all()
    assert (objects["localization_missing_reason"] == "missing_image_artifact").all()
    assert any(w.startswith("missing_image_artifact") for w in result.warnings)


# ─── binding guard: fails before any write ─────────────────────────────────


def test_mismatched_image_hash_raises_and_writes_nothing(tmp_path):
    image_dir, label_dir = _build_pair(tmp_path, subdir="hash_mismatch", wrong_hash=True)
    config = _config()
    output_dir = tmp_path / "hash_mismatch" / "measure_out"

    with pytest.raises(ArtifactBindingError):
        measure(label_dir, output_dir, config, image_artifact_dir=image_dir)

    assert not output_dir.exists()


def test_shape_mismatch_raises_before_any_write(tmp_path):
    image_dir, label_dir = _build_pair(tmp_path, subdir="shape_mismatch", wrong_shape=True)
    config = _config()
    output_dir = tmp_path / "shape_mismatch" / "measure_out"

    with pytest.raises(ArtifactBindingError, match="shape"):
        measure(label_dir, output_dir, config, image_artifact_dir=image_dir)

    assert not output_dir.exists()


def test_spacing_mismatch_raises_before_any_write(tmp_path):
    image_dir, label_dir = _build_pair(tmp_path, subdir="spacing_mismatch", wrong_spacing=True)
    config = _config()
    output_dir = tmp_path / "spacing_mismatch" / "measure_out"

    with pytest.raises(ArtifactBindingError, match="spacing"):
        measure(label_dir, output_dir, config, image_artifact_dir=image_dir)

    assert not output_dir.exists()


# ─── determinism + "no segmentation ever happened" ─────────────────────────


def test_rerun_is_byte_identical_and_touches_no_segmentation_modules(tmp_path):
    """Two measure() calls in ONE fresh subprocess (so sys.modules isn't
    polluted by anything else this test session has already imported):
    identical CSV bytes, and no segmentation/ML module ever got imported.
    """
    _, label_dir = _build_bound_pair(tmp_path)
    out_a = tmp_path / "run_a"
    out_b = tmp_path / "run_b"

    script = textwrap.dedent(
        f"""
        import sys
        from pathlib import Path
        sys.path.insert(0, {str(REPO_ROOT)!r})

        from s2_adhesion.config import (
            ArtifactConfig, ContactConfig, DirectInstanceConfig, MeasurementConfig, PipelineConfig,
        )
        from s2_adhesion.contracts import ChannelBinding, ChannelRole
        from s2_adhesion.commands.measure import measure

        {textwrap.indent(inspect.getsource(_config), "        ").strip()}

        config = _config()
        measure(Path({str(label_dir)!r}), Path({str(out_a)!r}), config, image_artifact_dir=None)
        measure(Path({str(label_dir)!r}), Path({str(out_b)!r}), config, image_artifact_dir=None)

        seg_modules = [m for m in sys.modules if m.startswith("s2_adhesion.segmentation")]
        assert seg_modules == [], seg_modules
        assert "torch" not in sys.modules, sys.modules.keys()
        assert "cellpose" not in sys.modules, sys.modules.keys()
        print("DETERMINISM_OK")
        """
    )
    result = _run_python(script)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "DETERMINISM_OK" in result.stdout

    for name in (
        "objects.csv",
        "contacts.csv",
        "aggregates.csv",
        "localization_profiles.csv",
        "metrics_manifest.json",
    ):
        a_bytes = (out_a / name).read_bytes()
        b_bytes = (out_b / name).read_bytes()
        assert a_bytes == b_bytes, f"{name} differs between identical runs"


# ─── atomicity: a mid-write failure never leaves a manifest claiming success ─


def test_partial_write_failure_leaves_no_run_manifest_claiming_success(tmp_path, monkeypatch):
    _, label_dir = _build_bound_pair(tmp_path)
    config = _config()
    output_dir = tmp_path / "atomic_out"

    def _boom(bundle, out_dir):
        # Simulate io.tables.write_measurement_bundle making real progress
        # (a file genuinely lands on disk) before failing partway through --
        # the strongest version of "partial write" to guard against.
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "objects.csv").write_text("partial-should-not-survive", encoding="utf-8")
        raise RuntimeError("simulated failure mid-write")

    monkeypatch.setattr("s2_adhesion.commands.measure.write_measurement_bundle", _boom)

    with pytest.raises(RuntimeError, match="simulated failure"):
        measure(label_dir, output_dir, config, image_artifact_dir=None)

    assert not output_dir.exists()
    # and no orphaned temp directory left behind either
    assert list(output_dir.parent.glob(f".{output_dir.name}.tmp-*")) == []
