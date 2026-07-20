"""End-to-end validation on synthetic data with an analytically known answer.

Segmentation is deliberately stubbed by writing label volumes directly. The
point of this file is the measurement chain and the artifact boundary, not
cellpose — and stubbing is what lets the whole thing run with no `.nd2` file and
no ML packages, which is exactly the configuration a measurement machine has.

Ground truth throughout: two spheres of radius 5 µm whose centres are 8 µm
apart, each voxel assigned to its nearer centre. That makes the interface the
flat radical plane, so both the truncated-sphere volume and the contact disc
area are known in closed form.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

from s2_adhesion.commands.measure import measure
from s2_adhesion.config import load_config
from s2_adhesion.contracts import (
    ChannelBinding,
    ChannelRole,
    FieldIdentity,
    ImageVolume,
    LabelVolume,
    SegmentationProvenance,
    VoxelGeometry,
)
from s2_adhesion.errors import ArtifactBindingError
from s2_adhesion.io.zarr_store import write_image_volume, write_label_volume

R_UM = 5.0
D_UM = 8.0
ANISO = (0.5, 0.1, 0.1)
ISO = (0.2, 0.2, 0.2)

CONFIG_YAML = """
schema_version: s2-pipeline-config/v1
analysis_backend: ml_instance_3d
channels:
  - {channel_id: membrane, source_index: 0, roles: [membrane]}
  - {channel_id: signal, source_index: 1, roles: [signal]}
segmentation:
  strategy: direct_cellpose
  input_channel_ids: [membrane]
measurement:
  contact:
    minimum_contact_area_um2: 0.5
artifacts:
  format: zarr
"""


def analytic_truncated_sphere_volume_um3(r: float = R_UM, d: float = D_UM) -> float:
    """Sphere minus the cap cut off by the bisecting plane."""
    h = r - d / 2.0
    cap = np.pi * h**2 * (3 * r - h) / 3.0
    return 4.0 / 3.0 * np.pi * r**3 - cap


def analytic_contact_disc_um2(r: float = R_UM, d: float = D_UM) -> float:
    return np.pi * (r**2 - (d / 2.0) ** 2)


def bisected_pair(spacing, shape, clip_last_z: bool = False) -> np.ndarray:
    """Two spheres split by their perpendicular bisector, separated along x."""
    zz, yy, xx = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]].astype(np.float64)
    zz *= spacing[0]
    yy *= spacing[1]
    xx *= spacing[2]
    cz = shape[0] * spacing[0] / 2.0
    cy = shape[1] * spacing[1] / 2.0
    cx = shape[2] * spacing[2] / 2.0
    if clip_last_z:
        # Push the pair down so it is cut by the final Z plane.
        cz = shape[0] * spacing[0] - R_UM / 2.0
    d1 = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - (cx - D_UM / 2)) ** 2
    d2 = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - (cx + D_UM / 2)) ** 2
    inside = (d1 < R_UM**2) | (d2 < R_UM**2)
    # Non-consecutive ids on purpose: nothing may silently renumber them.
    return np.where(inside, np.where(d1 <= d2, 1, 7), 0).astype(np.uint32)


def _write_pair(tmp_path, spacing, shape, *, clip_last_z=False, break_binding=False):
    labels_arr = bisected_pair(spacing, shape, clip_last_z=clip_last_z)

    data = np.zeros((2, *shape), np.float32)
    data[0][labels_arr > 0] = 1000.0          # membrane
    data[1][labels_arr == 1] = 500.0          # signal, in one cell only
    data += 10.0                              # uniform background

    identity = FieldIdentity(
        dataset_id="synthetic",
        field_id="field00",
        source_uri="memory://synthetic",
        source_field_index=0,
        image_content_sha256="0" * 64,
    )
    geom = VoxelGeometry(spacing_um_zyx=spacing)
    image = ImageVolume(
        data=data,
        geometry=geom,
        channels=(
            ChannelBinding("membrane", 0, frozenset({ChannelRole.MEMBRANE})),
            ChannelBinding("signal", 1, frozenset({ChannelRole.SIGNAL})),
        ),
        identity=identity,
    )
    image_dir = tmp_path / "image.ome.zarr"
    write_image_volume(image, image_dir, chunks=(1, 4, 64, 64))
    real_sha = image.identity.image_content_sha256

    labels = LabelVolume(
        cells=labels_arr,
        geometry=geom,
        identity=identity,
        provenance=SegmentationProvenance(
            run_id="run0",
            backend_id="synthetic",
            strategy="synthetic",
            config_sha256="c" * 64,
            input_image_sha256=("f" * 64) if break_binding else real_sha,
            device="cpu",
            host_platform="test",
        ),
    )
    label_dir = tmp_path / "labels.ome.zarr"
    write_label_volume(labels, label_dir, chunks=(4, 64, 64))
    return image_dir, label_dir


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text(textwrap.dedent(CONFIG_YAML), encoding="utf-8")
    return load_config(path)


def _read(out_dir, name):
    path = out_dir / name
    assert path.exists(), f"{name} was not written"
    return pd.read_csv(path)


class TestFullChain:
    def test_measures_two_cells_and_one_contact_with_known_values(
        self, tmp_path, config
    ):
        shape = (28, 140, 200)
        image_dir, label_dir = _write_pair(tmp_path, ANISO, shape)
        out = tmp_path / "out"
        measure(label_dir, out, config, image_artifact_dir=image_dir)

        objects = _read(out, "objects.csv")
        assert len(objects) == 2
        assert sorted(objects["object_id"]) == [1, 7], "label ids were renumbered"

        want_v = analytic_truncated_sphere_volume_um3()
        for volume in objects["volume_um3"]:
            assert volume == pytest.approx(want_v, rel=0.05)

        contacts = _read(out, "contacts.csv")
        assert len(contacts) == 1
        row = contacts.iloc[0]
        assert row["cell_id_a"] < row["cell_id_b"]

        aggregates = _read(out, "aggregates.csv")
        assert len(aggregates) == 1
        assert aggregates.iloc[0]["aggregate_cell_count"] == 2

        assert len(_read(out, "localization_profiles.csv")) > 0

    def test_contact_area_is_in_the_documented_range(self, tmp_path, config):
        """Loose on purpose.

        Contact area is orientation-dependent under anisotropy — measured
        6.3% to 45.9% error depending on how the interface sits relative to the
        optical axis. This asserts the documented bound, not a tight tolerance.
        Do NOT tighten it: see docs/metric_interpretation.md for why the error
        is a property of the optics, not of the implementation.
        """
        shape = (28, 140, 200)
        image_dir, label_dir = _write_pair(tmp_path, ANISO, shape)
        out = tmp_path / "out"
        measure(label_dir, out, config, image_artifact_dir=image_dir)

        got = _read(out, "contacts.csv").iloc[0]["contact_area_um2"]
        want = analytic_contact_disc_um2()
        assert want * 0.5 < got < want * 1.6


class TestArchitecturalGuarantees:
    def test_measure_completes_with_torch_and_cellpose_unimportable(
        self, tmp_path, config
    ):
        """The guarantee the whole split-machine design rests on.

        A hard import blocker, not a sys.modules check: a laptop doing the
        measuring genuinely does not have these packages.
        """
        shape = (28, 140, 200)
        image_dir, label_dir = _write_pair(tmp_path, ANISO, shape)
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(textwrap.dedent(CONFIG_YAML), encoding="utf-8")
        out = tmp_path / "out_blocked"

        script = textwrap.dedent(
            f"""
            import sys
            class Blocker:
                def find_module(self, name, path=None):
                    return self if name.split('.')[0] in ('torch','cellpose') else None
                def load_module(self, name):
                    raise ImportError('blocked: ' + name)
            sys.meta_path.insert(0, Blocker())

            from pathlib import Path
            from s2_adhesion.config import load_config
            from s2_adhesion.commands.measure import measure
            measure(Path(r"{label_dir}"), Path(r"{out}"),
                    load_config(Path(r"{cfg_path}")),
                    image_artifact_dir=Path(r"{image_dir}"))
            assert 'torch' not in sys.modules and 'cellpose' not in sys.modules
            print("BLOCKED_RUN_OK")
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True
        )
        assert "BLOCKED_RUN_OK" in result.stdout, (
            f"measure failed with ML blocked.\nstdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
        assert len(_read(out, "objects.csv")) == 2

    def test_geometry_only_without_an_image_artifact(self, tmp_path, config):
        shape = (28, 140, 200)
        _, label_dir = _write_pair(tmp_path, ANISO, shape)
        out = tmp_path / "out"
        measure(label_dir, out, config)  # no image

        objects = _read(out, "objects.csv")
        assert len(objects) == 2
        assert objects["volume_um3"].notna().all(), "geometry must still be measured"

        intensity_cols = [c for c in objects.columns if c.startswith("ch.")]
        for col in intensity_cols:
            assert objects[col].isna().all(), f"{col} must be null with no image"

    def test_mismatched_labels_write_nothing_at_all(self, tmp_path, config):
        """A partial CSV from a mismatched pair is worse than no output."""
        shape = (28, 140, 200)
        image_dir, label_dir = _write_pair(tmp_path, ANISO, shape, break_binding=True)
        out = tmp_path / "out"
        out.mkdir()

        with pytest.raises(ArtifactBindingError):
            measure(label_dir, out, config, image_artifact_dir=image_dir)

        assert list(out.iterdir()) == [], "output written despite a binding failure"

    def test_measurement_is_deterministic(self, tmp_path, config):
        shape = (28, 140, 200)
        image_dir, label_dir = _write_pair(tmp_path, ANISO, shape)
        first, second = tmp_path / "a", tmp_path / "b"
        measure(label_dir, first, config, image_artifact_dir=image_dir)
        measure(label_dir, second, config, image_artifact_dir=image_dir)

        for name in ("objects.csv", "contacts.csv", "aggregates.csv"):
            assert (first / name).read_bytes() == (second / name).read_bytes(), (
                f"{name} differs between two runs on identical input"
            )


class TestTruncation:
    def test_clipped_cell_keeps_observed_but_nulls_canonical_geometry(
        self, tmp_path, config
    ):
        shape = (16, 140, 200)
        image_dir, label_dir = _write_pair(
            tmp_path, ANISO, shape, clip_last_z=True
        )
        out = tmp_path / "out"
        measure(label_dir, out, config, image_artifact_dir=image_dir)

        objects = _read(out, "objects.csv")
        clipped = objects[objects["touches_z_border"]]
        assert len(clipped) > 0, "fixture did not actually clip anything"

        assert clipped["volume_um3_observed"].notna().all()
        assert clipped["volume_um3"].isna().all(), (
            "a cell cut off by the volume edge has no meaningful volume"
        )
        # A zero here would silently drag down any downstream mean.
        assert not (clipped["volume_um3"] == 0).any()

        aggregates = _read(out, "aggregates.csv")
        truncated_aggs = aggregates[aggregates["contains_truncated_cell"]]
        assert truncated_aggs["packing_fraction"].isna().all()


class TestAnisotropyConsistency:
    def test_volume_agrees_between_samplings(self, tmp_path, config):
        """Volume is the metric that must survive anisotropy — and does (~0.4%).

        Contact and surface area are deliberately NOT asserted here; they are
        orientation-dependent by tens of percent. See docs/metric_interpretation.md.
        """
        aniso_dir = tmp_path / "aniso"
        iso_dir = tmp_path / "iso"
        aniso_dir.mkdir()
        iso_dir.mkdir()

        _, lab_a = _write_pair(aniso_dir, ANISO, (28, 140, 200))
        _, lab_i = _write_pair(iso_dir, ISO, (70, 70, 100))

        out_a, out_i = tmp_path / "oa", tmp_path / "oi"
        measure(lab_a, out_a, config)
        measure(lab_i, out_i, config)

        va = _read(out_a, "objects.csv")["volume_um3"].mean()
        vi = _read(out_i, "objects.csv")["volume_um3"].mean()
        assert va == pytest.approx(vi, rel=0.05)
