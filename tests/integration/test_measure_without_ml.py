"""THE load-bearing test for the whole split-machine architecture.

The reason ``commands.segment``/``commands.measure`` exist as separate
programs at all: cellpose 4 needed 638 s for a single 256x256 plane on the
reference CPU host (cellpose 3 is roughly 290x faster but still slow), so
segmentation runs on a GPU machine and measurement runs on a laptop or CI
runner with NO torch/cellpose installed. If ``measure`` secretly needed
either package, that architecture would be a lie.

This is tested FOR REAL here, not by checking ``sys.modules`` in a process
that happens not to have imported them (that would pass even if
``commands.measure`` tried and failed to import them silently, or if the
test host simply never had them installed -- neither proves the claim).
Instead this file launches a subprocess with a ``sitecustomize.py`` on
``PYTHONPATH`` that installs a ``sys.meta_path`` import hook making
``import torch`` and ``import cellpose`` (and every submodule of either)
raise ``ImportError`` unconditionally -- REGARDLESS of whether they are
actually installed on this machine (they are, in this dev venv: cellpose 3
needs torch on CPU). If ``commands.measure`` -- or anything it imports,
transitively -- ever tries to import either package, this test fails with a
real ``ImportError`` surfacing from inside ``measure()``.
"""

from __future__ import annotations

import hashlib
import inspect
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
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
from s2_adhesion.io.zarr_store import write_image_volume, write_label_volume

from tests.synthetic_volumes import sphere_pair_bisected

REPO_ROOT = Path(__file__).resolve().parents[2]

SPACING = (0.5, 0.25, 0.25)
SHAPE = (24, 40, 60)
CENTRE_UM = (6.0, 5.0, 7.5)
RADIUS_UM = 3.0
SEPARATION_UM = 4.0

# Installed as sitecustomize.py's sole content on a PYTHONPATH directory
# prepended ahead of everything else, so it runs at interpreter startup,
# before any test code. Blocks `torch`/`cellpose` (and every dotted
# submodule of either, e.g. `cellpose.models`) at the meta_path level --
# earlier than any try/except an importer might have, and regardless of
# whether the real packages are actually installed.
_SITECUSTOMIZE_SRC = textwrap.dedent(
    """
    import sys
    from importlib.abc import MetaPathFinder

    _BLOCKED_ROOTS = ("torch", "cellpose")

    class _BlockMLImports(MetaPathFinder):
        # `find_spec`, NOT the legacy `find_module`: the import system's
        # _bootstrap._find_spec only calls a finder if it has a `find_spec`
        # attribute (`getattr(finder, "find_spec", None)`), so a finder that
        # implements only the deprecated find_module is silently skipped and
        # never consulted at all -- this is not merely a style choice.
        def find_spec(self, fullname, path=None, target=None):
            head = fullname.split(".", 1)[0]
            if head in _BLOCKED_ROOTS:
                raise ImportError(
                    f"blocked by test sitecustomize: {fullname!r} must not be "
                    "imported by s2_adhesion.commands.measure"
                )
            return None

    sys.meta_path.insert(0, _BlockMLImports())
    """
)


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


def _build_bound_pair(tmp_path: Path) -> tuple[Path, Path]:
    """Write a matching image + label artifact pair, no segmentation involved
    (labels are hand-built from a synthetic sphere pair, exactly as a real
    label artifact -- produced elsewhere, possibly with cellpose -- would
    look once it reaches a machine with no ML stack)."""
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
    data[0][cells > 0] += 3000.0

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
        input_image_sha256=image_identity.image_content_sha256,
        device="cpu",
        host_platform="test",
    )
    labels = LabelVolume(
        cells=cells, geometry=image.geometry, identity=label_identity, provenance=provenance
    )

    image_dir = write_image_volume(image, tmp_path / "image.ome.zarr", chunks=image.data.shape)
    label_dir = write_label_volume(labels, tmp_path / "labels.ome.zarr", chunks=cells.shape)
    return image_dir, label_dir


def _sitecustomize_dir(tmp_path: Path) -> Path:
    site_dir = tmp_path / "blocked_site"
    site_dir.mkdir(parents=True, exist_ok=True)
    (site_dir / "sitecustomize.py").write_text(_SITECUSTOMIZE_SRC, encoding="utf-8")
    return site_dir


def _run_blocked(script: str, site_dir: Path, *, timeout: int = 120) -> subprocess.CompletedProcess:
    import os

    env = dict(os.environ)
    # PYTHONPATH's sitecustomize.py runs automatically at interpreter
    # startup (the `site` module imports it if found on sys.path) --
    # prepending our blocking directory guarantees it loads before any of
    # our own code, so the meta_path hook is in place for every import this
    # script performs. REPO_ROOT is appended after it so `s2_adhesion`/
    # `tests` are still importable.
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(site_dir), str(REPO_ROOT)] + ([existing] if existing else [])
    )
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
        timeout=timeout,
    )


def test_sitecustomize_hook_actually_blocks_torch_and_cellpose(tmp_path):
    """Guard the guard: prove the hook really does block real imports of
    torch/cellpose (both genuinely installed in this dev venv) before
    trusting it to prove anything about measure()."""
    site_dir = _sitecustomize_dir(tmp_path)
    script = textwrap.dedent(
        """
        import sys
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            torch_blocked = "blocked by test sitecustomize" in str(exc)
        else:
            torch_blocked = False
        try:
            import cellpose  # noqa: F401
        except ImportError as exc:
            cellpose_blocked = "blocked by test sitecustomize" in str(exc)
        else:
            cellpose_blocked = False
        assert torch_blocked, "torch import was NOT blocked"
        assert cellpose_blocked, "cellpose import was NOT blocked"
        print("HOOK_WORKS")
        """
    )
    result = _run_blocked(script, site_dir)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "HOOK_WORKS" in result.stdout


def test_measure_completes_and_writes_correct_csvs_with_torch_and_cellpose_blocked(tmp_path):
    """THE acceptance criterion: measure() runs to completion, producing
    correct CSVs, in a process where importing torch or cellpose raises
    ImportError for real."""
    image_dir, label_dir = _build_bound_pair(tmp_path / "artifacts")
    site_dir = _sitecustomize_dir(tmp_path)
    output_dir = tmp_path / "measure_out"

    script = textwrap.dedent(
        f"""
        import sys
        from pathlib import Path
        sys.path.insert(0, {str(REPO_ROOT)!r})

        # If commands.measure (or anything it imports) ever tries `import
        # torch` / `import cellpose`, the sitecustomize hook makes that raise
        # ImportError right here -- this whole script would fail loudly
        # rather than silently succeeding.
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
            image_artifact_dir=Path({str(image_dir)!r}),
        )
        assert result.objects_csv.exists()
        assert result.run_manifest_json.exists()
        assert "torch" not in sys.modules
        assert "cellpose" not in sys.modules
        print("MEASURE_WITHOUT_ML_OK")
        """
    )
    result = _run_blocked(script, site_dir)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "MEASURE_WITHOUT_ML_OK" in result.stdout

    import pandas as pd

    objects = pd.read_csv(output_dir / "objects.csv")
    contacts = pd.read_csv(output_dir / "contacts.csv")
    aggregates = pd.read_csv(output_dir / "aggregates.csv")

    assert set(objects["object_id"]) == {1, 2}
    assert objects["volume_um3"].notna().all()
    assert objects["volume_um3"].between(20.0, 300.0).all()
    assert objects["ch.membrane.raw_mean"].notna().all()

    assert len(contacts) == 1
    assert bool(contacts.iloc[0]["qualifies_as_contact"]) is True

    assert len(aggregates) == 1
    assert aggregates.iloc[0]["member_cell_ids"] == "1;2"

    assert (output_dir / "run_manifest.json").exists()


def test_measure_geometry_only_completes_with_torch_and_cellpose_blocked(tmp_path):
    """Same guarantee, geometry-only (no image artifact) path."""
    _, label_dir = _build_bound_pair(tmp_path / "artifacts")
    site_dir = _sitecustomize_dir(tmp_path)
    output_dir = tmp_path / "measure_geo_out"

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
        result = measure(
            Path({str(label_dir)!r}),
            Path({str(output_dir)!r}),
            config,
            image_artifact_dir=None,
        )
        assert result.objects_csv.exists()
        assert "torch" not in sys.modules
        assert "cellpose" not in sys.modules
        print("MEASURE_GEOMETRY_ONLY_WITHOUT_ML_OK")
        """
    )
    result = _run_blocked(script, site_dir)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "MEASURE_GEOMETRY_ONLY_WITHOUT_ML_OK" in result.stdout
