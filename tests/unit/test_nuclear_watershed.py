"""Unit tests for segmentation/nuclear_watershed.py.

All ML is faked via a local ``FakeCellposeEngine`` -- these tests exercise the
real watershed/extent/postprocess pipeline against canned seed labels, with
no cellpose or torch installed anywhere in the process.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage as ndi

from s2_adhesion.config import CellposeModelConfig, NucleusSeededWatershedConfig
from s2_adhesion.contracts import ChannelBinding, ChannelRole, FieldIdentity, ImageVolume, VoxelGeometry
from s2_adhesion.errors import SegmentationError
from s2_adhesion.segmentation import nuclear_watershed as nw

REPO_ROOT = Path(__file__).resolve().parents[2]

ANISOTROPIC = (0.5, 0.1, 0.1)


# ─── test fixtures / fakes ───────────────────────────────────────────────────
#
# CellposeRawResult/PreparedCellposeInput are real, concrete dataclasses from
# segmentation.protocol (re-exported by nuclear_watershed's guarded import) --
# no need for local stand-ins for those. Only the engine itself is faked.


class FakeCellposeEngine:
    """A canned-output stand-in for ``CellposeEngine``. No ML, fully synchronous.

    ``labels`` (already on the ORIGINAL image grid) is wrapped in a
    ``CellposeRawResult`` and returned verbatim from every ``evaluate()``
    call -- this only works cleanly because these tests use spacing where
    in-plane resampling is a no-op (dy == dx already square), so the
    "model grid" and "original grid" coincide and ``map_labels_to_original_grid``
    is an identity pass-through. Calls are recorded so tests can assert what
    this backend actually sent the engine (prepared shape, config).
    """

    package_major = 3
    model_name = "fake"
    package_version = "0.0.0-fake"
    device = "cpu"
    model_checksum = None

    def __init__(self, labels: np.ndarray) -> None:
        self._labels = labels
        self.calls: list[tuple[object, CellposeModelConfig]] = []

    def evaluate(self, prepared: object, *, config: CellposeModelConfig) -> nw.CellposeRawResult:
        self.calls.append((prepared, config))
        return nw.CellposeRawResult(labels=self._labels)


def make_identity(content: bytes = b"synthetic") -> FieldIdentity:
    import hashlib

    return FieldIdentity(
        dataset_id="synthetic",
        field_id="field00",
        source_uri="memory://synthetic",
        source_field_index=0,
        image_content_sha256=hashlib.sha256(content).hexdigest(),
    )


def make_image(
    shape_zyx: tuple[int, int, int],
    boundary_value: np.ndarray | float,
    spacing: tuple[float, float, float] = ANISOTROPIC,
) -> ImageVolume:
    """A 2-channel (membrane, nucleus) synthetic image. Nucleus data is unused
    by these tests (seeds come from FakeCellposeEngine), but a channel must
    exist for the pipeline's channel lookups to succeed."""
    membrane = np.broadcast_to(np.asarray(boundary_value, dtype=np.float32), shape_zyx).copy()
    nucleus = np.zeros(shape_zyx, dtype=np.float32)
    data = np.stack([membrane, nucleus], axis=0)
    channels = (
        ChannelBinding(channel_id="membrane", source_index=0, roles=frozenset({ChannelRole.MEMBRANE})),
        ChannelBinding(channel_id="nucleus", source_index=1, roles=frozenset({ChannelRole.NUCLEUS})),
    )
    return ImageVolume(
        data=data,
        geometry=VoxelGeometry(spacing_um_zyx=spacing),
        channels=channels,
        identity=make_identity(),
    )


def make_config(**overrides) -> NucleusSeededWatershedConfig:
    defaults = dict(
        strategy="nucleus_seeded_watershed",
        nucleus_channel_id="nucleus",
        boundary_channel_ids=("membrane",),
        extent_mode="adaptive_intensity",
        gaussian_sigma_um=0.5,
        watershed_compactness=0.0,
        min_nucleus_volume_um3=0.0,
        min_cell_volume_um3=0.0,
        max_nuclei_per_cell=2,
    )
    defaults.update(overrides)
    return NucleusSeededWatershedConfig(**defaults)


def two_nucleus_seed_labels(shape_zyx: tuple[int, int, int]) -> tuple[np.ndarray, tuple[slice, ...], tuple[slice, ...]]:
    """Two well-separated cube seeds, id 1 and id 2, inside ``shape_zyx``."""
    seeds = np.zeros(shape_zyx, dtype=np.uint32)
    z, y, x = shape_zyx
    block_a = (slice(0, z), slice(2, 6), slice(2, 6))
    block_b = (slice(0, z), slice(2, 6), slice(x - 6, x - 2))
    seeds[block_a] = 1
    seeds[block_b] = 2
    return seeds, block_a, block_b


# ─── no-ML import guard ─────────────────────────────────────────────────────


def test_importing_nuclear_watershed_avoids_ml_dependencies() -> None:
    """Importing this backend must never pull torch or cellpose into
    sys.modules -- run in a fresh subprocess so other test modules in the
    same pytest session cannot pollute (or falsely clear) the result."""
    script = (
        "import sys\n"
        "import s2_adhesion.segmentation.nuclear_watershed\n"
        "assert 'torch' not in sys.modules, sorted(sys.modules)\n"
        "assert 'cellpose' not in sys.modules, sorted(sys.modules)\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


# ─── Gaussian sigma: um -> per-axis voxels ──────────────────────────────────


def test_gaussian_sigma_voxels_differ_per_axis_under_anisotropic_spacing() -> None:
    sigma_vox = nw.compute_gaussian_sigma_voxels(0.5, ANISOTROPIC)

    dz, dy, dx = ANISOTROPIC
    assert sigma_vox == pytest.approx((0.5 / dz, 0.5 / dy, 0.5 / dx))
    assert sigma_vox == pytest.approx((1.0, 5.0, 5.0))
    # The whole point: Z's voxel-sigma must differ from XY's when spacing does.
    assert sigma_vox[0] != pytest.approx(sigma_vox[1])
    assert sigma_vox[1] == pytest.approx(sigma_vox[2])  # isotropic in-plane


def test_gaussian_sigma_voxels_equal_under_isotropic_spacing() -> None:
    sigma_vox = nw.compute_gaussian_sigma_voxels(0.3, (0.2, 0.2, 0.2))
    assert sigma_vox == pytest.approx((1.5, 1.5, 1.5))


def test_gaussian_sigma_voxels_rejects_negative_sigma() -> None:
    with pytest.raises(SegmentationError):
        nw.compute_gaussian_sigma_voxels(-1.0, ANISOTROPIC)


def test_segment_applies_the_anisotropic_sigma_to_the_cost_surface(monkeypatch: pytest.MonkeyPatch) -> None:
    """Integration check that the per-axis sigma computed above is actually
    the one handed to the smoothing call inside segment(), not just correct
    in isolation."""
    shape = (3, 10, 20)
    image = make_image(shape, boundary_value=0.0, spacing=ANISOTROPIC)
    seeds, block_a, block_b = two_nucleus_seed_labels(shape)
    engine = FakeCellposeEngine(labels=seeds)
    config = make_config(gaussian_sigma_um=0.5)
    backend = nw.NucleusSeededWatershedBackend(config=config, seed_engine=engine)

    captured: dict[str, object] = {}
    real_gaussian_filter = ndi.gaussian_filter

    def spy(*args, **kwargs):
        captured["sigma"] = kwargs.get("sigma", args[1] if len(args) > 1 else None)
        return real_gaussian_filter(*args, **kwargs)

    monkeypatch.setattr(nw.ndi, "gaussian_filter", spy)

    request = nw.SegmentationRequest(image=image, run_id="run0")
    backend.segment(request)

    assert captured["sigma"] == pytest.approx(
        nw.compute_gaussian_sigma_voxels(0.5, ANISOTROPIC)
    )


# ─── core split behaviour ────────────────────────────────────────────────────


def test_two_nuclei_in_one_extent_split_into_two_cells_each_owning_one_nucleus() -> None:
    shape = (3, 10, 20)
    # Flat (constant) boundary channel: adaptive_intensity's degenerate-input
    # fallback makes the whole volume one connected extent, and a flat cost
    # surface makes watershed a pure geodesic-nearest-seed split -- the
    # cleanest possible test of "the boundary lands between the two seeds"
    # without coupling the test to any particular membrane-intensity shape.
    image = make_image(shape, boundary_value=0.0, spacing=ANISOTROPIC)
    seeds, block_a, block_b = two_nucleus_seed_labels(shape)
    engine = FakeCellposeEngine(labels=seeds)
    config = make_config(max_nuclei_per_cell=2)  # avoid an oversegmented flag here
    backend = nw.NucleusSeededWatershedBackend(config=config, seed_engine=engine)

    result = backend.segment(nw.SegmentationRequest(image=image, run_id="run0"))

    cells = result.labels.cells
    nuclei = result.labels.nuclei
    assert nuclei is not None

    nucleus_ids = [i for i in np.unique(nuclei) if i > 0]
    assert len(nucleus_ids) == 2

    cell_ids_per_nucleus = []
    for nucleus_id in nucleus_ids:
        under_nucleus = cells[nuclei == nucleus_id]
        assert np.all(under_nucleus == under_nucleus[0]), "a nucleus must sit inside exactly one cell"
        assert under_nucleus[0] != 0, "a seeded nucleus must produce a labelled cell"
        cell_ids_per_nucleus.append(int(under_nucleus[0]))

    assert cell_ids_per_nucleus[0] != cell_ids_per_nucleus[1]
    assert len(set(int(i) for i in np.unique(cells) if i > 0)) == 2

    assert not any("oversegmented_extent" in w for w in result.diagnostics.warnings)
    assert not any("zero_seed_extent" in w for w in result.diagnostics.warnings)
    assert not any("orphan_nucleus" in w for w in result.diagnostics.warnings)


# ─── flagged, never-silent edge cases ───────────────────────────────────────


def test_zero_seed_extent_component_is_flagged_and_left_as_background() -> None:
    shape = (2, 10, 20)
    # A bright wall spanning the full extent at x=9..10 splits the volume
    # into two disconnected (Otsu-thresholded) extent components.
    boundary = np.zeros(shape, dtype=np.float32)
    boundary[:, :, 9:11] = 200.0
    image = make_image(shape, boundary_value=boundary, spacing=ANISOTROPIC)

    seeds = np.zeros(shape, dtype=np.uint32)
    seeds[:, 2:6, 2:6] = 1  # only the left block gets a nucleus
    engine = FakeCellposeEngine(labels=seeds)
    config = make_config()
    backend = nw.NucleusSeededWatershedBackend(config=config, seed_engine=engine)

    result = backend.segment(nw.SegmentationRequest(image=image, run_id="run0"))

    assert result.diagnostics.extra["n_zero_seed_extents"] == 1
    assert any("zero_seed_extent" in w for w in result.diagnostics.warnings)
    # The unseeded right-hand block must not have silently acquired a cell.
    assert not np.any(result.labels.cells[:, 2:6, 14:18] != 0)


def test_more_seeds_than_max_nuclei_per_cell_is_flagged_but_still_segmented() -> None:
    shape = (3, 10, 20)
    image = make_image(shape, boundary_value=0.0, spacing=ANISOTROPIC)  # one connected extent
    seeds, block_a, block_b = two_nucleus_seed_labels(shape)
    engine = FakeCellposeEngine(labels=seeds)
    config = make_config(max_nuclei_per_cell=1)  # 2 seeds > limit of 1
    backend = nw.NucleusSeededWatershedBackend(config=config, seed_engine=engine)

    result = backend.segment(nw.SegmentationRequest(image=image, run_id="run0"))

    assert result.diagnostics.extra["n_oversegmented_extents"] == 1
    assert any("oversegmented_extent" in w for w in result.diagnostics.warnings)
    # Not silently dropped: both nuclei still produced distinct cells.
    nuclei = result.labels.nuclei
    assert nuclei is not None
    cells = result.labels.cells
    ids = {int(cells[nuclei == nid][0]) for nid in np.unique(nuclei) if nid > 0}
    assert len(ids) == 2
    assert 0 not in ids


def test_nucleus_outside_any_extent_is_flagged_and_produces_no_cell() -> None:
    shape = (2, 10, 20)
    boundary = np.zeros(shape, dtype=np.float32)
    boundary[:, :, 9:11] = 200.0  # thresholded OUT of the extent
    image = make_image(shape, boundary_value=boundary, spacing=ANISOTROPIC)

    seeds = np.zeros(shape, dtype=np.uint32)
    seeds[:, 4:6, 9:11] = 1  # the nucleus sits entirely inside the bright wall
    engine = FakeCellposeEngine(labels=seeds)
    config = make_config()
    backend = nw.NucleusSeededWatershedBackend(config=config, seed_engine=engine)

    result = backend.segment(nw.SegmentationRequest(image=image, run_id="run0"))

    assert result.diagnostics.extra["n_orphan_nuclei"] == 1
    assert any("orphan_nucleus" in w for w in result.diagnostics.warnings)
    assert not np.any(result.labels.cells != 0), "an orphan nucleus must not seed a cell"


def test_cell_below_min_volume_is_flagged_and_removed() -> None:
    shape = (3, 10, 20)
    image = make_image(shape, boundary_value=0.0, spacing=ANISOTROPIC)
    seeds, block_a, block_b = two_nucleus_seed_labels(shape)
    engine = FakeCellposeEngine(labels=seeds)
    # An enormous floor: every cell watershed can possibly produce here is
    # far smaller than this, so both must be flagged and dropped.
    config = make_config(max_nuclei_per_cell=2, min_cell_volume_um3=1.0e9)
    backend = nw.NucleusSeededWatershedBackend(config=config, seed_engine=engine)

    result = backend.segment(nw.SegmentationRequest(image=image, run_id="run0"))

    assert result.diagnostics.extra["n_cells_below_min_volume"] == 2
    assert any("cell_below_min_volume" in w for w in result.diagnostics.warnings)
    assert not np.any(result.labels.cells != 0), "undersized cells must be removed, not left in"


# ─── extent_mode = cellpose_union (a second injected engine) ───────────────


def test_cellpose_union_extent_mode_uses_the_injected_extent_engine() -> None:
    """extent_mode='cellpose_union' must consult extent_engine, not build the
    extent via adaptive thresholding, and must still split correctly."""
    shape = (3, 10, 20)
    image = make_image(shape, boundary_value=0.0, spacing=ANISOTROPIC)
    seeds, block_a, block_b = two_nucleus_seed_labels(shape)
    seed_engine = FakeCellposeEngine(labels=seeds)

    extent_labels = np.ones(shape, dtype=np.uint32)  # union covers the whole volume
    extent_engine = FakeCellposeEngine(labels=extent_labels)

    config = make_config(
        extent_mode="cellpose_union",
        extent_model=CellposeModelConfig(package_major=3, model_name="cyto3"),
        max_nuclei_per_cell=2,
    )
    backend = nw.NucleusSeededWatershedBackend(
        config=config, seed_engine=seed_engine, extent_engine=extent_engine
    )

    result = backend.segment(nw.SegmentationRequest(image=image, run_id="run0"))

    assert len(extent_engine.calls) == 1, "the extent engine must be consulted exactly once"
    cells = result.labels.cells
    nuclei = result.labels.nuclei
    assert nuclei is not None
    ids = {int(cells[nuclei == nid][0]) for nid in np.unique(nuclei) if nid > 0}
    assert len(ids) == 2
    assert 0 not in ids


def test_cellpose_union_without_extent_engine_raises() -> None:
    shape = (2, 4, 4)
    image = make_image(shape, boundary_value=0.0, spacing=ANISOTROPIC)
    seeds = np.zeros(shape, dtype=np.uint32)
    seed_engine = FakeCellposeEngine(labels=seeds)
    config = make_config(
        extent_mode="cellpose_union",
        extent_model=CellposeModelConfig(package_major=3, model_name="cyto3"),
    )
    backend = nw.NucleusSeededWatershedBackend(config=config, seed_engine=seed_engine)

    with pytest.raises(SegmentationError):
        backend.segment(nw.SegmentationRequest(image=image, run_id="run0"))
