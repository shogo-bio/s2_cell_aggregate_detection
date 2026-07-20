"""Tests for ``s2_adhesion.metrics.intensity``.

All synthetic arrays are built locally -- this file does not import
``tests/synthetic_volumes.py`` (owned by a concurrent agent).
"""

from __future__ import annotations

import math
import subprocess
import sys

import numpy as np
import pytest

from s2_adhesion.config import BackgroundConfig
from s2_adhesion.contracts import ChannelRole, VoxelGeometry
from s2_adhesion.errors import ContractViolation
from s2_adhesion.metrics.intensity import (
    _channel_background,
    _region_stats,
    _shell_masks,
    compute_intensity,
)

from tests.conftest import ANISOTROPIC, ISOTROPIC, make_image_volume, make_label_volume

FIXED_NOOP = BackgroundConfig(mode="fixed", fixed_value_by_channel={"signal": 0.0})


# ─── raw distributional stats ──────────────────────────────────────────────


def test_constant_region_exact_stats():
    """Constant intensity -> exact mean/median/percentiles, and sum == value*N."""
    value = 7.0
    shape = (4, 4, 8)  # 128 voxels, power of two so pairwise-sum stays exact
    data = np.full((1, *shape), value, dtype=np.float64)
    labels = np.ones(shape, dtype=np.uint32)

    image = make_image_volume(data, channel_ids=("signal",), roles=(ChannelRole.SIGNAL,))
    result = compute_intensity(image, labels, FIXED_NOOP, 1.0, 1.0)

    row = result[1]
    voxel_count = int(np.prod(shape))
    assert row["ch.signal.raw_mean"] == value
    assert row["ch.signal.raw_median"] == value
    assert row["ch.signal.raw_p10"] == value
    assert row["ch.signal.raw_p90"] == value
    assert row["ch.signal.raw_min"] == value
    assert row["ch.signal.raw_max"] == value
    assert row["ch.signal.raw_std"] == 0.0
    assert row["ch.signal.raw_sum"] == value * voxel_count  # exact, not approx


def test_linear_ramp_analytic_mean_and_percentiles():
    """A 0..100 ramp gives the closed-form linear-interpolation percentiles."""
    ramp = np.arange(101, dtype=np.float64).reshape(1, 1, 101)
    data = ramp[np.newaxis, ...]  # (1 channel, 1, 1, 101)
    labels = np.ones((1, 1, 101), dtype=np.uint32)

    image = make_image_volume(data, channel_ids=("signal",), roles=(ChannelRole.SIGNAL,))
    result = compute_intensity(image, labels, FIXED_NOOP, 1.0, 1.0)

    row = result[1]
    assert row["ch.signal.raw_mean"] == pytest.approx(50.0)
    assert row["ch.signal.raw_p10"] == pytest.approx(10.0)
    assert row["ch.signal.raw_median"] == pytest.approx(50.0)
    assert row["ch.signal.raw_p90"] == pytest.approx(90.0)
    assert row["ch.signal.raw_sum"] == pytest.approx(5050.0)


# ─── physical scaling ───────────────────────────────────────────────────────


def _padded_sphere_shape(radius_um, spacing, pad_vox=4):
    dz, dy, dx = spacing
    nz = math.ceil(2 * radius_um / dz) + 2 * pad_vox
    ny = math.ceil(2 * radius_um / dy) + 2 * pad_vox
    nx = math.ceil(2 * radius_um / dx) + 2 * pad_vox
    return (nz, ny, nx)


def _sphere_labels(shape, spacing, radius_um):
    geometry = VoxelGeometry(spacing_um_zyx=spacing)
    idx = np.stack(np.indices(shape), axis=-1)
    phys = geometry.index_to_um(idx)
    dz, dy, dx = spacing
    nz, ny, nx = shape
    center = np.array([nz * dz / 2.0, ny * dy / 2.0, nx * dx / 2.0])
    r = np.sqrt(((phys - center) ** 2).sum(axis=-1))
    labels = np.zeros(shape, dtype=np.uint32)
    labels[r <= radius_um] = 1
    return labels


def test_corrected_integrated_um3_invariant_under_spacing():
    """Same physical sphere, two spacings -- integrated corrected intensity agrees within 1%."""
    radius_um = 3.0
    value = 100.0
    background_cfg = BackgroundConfig(mode="fixed", fixed_value_by_channel={"signal": 0.0})

    integrated = {}
    for spacing in (ANISOTROPIC, ISOTROPIC):
        shape = _padded_sphere_shape(radius_um, spacing)
        labels = _sphere_labels(shape, spacing, radius_um)
        data = np.full((1, *shape), value, dtype=np.float64)
        image = make_image_volume(
            data, spacing=spacing, channel_ids=("signal",), roles=(ChannelRole.SIGNAL,)
        )
        result = compute_intensity(image, labels, background_cfg, 1.0, 1.0)
        integrated[spacing] = result[1]["ch.signal.corrected_integrated_um3"]

    aniso_val = integrated[ANISOTROPIC]
    iso_val = integrated[ISOTROPIC]
    assert aniso_val is not None and iso_val is not None
    rel_diff = abs(aniso_val - iso_val) / iso_val
    assert rel_diff < 0.01, f"relative difference {rel_diff:.4%} exceeds 1%"


# ─── background subtraction ────────────────────────────────────────────────


def test_background_subtraction_fixed_recovers_signal():
    background = 200.0
    signal = 37.0
    shape = (2, 5, 5)
    data = np.full((1, *shape), background + signal, dtype=np.float64)
    labels = np.ones(shape, dtype=np.uint32)

    cfg = BackgroundConfig(mode="fixed", fixed_value_by_channel={"signal": background})
    image = make_image_volume(data, channel_ids=("signal",), roles=(ChannelRole.SIGNAL,))
    result = compute_intensity(image, labels, cfg, 1.0, 1.0)

    row = result[1]
    voxel_count = int(np.prod(shape))
    voxel_volume = math.prod(ANISOTROPIC)
    assert row["ch.signal.corrected_mean"] == pytest.approx(signal)
    assert row["ch.signal.corrected_integrated_um3"] == pytest.approx(
        voxel_volume * signal * voxel_count
    )
    # fixed mode never yields a data-driven MAD
    assert row["ch.signal.background_mad"] is None
    assert row["ch.signal.positive_fraction"] is None
    assert "mad_undefined_for_fixed_background" in row["ch.signal.qc"]


def test_outside_cells_median_mad_ignores_cell_voxels():
    """A huge outlier inside the cell must not skew the background estimate."""
    shape = (2, 6, 6)  # 72 voxels
    labels = np.zeros(shape, dtype=np.uint32)
    labels[:, :3, :3] = 1  # 18-voxel cell block
    outside_mask = labels == 0
    outside_idx = np.argwhere(outside_mask)
    assert len(outside_idx) == 54  # even, so the median lands exactly on B

    baseline = 500.0
    data = np.zeros((1, *shape), dtype=np.float64)
    half = len(outside_idx) // 2
    for i, (z, y, x) in enumerate(outside_idx):
        data[0, z, y, x] = baseline - 1.0 if i < half else baseline + 1.0
    data[0][labels == 1] = 999_999.0  # outlier that must be ignored

    cfg = BackgroundConfig(mode="outside_cells_median_mad")
    image = make_image_volume(data, channel_ids=("signal",), roles=(ChannelRole.SIGNAL,))
    result = compute_intensity(image, labels, cfg, 1.0, 1.0)

    row = result[1]
    assert row["ch.signal.background"] == pytest.approx(baseline)
    assert row["ch.signal.background_mad"] == pytest.approx(1.0)


def test_background_region_empty_returns_none_with_qc():
    """No voxel has label 0 anywhere -- 'outside_cells' has nothing to estimate from."""
    shape = (2, 4, 4)
    labels = np.ones(shape, dtype=np.uint32)  # entire FOV is one cell
    data = np.full((1, *shape), 42.0, dtype=np.float64)

    cfg = BackgroundConfig(mode="outside_cells_median_mad")
    image = make_image_volume(data, channel_ids=("signal",), roles=(ChannelRole.SIGNAL,))
    result = compute_intensity(image, labels, cfg, 1.0, 1.0)

    row = result[1]
    assert row["ch.signal.background"] is None
    assert row["ch.signal.background_mad"] is None
    assert row["ch.signal.corrected_mean"] is None
    assert row["ch.signal.corrected_integrated_um3"] is None
    assert row["ch.signal.positive_fraction"] is None
    assert "background_region_empty" in row["ch.signal.qc"]
    # not NaN, not raising, not silently zero
    assert row["ch.signal.background"] is not np.nan


def test_zero_background_mad_fully_saturated_background_nulls_positive_fraction():
    """A perfectly uniform (e.g. saturated) background gives MAD == 0.

    background/background_mad/corrected_* remain well-defined real numbers,
    but positive_fraction becomes undefined (a zero threshold cannot separate
    signal from noise) and must be None with a QC code, not silently 0 or NaN.
    """
    shape = (2, 6, 6)
    labels = np.zeros(shape, dtype=np.uint32)
    labels[:, :2, :2] = 1
    saturated_value = 65535.0
    data = np.full((1, *shape), saturated_value, dtype=np.float64)
    data[0][labels == 1] = saturated_value + 10.0  # a tiny bit of "signal"

    cfg = BackgroundConfig(mode="outside_cells_median_mad")
    image = make_image_volume(data, channel_ids=("signal",), roles=(ChannelRole.SIGNAL,))
    result = compute_intensity(image, labels, cfg, 1.0, 1.0)

    row = result[1]
    assert row["ch.signal.background"] == pytest.approx(saturated_value)
    assert row["ch.signal.background_mad"] == 0.0
    assert row["ch.signal.corrected_mean"] == pytest.approx(10.0)
    assert row["ch.signal.positive_fraction"] is None
    assert "zero_background_mad" in row["ch.signal.qc"]


def test_missing_fixed_background_value_returns_none_with_qc():
    shape = (2, 3, 3)
    labels = np.ones(shape, dtype=np.uint32)
    data = np.full((1, *shape), 5.0, dtype=np.float64)
    cfg = BackgroundConfig(mode="fixed", fixed_value_by_channel={})  # nothing configured

    image = make_image_volume(data, channel_ids=("signal",), roles=(ChannelRole.SIGNAL,))
    result = compute_intensity(image, labels, cfg, 1.0, 1.0)

    row = result[1]
    assert row["ch.signal.background"] is None
    assert row["ch.signal.corrected_mean"] is None
    assert "missing_fixed_background_value" in row["ch.signal.qc"]


# ─── empty-region helper coverage (whitebox) ───────────────────────────────


def test_region_stats_empty_array_returns_none_not_nan():
    stats = _region_stats(np.array([], dtype=np.float64))
    for key, value in stats.items():
        assert value is None, f"{key} should be None for an empty region, got {value!r}"


def test_channel_background_outside_cells_empty_returns_none():
    labels = np.ones((2, 2, 2), dtype=np.uint32)
    chan = np.zeros((2, 2, 2), dtype=np.float64)
    cfg = BackgroundConfig(mode="outside_cells_median_mad")
    value, mad, codes = _channel_background(chan, labels, "signal", cfg)
    assert value is None
    assert mad is None
    assert "background_region_empty" in codes


# ─── shells under anisotropic spacing ──────────────────────────────────────


def test_shell_masks_use_per_axis_spacing_not_voxel_counts():
    """A one-voxel step means very different physical distances along z vs xy.

    Under ANISOTROPIC spacing (0.5, 0.1, 0.1), a single voxel step along z is
    0.5 um while a single voxel step along y or x is 0.1 um. With
    outer_shell_width_um=0.25, the z-adjacent neighbour must be excluded
    (0.5 > 0.25) while the y- and x-adjacent neighbours must be included
    (0.1 <= 0.25) -- despite all three being exactly one voxel away. A shell
    computation that used voxel-unit distances (ignoring ``sampling``) would
    treat all three identically.
    """
    shape = (10, 10, 10)
    labels = np.zeros(shape, dtype=np.uint32)
    labels[3:6, 3:6, 3:6] = 1
    mask = labels == 1

    z_neighbor = (2, 4, 4)
    y_neighbor = (4, 2, 4)
    x_neighbor = (4, 4, 2)

    _inner, _core, outer = _shell_masks(
        mask, ANISOTROPIC, inner_shell_width_um=1.0, outer_shell_width_um=0.25
    )
    assert not outer[z_neighbor]
    assert outer[y_neighbor]
    assert outer[x_neighbor]


def test_shell_means_sphere_with_bright_outer_rind():
    """A sphere with a bright rind of known physical thickness lands in the outer shell."""
    radius_um = 2.0
    shell_width_um = 0.4
    dim_value = 10.0
    bright_value = 1000.0

    shape = _padded_sphere_shape(radius_um, ANISOTROPIC)
    geometry = VoxelGeometry(spacing_um_zyx=ANISOTROPIC)
    idx = np.stack(np.indices(shape), axis=-1)
    phys = geometry.index_to_um(idx)
    dz, dy, dx = ANISOTROPIC
    nz, ny, nx = shape
    center = np.array([nz * dz / 2.0, ny * dy / 2.0, nx * dx / 2.0])
    r = np.sqrt(((phys - center) ** 2).sum(axis=-1))

    labels = np.zeros(shape, dtype=np.uint32)
    labels[r <= radius_um] = 1
    rind = (labels == 0) & (r > radius_um) & (r <= radius_um + shell_width_um)

    data = np.full((1, *shape), dim_value, dtype=np.float64)
    data[0][rind] = bright_value

    cfg = BackgroundConfig(mode="fixed", fixed_value_by_channel={"signal": 0.0})
    image = make_image_volume(data, channel_ids=("signal",), roles=(ChannelRole.SIGNAL,))
    result = compute_intensity(image, labels, cfg, inner_shell_width_um=1.0,
                                outer_shell_width_um=shell_width_um)

    row = result[1]
    assert row["ch.signal.outer_shell_mean"] == pytest.approx(bright_value, rel=1e-6)
    assert row["ch.signal.core_mean"] == pytest.approx(dim_value, rel=1e-6)
    assert row["ch.signal.inner_shell_mean"] == pytest.approx(dim_value, rel=1e-6)


def test_shell_region_empty_when_cell_smaller_than_inner_width():
    """A cell entirely thinner than inner_shell_width_um has no 'core'."""
    shape = (2, 6, 6)
    labels = np.zeros(shape, dtype=np.uint32)
    labels[:, 2:4, 2:4] = 1  # small block, well within a 5um "inner" width
    data = np.full((1, *shape), 3.0, dtype=np.float64)

    cfg = BackgroundConfig(mode="fixed", fixed_value_by_channel={"signal": 0.0})
    image = make_image_volume(data, channel_ids=("signal",), roles=(ChannelRole.SIGNAL,))
    result = compute_intensity(image, labels, cfg, inner_shell_width_um=5.0,
                                outer_shell_width_um=1.0)

    row = result[1]
    assert row["ch.signal.core_mean"] is None
    assert "core_empty" in row["ch.signal.qc"]


# ─── contract guard ─────────────────────────────────────────────────────────


def test_shape_mismatch_raises_contract_violation():
    shape = (2, 4, 4)
    data = np.zeros((1, *shape), dtype=np.float64)
    image = make_image_volume(data, channel_ids=("signal",), roles=(ChannelRole.SIGNAL,))
    wrong_labels = np.zeros((2, 4, 5), dtype=np.uint32)

    with pytest.raises(ContractViolation):
        compute_intensity(image, wrong_labels, FIXED_NOOP, 1.0, 1.0)


# ─── role-agnosticism ───────────────────────────────────────────────────────


def test_role_agnostic_output_identical_regardless_of_channel_role():
    """Same data, only the ChannelRole assignment differs -> byte-identical output.

    This is the property the module exists to guarantee: no function here may
    branch on ChannelRole, channel index, or channel name.
    """
    shape = (3, 5, 5)
    labels = np.zeros(shape, dtype=np.uint32)
    labels[:, 1:4, 1:4] = 1

    rng_a = np.fromfunction(lambda z, y, x: (z * 17 + y * 5 + x * 3) % 41, shape)
    rng_b = np.fromfunction(lambda z, y, x: (z * 7 + y * 11 + x * 13) % 29 + 3, shape)
    data = np.stack([rng_a, rng_b]).astype(np.float64)

    cfg = BackgroundConfig(mode="outside_cells_median_mad")

    image_a = make_image_volume(
        data,
        channel_ids=("chanA", "chanB"),
        roles=(ChannelRole.NUCLEUS, ChannelRole.MEMBRANE),
    )
    image_b = make_image_volume(
        data,
        channel_ids=("chanA", "chanB"),
        roles=(ChannelRole.SIGNAL, ChannelRole.ORGANELLE_MARKER),
    )

    result_a = compute_intensity(image_a, labels, cfg, 1.0, 1.0)
    result_b = compute_intensity(image_b, labels, cfg, 1.0, 1.0)

    assert result_a == result_b


def test_ignore_role_excludes_channel_from_output():
    shape = (2, 4, 4)
    labels = np.ones(shape, dtype=np.uint32)
    data = np.stack(
        [np.full(shape, 1.0), np.full(shape, 2.0)]
    ).astype(np.float64)

    image = make_image_volume(
        data,
        channel_ids=("kept", "dropped"),
        roles=(ChannelRole.SIGNAL, ChannelRole.IGNORE),
    )
    result = compute_intensity(image, labels, FIXED_NOOP, 1.0, 1.0)

    row = result[1]
    assert any(k.startswith("ch.kept.") for k in row)
    assert not any(k.startswith("ch.dropped.") for k in row)


# ─── no ML dependency ───────────────────────────────────────────────────────


def test_importing_intensity_module_does_not_import_torch_or_cellpose():
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import s2_adhesion.metrics.intensity; "
            "assert 'torch' not in sys.modules, 'torch was imported'; "
            "assert 'cellpose' not in sys.modules, 'cellpose was imported'",
        ],
        capture_output=True,
        text=True,
        cwd=str(__import__("pathlib").Path(__file__).resolve().parents[2]),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
