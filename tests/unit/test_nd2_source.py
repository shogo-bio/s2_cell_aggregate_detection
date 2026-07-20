"""Tests for ND2Source and extract(), against fake nd2-like objects only.

There is no .nd2 file anywhere in this environment and none is needed: every
fake here duck-types the surface this module actually reads (``.sizes``,
``.asarray()``, ``.voxel_size()``, ``.metadata``, and context-manager
protocol), the same surface ``nd2.ND2File`` exposes.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from s2_adhesion.commands.extract import extract
from s2_adhesion.config import PipelineConfig
from s2_adhesion.contracts import ChannelBinding, ChannelRole
from s2_adhesion.errors import ArtifactError
from s2_adhesion.io.nd2_source import ND2Source

# ─── fakes ──────────────────────────────────────────────────────────────────


class FakeVoxelSize:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class FakeChannelMeta:
    def __init__(self, name):
        self.name = name


class FakeChannelEntry:
    def __init__(self, name):
        self.channel = FakeChannelMeta(name)


class FakeMetadata:
    def __init__(self, channel_names=None):
        self.channels = (
            [FakeChannelEntry(n) for n in channel_names] if channel_names else None
        )


class FakeND2File:
    """Duck-typed stand-in for nd2.ND2File. Also a reusable context manager."""

    def __init__(self, sizes, data, voxel_size, channel_names=None, metadata=None):
        self.sizes = sizes
        self._data = data
        self._voxel_size = voxel_size
        self.metadata = metadata if metadata is not None else FakeMetadata(channel_names)

    def asarray(self):
        return self._data

    def voxel_size(self):
        return self._voxel_size

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def reader_factory_for(fake: FakeND2File):
    def factory(path):
        return fake

    return factory


def make_config(
    channel_ids=("membrane", "signal", "nucleus"),
    source_indices=(0, 1, 2),
) -> PipelineConfig:
    channels = tuple(
        ChannelBinding(
            channel_id=cid, source_index=idx, roles=frozenset({ChannelRole.SIGNAL})
        )
        for cid, idx in zip(channel_ids, source_indices)
    )
    return PipelineConfig(channels=channels)


# ─── field_ids() / P-axis handling ─────────────────────────────────────────


def test_multi_position_yields_one_field_per_position():
    sizes = {"P": 3, "Z": 10, "C": 3, "Y": 64, "X": 64}
    data = np.arange(np.prod(list(sizes.values())), dtype=np.uint16).reshape(
        tuple(sizes.values())
    )
    fake = FakeND2File(sizes, data, FakeVoxelSize(x=0.1, y=0.1, z=0.5))
    source = ND2Source("fake.nd2", make_config(), reader_factory=reader_factory_for(fake))

    ids = source.field_ids()
    assert len(ids) == 3
    assert len(set(ids)) == 3  # stable, distinct

    for fid in ids:
        vol = source.read_field(fid)
        assert vol.axes == "CZYX"
        assert vol.data.shape == (3, 10, 64, 64)


def test_no_p_axis_yields_exactly_one_field():
    sizes = {"C": 3, "Z": 8, "Y": 32, "X": 32}
    data = np.zeros(tuple(sizes.values()), dtype=np.uint16)
    fake = FakeND2File(sizes, data, FakeVoxelSize(x=0.2, y=0.2, z=1.0))
    source = ND2Source("fake.nd2", make_config(), reader_factory=reader_factory_for(fake))

    ids = source.field_ids()
    assert len(ids) == 1

    vol = source.read_field(ids[0])
    assert vol.data.shape == (3, 8, 32, 32)
    assert vol.identity.source_field_index == 0


def test_different_key_order_still_transposes_to_czyx():
    # C first, then spatial, then Z last -- deliberately not C,Z,Y,X order.
    sizes = {"C": 2, "Y": 16, "X": 20, "Z": 5}
    shape = tuple(sizes.values())  # (2, 16, 20, 5)
    data = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    fake = FakeND2File(sizes, data, FakeVoxelSize(x=0.1, y=0.1, z=0.5))
    config = make_config(channel_ids=("a", "b"), source_indices=(0, 1))
    source = ND2Source("fake.nd2", config, reader_factory=reader_factory_for(fake))

    vol = source.read_field(source.field_ids()[0])
    assert vol.data.shape == (2, 5, 16, 20)  # C, Z, Y, X

    # Content must actually be the transposed data, not just the right shape.
    expected = np.transpose(data, (0, 3, 1, 2))  # dim_keys C,Y,X,Z -> C,Z,Y,X
    np.testing.assert_array_equal(vol.data, expected)


def test_p_axis_in_different_position_selects_correct_field():
    # P is not axis 0 here.
    sizes = {"C": 2, "P": 2, "Y": 4, "X": 4}
    shape = tuple(sizes.values())
    data = np.arange(np.prod(shape)).reshape(shape).astype(np.uint16)
    fake = FakeND2File(sizes, data, FakeVoxelSize(x=0.1, y=0.1, z=0.5))
    config = make_config(channel_ids=("a", "b"), source_indices=(0, 1))
    source = ND2Source("fake.nd2", config, reader_factory=reader_factory_for(fake))

    ids = source.field_ids()
    assert len(ids) == 2

    vol0 = source.read_field(ids[0])
    vol1 = source.read_field(ids[1])
    # field 0 and field 1 must differ (P axis actually selected, not ignored)
    assert not np.array_equal(vol0.data, vol1.data)

    p_axis = list(sizes.keys()).index("P")
    expected0 = np.take(data, 0, axis=p_axis)  # remaining dims: C, Y, X (no Z)
    expected0 = expected0[:, np.newaxis, :, :]  # insert singleton Z
    np.testing.assert_array_equal(vol0.data, expected0)


# ─── voxel spacing: the actual bug being fixed ─────────────────────────────


def test_dz_dy_dx_all_read_correctly():
    sizes = {"Z": 4, "C": 1, "Y": 8, "X": 8}
    data = np.zeros(tuple(sizes.values()), dtype=np.uint16)
    fake = FakeND2File(sizes, data, FakeVoxelSize(x=0.11, y=0.12, z=0.55))
    config = make_config(channel_ids=("a",), source_indices=(0,))
    source = ND2Source("fake.nd2", config, reader_factory=reader_factory_for(fake))

    vol = source.read_field(source.field_ids()[0])
    dz, dy, dx = vol.geometry.spacing_um_zyx
    assert dz == pytest.approx(0.55)
    assert dy == pytest.approx(0.12)
    assert dx == pytest.approx(0.11)


def test_zero_z_spacing_raises_naming_z():
    sizes = {"Z": 4, "C": 1, "Y": 8, "X": 8}
    data = np.zeros(tuple(sizes.values()), dtype=np.uint16)
    fake = FakeND2File(sizes, data, FakeVoxelSize(x=0.1, y=0.1, z=0.0))
    config = make_config(channel_ids=("a",), source_indices=(0,))
    source = ND2Source("fake.nd2", config, reader_factory=reader_factory_for(fake))

    with pytest.raises(ArtifactError) as excinfo:
        source.read_field(source.field_ids()[0])
    assert "z" in str(excinfo.value).lower()
    assert "0" in str(excinfo.value)


def test_missing_z_spacing_raises_naming_z():
    class VoxelSizeNoZ:
        def __init__(self, x, y):
            self.x = x
            self.y = y
            # deliberately no .z attribute

    sizes = {"Z": 4, "C": 1, "Y": 8, "X": 8}
    data = np.zeros(tuple(sizes.values()), dtype=np.uint16)
    fake = FakeND2File(sizes, data, VoxelSizeNoZ(x=0.1, y=0.1))
    config = make_config(channel_ids=("a",), source_indices=(0,))
    source = ND2Source("fake.nd2", config, reader_factory=reader_factory_for(fake))

    with pytest.raises(ArtifactError) as excinfo:
        source.read_field(source.field_ids()[0])
    msg = str(excinfo.value).lower()
    assert "z" in msg
    assert "voxel_size" in msg or "spacing" in msg


def test_negative_z_spacing_raises_naming_z():
    sizes = {"Z": 4, "C": 1, "Y": 8, "X": 8}
    data = np.zeros(tuple(sizes.values()), dtype=np.uint16)
    fake = FakeND2File(sizes, data, FakeVoxelSize(x=0.1, y=0.1, z=-0.5))
    config = make_config(channel_ids=("a",), source_indices=(0,))
    source = ND2Source("fake.nd2", config, reader_factory=reader_factory_for(fake))

    with pytest.raises(ArtifactError) as excinfo:
        source.read_field(source.field_ids()[0])
    assert "z" in str(excinfo.value).lower()


# ─── dtype preservation ─────────────────────────────────────────────────────


def test_source_dtype_is_preserved_not_converted():
    sizes = {"C": 1, "Z": 3, "Y": 4, "X": 4}
    data = (np.random.default_rng(0).integers(0, 60000, size=tuple(sizes.values()))).astype(
        np.uint16
    )
    fake = FakeND2File(sizes, data, FakeVoxelSize(x=0.1, y=0.1, z=0.5))
    config = make_config(channel_ids=("a",), source_indices=(0,))
    source = ND2Source("fake.nd2", config, reader_factory=reader_factory_for(fake))

    vol = source.read_field(source.field_ids()[0])
    assert vol.data.dtype == np.uint16
    assert vol.data.dtype != np.uint8


# ─── channel binding ────────────────────────────────────────────────────────


def test_channel_bindings_come_from_config_with_source_names():
    sizes = {"C": 3, "Z": 2, "Y": 4, "X": 4}
    data = np.zeros(tuple(sizes.values()), dtype=np.uint16)
    fake = FakeND2File(
        sizes,
        data,
        FakeVoxelSize(x=0.1, y=0.1, z=0.5),
        channel_names=["GFP", "RFP", "DAPI"],
    )
    config = make_config(
        channel_ids=("membrane", "signal", "nucleus"), source_indices=(0, 1, 2)
    )
    source = ND2Source("fake.nd2", config, reader_factory=reader_factory_for(fake))

    vol = source.read_field(source.field_ids()[0])
    by_id = {c.channel_id: c for c in vol.channels}
    assert by_id["membrane"].source_index == 0
    assert by_id["membrane"].source_name == "GFP"
    assert by_id["signal"].source_name == "RFP"
    assert by_id["nucleus"].source_name == "DAPI"


def test_config_binding_more_channels_than_available_raises():
    sizes = {"C": 2, "Z": 2, "Y": 4, "X": 4}
    data = np.zeros(tuple(sizes.values()), dtype=np.uint16)
    fake = FakeND2File(sizes, data, FakeVoxelSize(x=0.1, y=0.1, z=0.5))
    # config expects 3 channels (indices 0,1,2) but the nd2 field has only 2
    config = make_config(
        channel_ids=("membrane", "signal", "nucleus"), source_indices=(0, 1, 2)
    )
    source = ND2Source("fake.nd2", config, reader_factory=reader_factory_for(fake))

    with pytest.raises(ArtifactError) as excinfo:
        source.read_field(source.field_ids()[0])
    msg = str(excinfo.value)
    assert "2" in msg  # names the actual available count


# ─── identity / content hash stability ──────────────────────────────────────


def test_field_identity_and_hash_stable_across_two_reads():
    sizes = {"C": 2, "Z": 3, "Y": 8, "X": 8}
    data = np.arange(np.prod(list(sizes.values())), dtype=np.uint16).reshape(
        tuple(sizes.values())
    )
    fake = FakeND2File(sizes, data, FakeVoxelSize(x=0.1, y=0.1, z=0.5))
    config = make_config(channel_ids=("a", "b"), source_indices=(0, 1))
    source = ND2Source(
        "fake.nd2", config, dataset_id="ds1", reader_factory=reader_factory_for(fake)
    )

    fid = source.field_ids()[0]
    vol1 = source.read_field(fid)
    vol2 = source.read_field(fid)

    assert vol1.identity == vol2.identity
    assert vol1.identity.image_content_sha256 == vol2.identity.image_content_sha256
    assert vol1.identity.field_id == vol2.identity.field_id == fid

    # A second, independently-constructed source over the same fake data must
    # agree too -- the hash is a property of content, not of object identity.
    source2 = ND2Source(
        "fake.nd2", config, dataset_id="ds1", reader_factory=reader_factory_for(fake)
    )
    vol3 = source2.read_field(source2.field_ids()[0])
    assert vol3.identity.image_content_sha256 == vol1.identity.image_content_sha256


def test_different_content_yields_different_hash():
    sizes = {"C": 1, "Z": 2, "Y": 4, "X": 4}
    data_a = np.zeros(tuple(sizes.values()), dtype=np.uint16)
    data_b = np.ones(tuple(sizes.values()), dtype=np.uint16)
    config = make_config(channel_ids=("a",), source_indices=(0,))

    fake_a = FakeND2File(sizes, data_a, FakeVoxelSize(x=0.1, y=0.1, z=0.5))
    fake_b = FakeND2File(sizes, data_b, FakeVoxelSize(x=0.1, y=0.1, z=0.5))
    source_a = ND2Source("fake.nd2", config, reader_factory=reader_factory_for(fake_a))
    source_b = ND2Source("fake.nd2", config, reader_factory=reader_factory_for(fake_b))

    hash_a = source_a.read_field(source_a.field_ids()[0]).identity.image_content_sha256
    hash_b = source_b.read_field(source_b.field_ids()[0]).identity.image_content_sha256
    assert hash_a != hash_b


# ─── singleton non-canonical axes (e.g. a T loop) ──────────────────────────


def test_singleton_t_axis_is_tolerated():
    sizes = {"T": 1, "C": 2, "Z": 3, "Y": 4, "X": 4}
    data = np.zeros(tuple(sizes.values()), dtype=np.uint16)
    fake = FakeND2File(sizes, data, FakeVoxelSize(x=0.1, y=0.1, z=0.5))
    config = make_config(channel_ids=("a", "b"), source_indices=(0, 1))
    source = ND2Source("fake.nd2", config, reader_factory=reader_factory_for(fake))

    vol = source.read_field(source.field_ids()[0])
    assert vol.data.shape == (2, 3, 4, 4)


def test_non_singleton_extra_axis_raises():
    sizes = {"T": 2, "C": 2, "Z": 3, "Y": 4, "X": 4}
    data = np.zeros(tuple(sizes.values()), dtype=np.uint16)
    fake = FakeND2File(sizes, data, FakeVoxelSize(x=0.1, y=0.1, z=0.5))
    config = make_config(channel_ids=("a", "b"), source_indices=(0, 1))
    source = ND2Source("fake.nd2", config, reader_factory=reader_factory_for(fake))

    with pytest.raises(ArtifactError) as excinfo:
        source.read_field(source.field_ids()[0])
    assert "T" in str(excinfo.value)


# ─── extract() command ──────────────────────────────────────────────────────


def test_extract_writes_one_artifact_per_field_via_injected_writer(tmp_path):
    sizes = {"P": 2, "C": 2, "Z": 3, "Y": 4, "X": 4}
    data = np.zeros(tuple(sizes.values()), dtype=np.uint16)
    fake = FakeND2File(sizes, data, FakeVoxelSize(x=0.1, y=0.1, z=0.5))
    config = make_config(channel_ids=("a", "b"), source_indices=(0, 1))
    source = ND2Source("fake.nd2", config, reader_factory=reader_factory_for(fake))

    written_calls = []

    def fake_writer(vol, path, chunks, compression_level):
        written_calls.append((vol, path, chunks, compression_level))
        return path

    result = extract(
        "fake.nd2",
        tmp_path / "out_dir",
        config,
        source=source,
        writer=fake_writer,
    )

    assert len(result) == 2
    assert len(written_calls) == 2
    for vol, path, chunks, level in written_calls:
        assert chunks == config.artifacts.image_chunks_czyx
        assert level == config.artifacts.compression_level
        assert path.suffix == ".zarr"


def test_extract_without_writer_raises_clear_error_if_zarr_store_missing(tmp_path):
    sizes = {"C": 1, "Z": 2, "Y": 4, "X": 4}
    data = np.zeros(tuple(sizes.values()), dtype=np.uint16)
    fake = FakeND2File(sizes, data, FakeVoxelSize(x=0.1, y=0.1, z=0.5))
    config = make_config(channel_ids=("a",), source_indices=(0,))
    source = ND2Source("fake.nd2", config, reader_factory=reader_factory_for(fake))

    try:
        import s2_adhesion.io.zarr_store  # noqa: F401

        pytest.skip("zarr_store is importable in this environment; nothing to test here")
    except ImportError:
        pass

    with pytest.raises(ArtifactError):
        extract("fake.nd2", tmp_path / "out_dir", config, source=source)


# ─── no heavy/ML imports ────────────────────────────────────────────────────


def test_importing_modules_does_not_import_torch_or_cellpose():
    code = (
        "import sys\n"
        "import s2_adhesion.io.nd2_source\n"
        "import s2_adhesion.commands.extract\n"
        "assert 'torch' not in sys.modules, sys.modules.keys()\n"
        "assert 'cellpose' not in sys.modules, sys.modules.keys()\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout
