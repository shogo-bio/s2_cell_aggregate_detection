"""Shared fixtures. Sequentially owned -- parallel agents must not edit this file.

Everything here is synthetic and analytic. There is no .nd2 in the repository and
none is needed: geometric metrics are validated against closed-form ground truth,
which is stricter than comparing against real data nobody has annotated.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from s2_adhesion.contracts import (
    ChannelBinding,
    ChannelRole,
    FieldIdentity,
    ImageVolume,
    LabelVolume,
    SegmentationProvenance,
    VoxelGeometry,
)

# Realistic confocal sampling: Z far coarser than XY.
ANISOTROPIC = (0.5, 0.1, 0.1)
ISOTROPIC = (0.1, 0.1, 0.1)


@pytest.fixture
def anisotropic_geometry() -> VoxelGeometry:
    return VoxelGeometry(spacing_um_zyx=ANISOTROPIC)


@pytest.fixture
def isotropic_geometry() -> VoxelGeometry:
    return VoxelGeometry(spacing_um_zyx=ISOTROPIC)


def make_identity(field_id: str = "field00", content: bytes = b"") -> FieldIdentity:
    return FieldIdentity(
        dataset_id="synthetic",
        field_id=field_id,
        source_uri="memory://synthetic",
        source_field_index=0,
        image_content_sha256=hashlib.sha256(content).hexdigest(),
    )


def make_provenance(image_sha: str, run_id: str = "run0") -> SegmentationProvenance:
    return SegmentationProvenance(
        run_id=run_id,
        backend_id="synthetic",
        strategy="synthetic",
        config_sha256=hashlib.sha256(b"cfg").hexdigest(),
        input_image_sha256=image_sha,
        device="cpu",
        host_platform="test",
    )


def make_label_volume(
    cells: np.ndarray,
    spacing: tuple[float, float, float] = ANISOTROPIC,
    nuclei: np.ndarray | None = None,
) -> LabelVolume:
    """Wrap a raw ZYX label array in a valid LabelVolume."""
    identity = make_identity(content=cells.tobytes())
    return LabelVolume(
        cells=cells.astype(np.uint32),
        geometry=VoxelGeometry(spacing_um_zyx=spacing),
        identity=identity,
        provenance=make_provenance(identity.image_content_sha256),
        nuclei=None if nuclei is None else nuclei.astype(np.uint32),
    )


def make_image_volume(
    data: np.ndarray,
    spacing: tuple[float, float, float] = ANISOTROPIC,
    channel_ids: tuple[str, ...] = ("membrane", "signal", "nucleus"),
    roles: tuple[ChannelRole, ...] = (
        ChannelRole.MEMBRANE,
        ChannelRole.SIGNAL,
        ChannelRole.NUCLEUS,
    ),
) -> ImageVolume:
    """Wrap a raw CZYX array in a valid ImageVolume."""
    channels = tuple(
        ChannelBinding(channel_id=cid, source_index=i, roles=frozenset({role}))
        for i, (cid, role) in enumerate(zip(channel_ids, roles))
    )
    return ImageVolume(
        data=data,
        geometry=VoxelGeometry(spacing_um_zyx=spacing),
        channels=channels[: data.shape[0]],
        identity=make_identity(content=data.tobytes()),
    )
