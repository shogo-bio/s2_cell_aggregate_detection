"""VolumeSource backed by a Nikon ND2 file.

The historical script (``detect_aggregations.py``) read only ``voxel_size().x``
and silently assumed Z spacing did not matter. Confocal Z steps are typically
5-10x the XY pixel size, so that assumption corrupted every downstream 3D
measurement (volume, surface area, sphericity). This module reads x, y AND z
and fails loudly -- naming the axis -- when z is missing or non-positive.

No eager import of the ``nd2`` package: it is an optional dependency
(``pip install s2-adhesion[nd2]``) and the measurement layer must keep working
on a machine that never installed it. ``nd2`` is imported lazily, only when a
real file is actually opened, so importing this module is always safe. Tests
never touch a real ``.nd2`` file -- they inject a ``reader_factory`` that
yields duck-typed fake objects exposing ``.sizes``, ``.asarray()``,
``.voxel_size()`` and ``.metadata``, the same surface as ``nd2.ND2File``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from ..config import PipelineConfig
from ..contracts import (
    ChannelBinding,
    FieldIdentity,
    ImageVolume,
    VoxelGeometry,
)
from ..errors import ArtifactError

# Canonical axis order every ImageVolume must end up in. See contracts.py.
_CANONICAL_AXES = "CZYX"


@runtime_checkable
class Nd2Like(Protocol):
    """Structural shape this module actually reads.

    Satisfied by ``nd2.ND2File`` and by the fakes used in tests -- nothing here
    requires the real ``nd2`` package to be installed.
    """

    sizes: Mapping[str, int]
    metadata: Any

    def asarray(self) -> Any: ...
    def voxel_size(self) -> Any: ...


ReaderFactory = Callable[[Path], AbstractContextManager[Nd2Like]]


def _default_reader_factory(path: Path) -> AbstractContextManager[Nd2Like]:
    """Open a real nd2 file. The only place ``nd2`` is imported."""
    import nd2  # lazy: optional dependency, only needed to read a real file

    return nd2.ND2File(path)


class ND2Source:
    """Hands out :class:`ImageVolume` fields from a single (possibly
    multi-position) nd2 file.

    Field ids are stable strings, never assumed to correspond to a numeric
    ``P`` axis: a file with no ``P`` axis yields exactly one field.
    """

    def __init__(
        self,
        path: str | Path,
        config: PipelineConfig,
        *,
        dataset_id: str | None = None,
        reader_factory: ReaderFactory = _default_reader_factory,
        max_fields: int | None = None,
    ) -> None:
        self._path = Path(path)
        self._config = config
        self._dataset_id = dataset_id if dataset_id is not None else self._path.stem
        self._reader_factory = reader_factory
        # Cap the number of fields exposed, so the whole pipeline can be tried on
        # a couple of fields before a long full run. Field ids keep their normal
        # zero-padded width (based on the true field count), so a later full run
        # produces the same ids for the same fields.
        self._max_fields = max_fields
        self._field_ids: tuple[str, ...] | None = None

    def field_ids(self) -> Sequence[str]:
        if self._field_ids is None:
            with self._reader_factory(self._path) as reader:
                sizes = dict(reader.sizes)
            n_fields = sizes.get("P", 1)
            width = max(3, len(str(max(n_fields - 1, 0))))
            ids = tuple(f"field{i:0{width}d}" for i in range(n_fields))
            if self._max_fields is not None:
                ids = ids[: self._max_fields]
            self._field_ids = ids
        return self._field_ids

    def read_field(self, field_id: str) -> ImageVolume:
        ids = self.field_ids()
        try:
            field_index = ids.index(field_id)
        except ValueError as exc:
            raise ArtifactError(
                f"unknown field_id {field_id!r} for {self._path}; known fields: "
                f"{list(ids)}"
            ) from exc

        with self._reader_factory(self._path) as reader:
            sizes = dict(reader.sizes)
            raw = np.asarray(reader.asarray())
            voxel = reader.voxel_size()
            metadata = getattr(reader, "metadata", None)

        spacing_zyx = _read_spacing_zyx(voxel, source=str(self._path))
        canonical = _to_canonical_czyx(raw, sizes, field_index, source=str(self._path))

        channel_names = _extract_channel_names(metadata)
        channels = _bind_channels(
            self._config.channels, canonical.shape[0], channel_names, source=str(self._path)
        )

        content_hash = hashlib.sha256(canonical.tobytes()).hexdigest()
        identity = FieldIdentity(
            dataset_id=self._dataset_id,
            field_id=field_id,
            source_uri=str(self._path),
            source_field_index=field_index,
            image_content_sha256=content_hash,
        )

        return ImageVolume(
            data=canonical,
            geometry=VoxelGeometry(spacing_um_zyx=spacing_zyx),
            channels=channels,
            identity=identity,
        )


# ─── helpers ────────────────────────────────────────────────────────────────


def _read_spacing_zyx(voxel: Any, *, source: str) -> tuple[float, float, float]:
    """Read x, y AND z from ``voxel_size()``. Never defaults a missing axis.

    The old script read only ``.x`` and reused it everywhere; Z spacing on a
    confocal stack is typically 5-10x coarser than XY, so a missing or
    zero/negative Z step must fail loudly rather than silently corrupt every
    3D metric downstream.
    """
    x = getattr(voxel, "x", None)
    y = getattr(voxel, "y", None)
    z = getattr(voxel, "z", None)

    if z is None:
        raise ArtifactError(
            f"{source}: voxel_size() has no z spacing. Every 3D measurement "
            "(volume, surface area, sphericity) depends on dz; refusing to "
            "default it to 1.0 or to the XY pixel size."
        )
    if x is None or y is None:
        missing = [name for name, v in (("x", x), ("y", y)) if v is None]
        raise ArtifactError(
            f"{source}: voxel_size() is missing in-plane spacing for {missing}"
        )
    if z <= 0:
        raise ArtifactError(
            f"{source}: voxel_size().z = {z!r} is not strictly positive. A "
            "zero or negative Z step means the acquisition's Z metadata is "
            "broken -- every 3D measurement depends on dz, so this is not a "
            "value that can be silently defaulted."
        )
    if x <= 0 or y <= 0:
        raise ArtifactError(
            f"{source}: voxel_size() has non-positive in-plane spacing "
            f"(x={x!r}, y={y!r})"
        )
    return float(z), float(y), float(x)


def _to_canonical_czyx(
    raw: NDArray[np.generic],
    sizes: Mapping[str, int],
    field_index: int,
    *,
    source: str,
) -> NDArray[np.generic]:
    """Transpose an arbitrarily-ordered nd2 array into CZYX.

    ``sizes`` keys may include any subset of P/T/C/Z/Y/X in any order; axis
    positions are always looked up by key, never assumed by position.
    """
    dim_keys = list(sizes.keys())

    if "P" in dim_keys:
        p_axis = dim_keys.index("P")
        raw = np.take(raw, field_index, axis=p_axis)
        dim_keys = [k for k in dim_keys if k != "P"]

    # Any other non-canonical axis (e.g. a T loop) is tolerated only if it is
    # a singleton -- ImageVolume has no time axis to put a real one in.
    for k in [k for k in dim_keys if k not in _CANONICAL_AXES]:
        axis = dim_keys.index(k)
        if raw.shape[axis] != 1:
            raise ArtifactError(
                f"{source}: nd2 dimension {k!r} has size {raw.shape[axis]} > 1; "
                "only C/Z/Y/X may vary within a field (plus a singleton on any "
                f"other axis). Full sizes: {dict(sizes)!r}"
            )
        raw = np.squeeze(raw, axis=axis)
        dim_keys.pop(axis)

    missing_spatial = [k for k in "YX" if k not in dim_keys]
    if missing_spatial:
        raise ArtifactError(
            f"{source}: nd2 sizes {dict(sizes)!r} has no {missing_spatial} axis"
        )

    # C and Z are allowed to be absent (single-channel / single-plane
    # acquisitions omit singleton axes from nd2's own `sizes`); insert them.
    for k in "CZ":
        if k not in dim_keys:
            raw = raw[np.newaxis, ...]
            dim_keys.insert(0, k)

    order = [dim_keys.index(k) for k in _CANONICAL_AXES]
    return np.transpose(raw, order)


def _extract_channel_names(metadata: Any) -> list[str | None]:
    channels = getattr(metadata, "channels", None) if metadata is not None else None
    if not channels:
        return []
    names: list[str | None] = []
    for ch in channels:
        inner = getattr(ch, "channel", None)
        names.append(getattr(inner, "name", None) if inner is not None else None)
    return names


def _bind_channels(
    configured: tuple[ChannelBinding, ...],
    n_channels_available: int,
    source_names: Sequence[str | None],
    *,
    source: str,
) -> tuple[ChannelBinding, ...]:
    if not configured:
        raise ArtifactError(f"{source}: PipelineConfig.channels is empty; nothing to bind")

    max_index = max(c.source_index for c in configured)
    if max_index >= n_channels_available:
        raise ArtifactError(
            f"{source}: config binds channel source_index {max_index} but this "
            f"nd2 field has only {n_channels_available} channel(s)"
        )

    bound: list[ChannelBinding] = []
    for c in configured:
        name = (
            source_names[c.source_index]
            if c.source_index < len(source_names)
            else None
        )
        bound.append(
            ChannelBinding(
                channel_id=c.channel_id,
                source_index=c.source_index,
                roles=c.roles,
                source_name=name if name is not None else c.source_name,
                color=c.color,
                population=c.population,
                protein=c.protein,
            )
        )
    return tuple(bound)
