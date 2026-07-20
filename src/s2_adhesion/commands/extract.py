"""Extract every field of an nd2 file to on-disk image artifacts.

The zarr writer (``s2_adhesion.io.zarr_store.write_image_volume``) is imported
lazily, inside :func:`extract`, not at module import time: another agent owns
that module and it may not exist yet, and this module must still be safely
importable (and unit-testable via an injected fake writer) regardless.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..config import PipelineConfig
from ..contracts import ImageVolume, VolumeSource
from ..errors import ArtifactError
from ..io.nd2_source import ND2Source


class ImageVolumeWriter(Protocol):
    """Matches ``s2_adhesion.io.zarr_store.write_image_volume``."""

    def __call__(
        self,
        vol: ImageVolume,
        path: Path,
        chunks: tuple[int, ...],
        compression_level: int,
    ) -> Path: ...


def _default_writer() -> ImageVolumeWriter:
    try:
        from ..io.zarr_store import write_image_volume
    except ImportError as exc:
        raise ArtifactError(
            "s2_adhesion.io.zarr_store.write_image_volume is not importable; "
            "the zarr writer is required to extract nd2 fields to disk"
        ) from exc
    return write_image_volume


def extract(
    nd2_path: str | Path,
    output_dir: str | Path,
    config: PipelineConfig,
    *,
    source: VolumeSource | None = None,
    writer: ImageVolumeWriter | None = None,
) -> list[Path]:
    """Read every field of ``nd2_path`` and write each as an image artifact.

    Returns the list of written artifact paths, one per field, in field order.
    ``source`` and ``writer`` are injection points for testing; production
    callers leave both as their defaults (a real :class:`ND2Source` and the
    lazily-imported zarr writer).
    """
    nd2_path = Path(nd2_path)
    output_dir = Path(output_dir)

    write = writer if writer is not None else _default_writer()
    vol_source = source if source is not None else ND2Source(nd2_path, config)

    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for field_id in vol_source.field_ids():
        volume = vol_source.read_field(field_id)
        dest = output_dir / f"{field_id}.zarr"
        written_path = write(
            volume,
            dest,
            chunks=config.artifacts.image_chunks_czyx,
            compression_level=config.artifacts.compression_level,
        )
        written.append(written_path)
    return written
