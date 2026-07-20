"""``segment``: read an image artifact, run the configured segmentation
strategy, write a label artifact.

THE ONLY command in this package permitted to require an ML environment
(cellpose/torch). It gets that dependency exclusively through
``segmentation.factory.create_cellpose_engine``, which itself never imports
``cellpose``/``torch`` at module level -- only inside that one function, and
only after confirming the installed cellpose major version matches what is
configured (see ``segmentation/factory.py``'s docstring). If cellpose is
missing, or is installed at a mismatched major version,
``create_cellpose_engine`` raises :class:`~s2_adhesion.errors.MLDependencyError`
-- an actionable, typed error naming both the configured and installed
majors -- which this module lets propagate unchanged rather than catching
and re-raising as a generic ``ImportError``.

``commands.measure`` (the counterpart module) must NEVER import this module,
or anything it imports here (``segmentation.factory``,
``segmentation.direct_cellpose``, ``segmentation.nuclear_watershed``), even
transitively -- that is the whole point of the split-machine architecture
this module and ``commands.measure`` implement together.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from ..config import (
    DirectInstanceConfig,
    DirectStarDistConfig,
    NucleusSeededWatershedConfig,
    PipelineConfig,
)
from ..errors import ConfigError
from ..io.zarr_store import read_image_volume, write_label_volume
from ..segmentation.direct_cellpose import DirectCellposeBackend
from ..segmentation.factory import create_cellpose_engine, create_stardist_engine
from ..segmentation.nuclear_watershed import NucleusSeededWatershedBackend
from ..segmentation.protocol import SegmentationBackend, SegmentationRequest

__all__ = ["segment"]


def _build_backend(config: PipelineConfig) -> SegmentationBackend:
    """Build the configured :class:`SegmentationBackend`, injecting real
    cellpose engines from ``segmentation.factory``.

    This is the one place in this module that actually needs cellpose
    installed (via ``create_cellpose_engine``) -- called here, at ``segment``
    call time, never at import time.
    """
    seg = config.segmentation
    if seg is None:
        raise ConfigError(
            "segment() requires config.segmentation to be set (analysis_backend "
            f"{config.analysis_backend!r} needs a segmentation block)"
        )
    if isinstance(seg, DirectInstanceConfig):
        engine = create_cellpose_engine(seg.model)
        return DirectCellposeBackend(config=seg, engine=engine)
    if isinstance(seg, DirectStarDistConfig):
        # StarDist lives in its own environment (TensorFlow, vs cellpose's
        # PyTorch), so this import is deferred exactly like the cellpose one --
        # a machine set up for one backend must not need the other installed.
        from ..segmentation.direct_stardist import DirectStarDistBackend

        engine = create_stardist_engine(seg.model)
        return DirectStarDistBackend(config=seg, engine=engine)
    if isinstance(seg, NucleusSeededWatershedConfig):
        seed_engine = create_cellpose_engine(seg.seed_model)
        extent_engine = None
        if seg.extent_mode == "cellpose_union" and seg.extent_model is not None:
            extent_engine = create_cellpose_engine(seg.extent_model)
        return NucleusSeededWatershedBackend(
            config=seg, seed_engine=seed_engine, extent_engine=extent_engine
        )
    raise ConfigError(f"segment(): unsupported segmentation config {seg!r}")  # pragma: no cover


def _input_image_artifact_id(image_artifact_dir: Path) -> str | None:
    """Best-effort ``artifact_id`` of the source image artifact, for provenance.

    Reads the manifest directly (rather than through ``read_image_volume``,
    which reconstructs an ``ImageVolume`` and does not carry ``artifact_id``
    along). Never fatal: a missing/unreadable manifest here just means the
    label artifact's ``input_image_artifact_id`` is recorded ``None`` --
    ``input_image_sha256`` (the field the binding guard actually checks) is
    always set from the in-memory ``ImageVolume`` regardless.
    """
    manifest_path = image_artifact_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    artifact_id = manifest.get("artifact_id")
    return artifact_id if isinstance(artifact_id, str) else None


def segment(
    image_artifact_dir: Path | str,
    output_dir: Path | str,
    config: PipelineConfig,
) -> Path:
    """Segment one image artifact; write one label artifact.

    Reads ``image_artifact_dir`` (via ``io.zarr_store.read_image_volume``),
    builds the segmentation backend named by ``config.segmentation``
    (``direct_cellpose`` or ``nucleus_seeded_watershed``), runs it, and
    writes the resulting :class:`~s2_adhesion.contracts.LabelVolume` to
    ``output_dir`` (the exact artifact path, mirroring
    ``io.zarr_store.write_image_volume``'s one-artifact-per-call
    convention -- not a parent directory ``segment`` invents a name under).
    Returns ``output_dir``.
    """
    image_artifact_dir = Path(image_artifact_dir)
    output_dir = Path(output_dir)

    image = read_image_volume(image_artifact_dir, verify_hashes=config.artifacts.verify_content_hashes)
    backend = _build_backend(config)

    run_id = f"run-{uuid.uuid4().hex[:16]}"
    result = backend.segment(SegmentationRequest(image=image, run_id=run_id))

    return write_label_volume(
        result.labels,
        output_dir,
        chunks=config.artifacts.label_chunks_zyx,
        compression_level=config.artifacts.compression_level,
        input_image_artifact_id=_input_image_artifact_id(image_artifact_dir),
    )
