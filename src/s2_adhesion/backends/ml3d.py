"""``MLInstance3DBackend``: the ``analysis_backend="ml_instance_3d"`` whole-run
backend -- segment every field of a source, then measure it, in one process.

ASSUMPTION, stated up front because it shapes everything below: this branch
has no ``backends/protocol.py`` and no ``AnalysisBackend`` protocol (checked
by directory listing before writing this module -- ``backends/`` contains
only ``__init__.py`` besides this file). So there is no shared protocol to
implement here. Instead this module DEFINES the shape every analysis backend
on this branch follows: a ``backend_id: str`` attribute, and a

    run(self, source: VolumeSource, *, config: PipelineConfig, output_dir: Path) -> RunArtifacts

method. ``RunArtifacts`` is imported from ``..commands.measure`` rather than
defined here -- see that module's docstring for why (measure's import graph
must stay ML-free, so it, not this ML-capable module, owns the type). A
concurrent agent writing ``backends/legacy.py`` (the
``analysis_backend="legacy_threshold_2d"`` counterpart) has been told to
match this same ``backend_id`` + ``run(...)`` shape.

Unlike ``commands.measure``, this module is NOT required to import cleanly
without torch/cellpose -- it is the ML backend, and calling ``run`` on a
config that requests a segmentation strategy necessarily needs cellpose
installed (via ``commands.segment`` -> ``segmentation.factory``). What is
still required (matching ``backends/__init__.py``'s docstring, "No eager
imports") is that *importing* this module does not itself force torch/
cellpose into ``sys.modules`` -- and it does not: every module imported here
at module level (``commands.segment``, ``commands.measure``,
``segmentation.factory``/``direct_cellpose``/``nuclear_watershed``/
``protocol``) imports cellpose/torch only lazily, inside
``segmentation.factory.create_cellpose_engine``, called only when ``run`` is
actually invoked.

Per field of ``source``: write an image artifact, segment it to a label
artifact (both persisted to ``output_dir``, so the same on-disk trail
``commands.segment``/``commands.measure`` would leave running as separate
processes is left here too), then compute that field's
:class:`~s2_adhesion.contracts.MeasurementBundle` directly via
``metrics.engine.compute_measurements`` (in-process, not by shelling out
through ``commands.measure`` again -- this backend already has both the
``LabelVolume`` and the ``ImageVolume`` in memory, so re-reading them back
from disk would be pure overhead). Every field's bundle is merged
(``metrics.records.merge_bundles``) and written once as one combined set of
CSVs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import PipelineConfig
from ..contracts import VolumeSource
from ..io.zarr_store import read_label_volume, write_image_volume
from ..metrics.engine import compute_measurements
from ..metrics.records import merge_bundles
from ..commands.measure import RunArtifacts, _write_bundle_atomically
from ..commands.segment import segment

__all__ = ["MLInstance3DBackend"]

_BACKEND_ID = "ml_instance_3d"


@dataclass(frozen=True, slots=True)
class MLInstance3DBackend:
    """Runs ``direct_cellpose``/``nucleus_seeded_watershed`` + full metrics
    over every field of a :class:`~s2_adhesion.contracts.VolumeSource`.

    See the module docstring for the ``run(source, *, config, output_dir) ->
    RunArtifacts`` shape this establishes and the ``ASSUMPTION`` behind it.
    """

    backend_id: str = _BACKEND_ID

    def run(
        self,
        source: VolumeSource,
        *,
        config: PipelineConfig,
        output_dir: Path | str,
    ) -> RunArtifacts:
        output_dir = Path(output_dir)
        images_dir = output_dir / "images"
        labels_dir = output_dir / "labels"
        # A dedicated subdirectory, NOT output_dir itself: the atomic swap
        # below (_write_bundle_atomically) replaces its whole target
        # directory wholesale (see io.zarr_store._replace_dir / its local
        # port in commands.measure). Writing the CSVs straight into
        # output_dir would make that swap clobber the images/ and labels/
        # artifact trees already written above.
        measurements_dir = output_dir / "measurements"

        bundles = []
        warnings: list[str] = []
        label_artifact_dirs: list[Path] = []

        for field_id in source.field_ids():
            image = source.read_field(field_id)

            image_artifact = write_image_volume(
                image,
                images_dir / f"{field_id}.image.ome.zarr",
                chunks=config.artifacts.image_chunks_czyx,
                compression_level=config.artifacts.compression_level,
            )

            label_artifact = segment(
                image_artifact, labels_dir / f"{field_id}.labels.ome.zarr", config
            )
            label_artifact_dirs.append(label_artifact)

            labels = read_label_volume(
                label_artifact,
                verify_hashes=config.artifacts.verify_content_hashes,
                image=image,
            )

            bundle = compute_measurements(
                labels=labels,
                image=image,
                config=config.measurement,
                backend_id=self.backend_id,
            )
            bundles.append(bundle)
            warnings.extend(bundle.warnings)

        merged = merge_bundles(*bundles)
        paths = _write_bundle_atomically(merged, measurements_dir)

        return RunArtifacts(
            output_dir=measurements_dir,
            objects_csv=paths["objects"],
            contacts_csv=paths["contacts"],
            aggregates_csv=paths["aggregates"],
            localization_profiles_csv=paths["localization_profiles"],
            metrics_manifest_json=paths["metrics_manifest"],
            run_manifest_json=paths["run_manifest"],
            label_artifact_dirs=tuple(label_artifact_dirs),
            warnings=tuple(warnings),
        )
