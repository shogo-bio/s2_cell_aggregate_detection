"""``measure``: read a label artifact (+ optional image artifact), compute
metrics, write CSVs.

THE GUARANTEE THIS MODULE EXISTS TO PROVIDE: segmentation is far too slow to
run on the machine doing measurement (cellpose 4 needed 638 s for a single
256x256 plane on the reference CPU host; cellpose 3 is roughly 290x faster
but still slow), so labels are routinely produced on a GPU machine and
measured on a laptop or CI runner with no torch/cellpose installed at all.
That only works if this module's import graph never needs them.

Consequently this module -- and every module it imports at module level --
MUST NOT import ``s2_adhesion.segmentation.factory``,
``s2_adhesion.segmentation.direct_cellpose``,
``s2_adhesion.segmentation.nuclear_watershed``,
``s2_adhesion.segmentation.cellpose_v3``/``cellpose_v4``, ``torch``, or
``cellpose`` -- not even lazily, not even in a ``try/except``. Every import
here is config/contracts/io.tables/io.zarr_store/metrics.engine -- all of
which are themselves ML-import-free (see each of those modules' own
docstrings). ``tests/integration/test_measure_without_ml.py`` enforces this
for real, in a subprocess where ``import torch``/``import cellpose`` are
made to raise ``ImportError`` by an import hook -- not merely by checking
``sys.modules``.

``RunArtifacts`` (the return type) is defined HERE, not in
``backends/ml3d.py``, precisely because of the constraint above:
``backends/ml3d.py`` is the ML-capable backend and necessarily imports
segmentation machinery, so it cannot be the module ``measure`` depends on.
ASSUMPTION (no ``backends/protocol.py`` / ``AnalysisBackend`` protocol exists
on this branch, confirmed by directory listing before writing this module):
``backends/ml3d.py`` instead imports ``RunArtifacts`` from here.

``ArtifactBindingError`` (label/image mismatch: shape, spacing, or content
hash) is raised from inside ``io.zarr_store.read_label_volume`` -- BEFORE
this function calls anything that writes to ``output_dir`` -- so a mismatch
never leaves a half-written CSV behind and never creates ``output_dir`` at
all.

ATOMICITY OF THE WRITE ITSELF: ``io.tables.write_measurement_bundle`` writes
five files (four CSVs + a metrics manifest) one at a time with no atomicity
of its own -- a failure after the third file leaves the first two sitting in
``output_dir`` looking like real output. This module does not own
``io.tables`` and cannot change that, so it wraps the call the same way
``io.zarr_store`` wraps its own array writes: build the whole bundle in a
temporary sibling directory, write a ``run_manifest.json`` recording success
into it LAST, and only then atomically swap the temp directory into
``output_dir`` (``_replace_dir`` below is a local port of
``io.zarr_store._replace_dir``'s Windows-safe swap -- that function is
private to ``zarr_store`` and not this module's to import). Any exception
during that whole sequence discards the temp directory and leaves
``output_dir`` exactly as it was before the call -- never a partial write,
and never a ``run_manifest.json`` claiming success for a run that didn't
finish.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..config import PipelineConfig
from ..contracts import MeasurementBundle
from ..io.tables import write_measurement_bundle
from ..io.zarr_store import read_image_volume, read_label_volume
from ..metrics.engine import compute_measurements

__all__ = ["RunArtifacts", "measure"]

_RUN_MANIFEST_NAME = "run_manifest.json"


@dataclass(frozen=True, slots=True)
class RunArtifacts:
    """Paths written by one run of :func:`measure` (or
    ``backends.ml3d.MLInstance3DBackend.run``, which merges several fields'
    bundles and writes one combined set of these).

    ASSUMPTION: no ``backends/protocol.py`` / ``AnalysisBackend`` protocol
    exists on this branch, so this shape -- a ``backend_id`` attribute plus a
    ``run(source, *, config, output_dir) -> RunArtifacts``-shaped method on
    the backend side -- is established here rather than inherited from a
    shared protocol. A concurrent agent writing ``backends/legacy.py`` has
    been told to match it.
    """

    output_dir: Path
    objects_csv: Path
    contacts_csv: Path
    aggregates_csv: Path
    localization_profiles_csv: Path
    metrics_manifest_json: Path
    run_manifest_json: Path
    label_artifact_dirs: tuple[Path, ...] = ()
    warnings: tuple[str, ...] = ()


def _replace_dir(src: Path, dst: Path) -> None:
    """Swap ``src`` into ``dst``, Windows-safely. Local port of
    ``io.zarr_store._replace_dir`` -- see this module's docstring for why it
    is duplicated here rather than imported.
    """
    if not dst.exists():
        os.replace(src, dst)
        return
    backup = dst.parent / f".{dst.name}.bak-{uuid.uuid4().hex}"
    os.replace(dst, backup)
    try:
        os.replace(src, dst)
    except Exception:
        os.replace(backup, dst)
        raise
    shutil.rmtree(backup, ignore_errors=True)


def _write_bundle_atomically(bundle: MeasurementBundle, output_dir: Path) -> dict[str, Path]:
    """Write every CSV + manifest for ``bundle`` into ``output_dir``, all at
    once or not at all. Returns final (post-swap) paths keyed like
    ``io.tables.write_measurement_bundle``, plus ``"run_manifest"``.
    """
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir.parent / f".{output_dir.name}.tmp-{uuid.uuid4().hex}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    try:
        write_measurement_bundle(bundle, tmp_dir)
        run_manifest = {
            "status": "success",
            "written_utc": datetime.now(timezone.utc).isoformat(),
            "n_objects": len(bundle.objects),
            "n_contacts": len(bundle.contacts),
            "n_aggregates": len(bundle.aggregates),
            "n_localization_profiles": len(bundle.localization_profiles),
            "warnings": list(bundle.warnings),
        }
        (tmp_dir / _RUN_MANIFEST_NAME).write_text(
            json.dumps(run_manifest, sort_keys=True, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        _replace_dir(tmp_dir, output_dir)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return {
        "objects": output_dir / "objects.csv",
        "contacts": output_dir / "contacts.csv",
        "aggregates": output_dir / "aggregates.csv",
        "localization_profiles": output_dir / "localization_profiles.csv",
        "metrics_manifest": output_dir / "metrics_manifest.json",
        "run_manifest": output_dir / _RUN_MANIFEST_NAME,
    }


def measure(
    label_artifact_dir: Path | str,
    output_dir: Path | str,
    config: PipelineConfig,
    image_artifact_dir: Path | str | None = None,
) -> RunArtifacts:
    """Measure one label artifact and write ``objects.csv`` et al.

    ``image_artifact_dir`` is optional: omitted, this is a geometry-only run
    (``metrics.engine.compute_measurements`` with ``image=None`` -- contact
    and aggregate metrics are still fully populated; every
    intensity/nuclei/localization field is recorded missing with reason
    ``metrics.engine.MISSING_IMAGE_REASON``). Supplied, the label artifact's
    recorded ``input_image_sha256`` (and shape, and spacing) must match the
    image artifact exactly, checked by
    ``io.zarr_store.read_label_volume``'s ``image=`` binding guard --
    raises :class:`~s2_adhesion.errors.ArtifactBindingError` on any mismatch,
    and does so before this function has written anything.

    Deterministic and side-effect-free with respect to segmentation: this
    function never invokes a segmentation backend and, per the module
    docstring, cannot even import one. Re-running it on the same artifacts
    with the same config produces byte-identical CSVs.
    """
    label_artifact_dir = Path(label_artifact_dir)
    output_dir = Path(output_dir)
    verify_hashes = config.artifacts.verify_content_hashes

    image = None
    if image_artifact_dir is not None:
        image = read_image_volume(Path(image_artifact_dir), verify_hashes=verify_hashes)

    # Binding validation (shape / spacing / input_image_sha256) happens
    # inside read_label_volume when image is not None -- raises
    # ArtifactBindingError immediately, before write_measurement_bundle (or
    # anything else that touches output_dir) is ever called.
    labels = read_label_volume(label_artifact_dir, verify_hashes=verify_hashes, image=image)

    bundle = compute_measurements(
        labels=labels,
        image=image,
        config=config.measurement,
        backend_id=config.analysis_backend,
    )

    paths = _write_bundle_atomically(bundle, output_dir)

    return RunArtifacts(
        output_dir=output_dir,
        objects_csv=paths["objects"],
        contacts_csv=paths["contacts"],
        aggregates_csv=paths["aggregates"],
        localization_profiles_csv=paths["localization_profiles"],
        metrics_manifest_json=paths["metrics_manifest"],
        run_manifest_json=paths["run_manifest"],
        label_artifact_dirs=(label_artifact_dir,),
        warnings=bundle.warnings,
    )
