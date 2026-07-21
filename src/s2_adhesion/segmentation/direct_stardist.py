"""``direct_stardist`` strategy: one StarDist 3D model, no nuclear seeding.

Mirrors ``direct_cellpose`` deliberately -- same injected-engine pattern, same
provenance shape, same "never leave a partial result" contract -- so the two can
be compared on identical fields by changing one line of config.

Why StarDist is worth having alongside Cellpose for this data: the labelled
proteins are mostly transmembrane, so the signal is a shell and the bright ring
IS the boundary StarDist predicts ray distances to. Where the ring is incomplete
-- which is expected, since the protein is not expressed over the whole surface
-- the star-convex shape prior closes the outline, whereas a flow field can leak
through the gap and merge two cells. S2 cells are near-spherical, so the
star-convexity assumption costs little.

What it will NOT do out of the box: ``StarDist3D`` registers exactly one
pretrained model, ``3D_demo``, trained on nuclei. Running it on membrane-labelled
cells is a plumbing test, not an analysis. Real use needs a trained or
fine-tuned model, which is why ``is_pretrained`` is carried into every warning
and into provenance.
"""

from __future__ import annotations

import hashlib
import platform
import time
from dataclasses import asdict, dataclass, is_dataclass

import numpy as np

from ..config import DirectStarDistConfig
from ..contracts import LabelVolume, SegmentationProvenance
from .postprocess import (
    fill_internal_holes,
    filter_by_physical_volume,
    split_disconnected_labels,
)
from .protocol import (
    SegmentationDiagnostics,
    SegmentationRequest,
    SegmentationResult,
    StarDistEngine,
)
from .stardist_preprocess import map_labels_to_original_grid, prepare_stardist_input

_BACKEND_ID = "direct_stardist"

# How far the data's own anisotropy may differ from the model's trained
# anisotropy before we warn. StarDist has no predict-time anisotropy knob, so a
# mismatch is silent and produces plausible-looking wrong instances.
_ANISOTROPY_TOLERANCE = 0.2


def _config_sha256(config: DirectStarDistConfig) -> str:
    def _to_plain(value: object) -> object:
        if is_dataclass(value) and not isinstance(value, type):
            return {f: _to_plain(v) for f, v in asdict(value).items()}  # type: ignore[arg-type]
        if isinstance(value, (list, tuple)):
            return [_to_plain(v) for v in value]
        if isinstance(value, dict):
            return {k: _to_plain(v) for k, v in value.items()}
        return repr(value)

    return hashlib.sha256(repr(_to_plain(config)).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class DirectStarDistBackend:
    """Segments cells directly with one injected StarDist engine."""

    config: DirectStarDistConfig
    engine: StarDistEngine
    backend_id: str = _BACKEND_ID

    def segment(self, request: SegmentationRequest) -> SegmentationResult:
        image = request.image
        warnings: list[str] = []

        if self.engine.is_pretrained:
            warnings.append(
                f"stardist_pretrained_model: using '{self.engine.model_name}', a "
                "published model. The only registered StarDist3D model is a "
                "nuclei demo -- treat instances from it as a plumbing check, not "
                "an analysis, until a model trained on this data is available."
            )

        warnings.extend(self._anisotropy_warnings(image))

        t0 = time.perf_counter()
        prepared = prepare_stardist_input(image, self.config)
        prep_seconds = time.perf_counter() - t0

        t1 = time.perf_counter()
        raw = self.engine.predict(prepared)
        predict_seconds = time.perf_counter() - t1

        cells = map_labels_to_original_grid(raw.labels, prepared)

        # Shared post-processing, in physical units so the same config behaves
        # identically at a different Z step.
        if self.config.split_disconnected_labels:
            cells = split_disconnected_labels(cells)
        if self.config.fill_internal_holes:
            cells = fill_internal_holes(cells)
        before = int((np.unique(cells) > 0).sum())
        cells = filter_by_physical_volume(
            cells, image.geometry.spacing_um_zyx, self.config.min_cell_volume_um3
        )
        removed = before - int((np.unique(cells) > 0).sum())
        if removed:
            warnings.append(
                f"stardist_small_objects_removed: {removed} instance(s) below "
                f"{self.config.min_cell_volume_um3} um^3"
            )

        n_final = int(len(np.unique(cells)) - (1 if (cells == 0).any() else 0))
        if n_final == 0:
            warnings.append(
                "stardist_no_instances: the model found nothing. Check that the "
                "configured channel is the one carrying cell outlines, and that "
                "the model was trained on comparable data."
            )

        provenance = SegmentationProvenance(
            run_id=request.run_id,
            backend_id=self.backend_id,
            strategy=_BACKEND_ID,
            config_sha256=_config_sha256(self.config),
            input_image_sha256=image.identity.image_content_sha256,
            device="cpu",
            host_platform=platform.platform(),
            package_name="stardist",
            package_version=self.engine.package_version,
            model_name=self.engine.model_name,
            model_sha256=None,
        )

        labels = LabelVolume(
            cells=cells.astype(np.uint32),
            geometry=image.geometry,
            identity=image.identity,
            provenance=provenance,
        )

        diagnostics = SegmentationDiagnostics(
            warnings=tuple(warnings),
            timing_seconds={"preprocess": prep_seconds, "predict": predict_seconds},
            extra={
                "n_instances_raw": raw.n_instances,
                "n_instances_final": n_final,
                "was_resampled": prepared.was_resampled,
                "model_grid_spacing_um": repr(prepared.spacing_um_zyx),
                "is_pretrained": self.engine.is_pretrained,
            },
        )
        return SegmentationResult(labels=labels, diagnostics=diagnostics)

    def _anisotropy_warnings(self, image) -> list[str]:
        """Warn when the model's trained sampling disagrees with the data's.

        StarDist has no predict-time anisotropy parameter, so this mismatch
        cannot be corrected at inference -- it can only be resampled around
        (``resample_isotropic``) or trained around. Saying nothing would let a
        silently wrong sampling reach the results.
        """
        trained = self.engine.model_anisotropy
        if trained is None:
            return []

        if self.config.model.resample_isotropic:
            # We hand the model an isotropic volume, so a model trained
            # anisotropically is the mismatch.
            spread = max(trained) / min(trained) if min(trained) > 0 else float("inf")
            if spread > 1.0 + _ANISOTROPY_TOLERANCE:
                return [
                    "stardist_anisotropy_mismatch: resample_isotropic=True feeds "
                    f"the model isotropic data, but it was trained at anisotropy "
                    f"{trained}. Either turn resampling off or use a model "
                    "trained on isotropic data."
                ]
            return []

        data = image.geometry.spacing_um_zyx
        data_ratio = max(data) / min(data)
        trained_ratio = max(trained) / min(trained) if min(trained) > 0 else float("inf")
        if abs(data_ratio - trained_ratio) > _ANISOTROPY_TOLERANCE * max(
            1.0, trained_ratio
        ):
            return [
                "stardist_anisotropy_mismatch: data sampled at anisotropy ratio "
                f"{data_ratio:.2f} but the model was trained at {trained_ratio:.2f}, "
                "and StarDist has no predict-time anisotropy parameter."
            ]
        return []
