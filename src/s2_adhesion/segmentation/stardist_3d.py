"""The only module allowed to import ``stardist`` or ``tensorflow``.

Everything else in the package talks to the ``StarDistEngine`` protocol, so the
measurement layer -- and every unit test -- runs with no ML stack installed.

Verified against stardist 0.9.2 / tensorflow 2.21.0 on this machine.

Two facts about the real library that shape this adapter:

* ``StarDist3D.predict_instances`` has NO ``anisotropy`` parameter. Anisotropy
  is a training-time property of the model (``Config3D(anisotropy=...)``), so
  the caller must feed a volume sampled the way the model expects. That is what
  ``stardist_preprocess.prepare_stardist_input`` does.
* There is exactly ONE registered pretrained StarDist3D model, ``3D_demo``, and
  it is a nuclei demo. It is useful for smoke-testing the plumbing and almost
  certainly not for production S2 cells -- expect to train or fine-tune. The
  engine records ``is_pretrained`` so that fact travels into provenance rather
  than being forgotten between the run and the figure.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from ..config import StarDistModelConfig
from ..errors import MLDependencyError, SegmentationError
from .protocol import PreparedStarDistInput, StarDistRawResult

# TensorFlow prints a wall of oneDNN/absl notices on import. Quieten it before
# the import so a run's real warnings stay visible.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


class StarDist3DEngine:
    """A loaded StarDist 3D model, ready to predict."""

    def __init__(
        self,
        model: Any,
        *,
        model_name: str,
        package_version: str,
        is_pretrained: bool,
        config: StarDistModelConfig,
    ) -> None:
        self._model = model
        self._config = config
        self.model_name = model_name
        self.package_version = package_version
        self.is_pretrained = is_pretrained
        self.model_anisotropy = _model_anisotropy(model)

    def predict(self, prepared: PreparedStarDistInput) -> StarDistRawResult:
        kwargs: dict[str, Any] = {"axes": "ZYX", "verbose": False}
        if self._config.prob_threshold is not None:
            kwargs["prob_thresh"] = self._config.prob_threshold
        if self._config.nms_threshold is not None:
            kwargs["nms_thresh"] = self._config.nms_threshold
        if self._config.n_tiles_zyx is not None:
            kwargs["n_tiles"] = tuple(self._config.n_tiles_zyx)

        try:
            labels, details = self._model.predict_instances(prepared.data, **kwargs)
        except Exception as exc:  # noqa: BLE001 - re-raise as our own error type
            raise SegmentationError(
                f"StarDist prediction failed on a {prepared.data.shape} volume: {exc}"
            ) from exc

        labels = np.asarray(labels)
        n = int(labels.max()) if labels.size else 0
        return StarDistRawResult(
            labels=labels,
            n_instances=n,
            extra={
                "stardist_model": self.model_name,
                "stardist_is_pretrained": self.is_pretrained,
                "stardist_points_found": int(
                    len(details.get("points", [])) if isinstance(details, dict) else 0
                ),
            },
        )


def _model_anisotropy(model: Any) -> tuple[float, float, float] | None:
    """The anisotropy the model was TRAINED with, if it recorded one.

    Exposed so a caller can compare it against the data's own sampling instead
    of assuming they match -- feeding a model a volume at the wrong sampling is
    silent, not an error, and produces plausible-looking wrong instances.
    """
    config = getattr(model, "config", None)
    value = getattr(config, "anisotropy", None)
    if value is None:
        return None
    try:
        z, y, x = (float(v) for v in value)
    except (TypeError, ValueError):
        return None
    return (z, y, x)


def build_engine(config: StarDistModelConfig) -> StarDist3DEngine:
    """Load the configured model. The only entry point ``factory`` should call."""
    try:
        import stardist
        from stardist.models import StarDist3D
    except ImportError as exc:
        raise MLDependencyError(
            "StarDist segmentation needs the 'stardist' package (and "
            "tensorflow), which is not importable here.\n"
            "  install with:  uv pip install 'stardist' 'tensorflow'\n"
            "  note: stardist is TensorFlow-based while cellpose is PyTorch-based; "
            "keeping them in separate environments avoids version conflicts."
        ) from exc

    package_version = getattr(stardist, "__version__", "unknown")

    if config.custom_model_dir is not None:
        if not config.custom_model_name:
            raise SegmentationError(
                "custom_model_dir was given without custom_model_name; StarDist "
                "loads a model by name from a base directory."
            )
        basedir = config.custom_model_dir
        if not basedir.exists():
            raise SegmentationError(f"custom_model_dir does not exist: {basedir}")
        try:
            model = StarDist3D(None, name=config.custom_model_name, basedir=str(basedir))
        except Exception as exc:  # noqa: BLE001
            raise SegmentationError(
                f"could not load StarDist model {config.custom_model_name!r} from "
                f"{basedir}: {exc}"
            ) from exc
        return StarDist3DEngine(
            model,
            model_name=config.custom_model_name,
            package_version=package_version,
            is_pretrained=False,
            config=config,
        )

    if config.pretrained_name is None:
        raise SegmentationError(
            "no StarDist model configured: give either pretrained_name or "
            "custom_model_dir + custom_model_name."
        )

    try:
        model = StarDist3D.from_pretrained(config.pretrained_name)
    except OSError as exc:
        # Windows without Developer Mode: csbdeep downloads and extracts the
        # model fine, then fails creating a symlink to it (WinError 1314,
        # "a required privilege is not held by the client"). The weights are
        # already on disk at that point, so load them directly rather than
        # asking the user to change a system setting for a smoke test.
        model = _load_extracted_pretrained(StarDist3D, config.pretrained_name, exc)
    except Exception as exc:  # noqa: BLE001
        raise SegmentationError(
            f"could not load pretrained StarDist model "
            f"{config.pretrained_name!r}: {exc}\n"
            "  StarDist3D currently registers only '3D_demo', a nuclei demo "
            "model. For membrane-labelled cells expect to train or fine-tune "
            "your own and point custom_model_dir at it."
        ) from exc

    return StarDist3DEngine(
        model,
        model_name=config.pretrained_name,
        package_version=package_version,
        is_pretrained=True,
        config=config,
    )


def _load_extracted_pretrained(star_dist_cls: Any, name: str, original: OSError) -> Any:
    """Load a pretrained model straight from csbdeep's extraction directory.

    csbdeep caches under ``~/.keras/models/StarDist3D/<name>/`` and unpacks the
    weights into ``<name>_extracted/`` before symlinking. On Windows the symlink
    needs Developer Mode or Administrator, so that last step raises WinError
    1314 even though every file we need is already there.
    """
    from pathlib import Path

    cache = Path.home() / ".keras" / "models" / "StarDist3D" / name
    extracted = cache / f"{name}_extracted"
    if not (extracted / "config.json").exists():
        raise SegmentationError(
            f"could not load pretrained StarDist model {name!r}: {original}\n"
            f"  looked for an already-extracted copy at {extracted} and did not "
            "find one.\n"
            "  On Windows this is usually a symlink privilege error (WinError "
            "1314). Either enable Developer Mode, or download the model once and "
            "point custom_model_dir/custom_model_name at it."
        ) from original

    try:
        return star_dist_cls(None, name=extracted.name, basedir=str(cache))
    except Exception as exc:  # noqa: BLE001
        raise SegmentationError(
            f"found an extracted copy of {name!r} at {extracted} but could not "
            f"load it: {exc}"
        ) from exc
