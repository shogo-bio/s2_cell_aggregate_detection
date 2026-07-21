"""Render per-field review images from a pipeline run's artifacts.

For every field a run produced, this reads the image artifact and the label
artifact and writes a side-by-side PNG: each channel, the merge, and the
segmentation outline over the merge -- so a human can judge the segmentation
without opening a viewer.

Kept out of the measurement path on purpose. It needs matplotlib (a plotting
dependency, not part of the measurement stack) and it reads the SAME artifacts
``run``/``segment`` already wrote, so it never re-runs a model and never needs
torch/cellpose. Run it after a pipeline run, pointing at the run directory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..io.zarr_store import read_image_volume, read_label_volume


def _norm(a: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.5) -> np.ndarray:
    lo, hi = np.percentile(a, lo_pct), np.percentile(a, hi_pct)
    return np.clip((a - lo) / (hi - lo + 1e-9), 0.0, 1.0)


def _mip(channel_zyx: np.ndarray) -> np.ndarray:
    return _norm(channel_zyx.max(axis=0).astype(np.float64))


def render_field_review(
    image_artifact_dir: Path,
    label_artifact_dir: Path,
    out_png: Path,
    *,
    verify_hashes: bool = True,
) -> Path:
    """Write one comparison PNG for a single field. Returns the path written."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries

    image = read_image_volume(image_artifact_dir, verify_hashes=verify_hashes)
    labels = read_label_volume(
        label_artifact_dir, verify_hashes=verify_hashes, image=image
    )
    cells = labels.cells
    n_cells = int(cells.max())

    # Per-channel MIPs, in channel order. Colour the first two green/red for the
    # merge (the common two-population layout); extra channels are shown mono.
    channel_mips = [_mip(image.data[i]) for i in range(image.data.shape[0])]
    names = [c.channel_id for c in image.channels]

    green = channel_mips[0]
    red = channel_mips[1] if len(channel_mips) > 1 else np.zeros_like(green)
    merge = np.stack([red, green, np.zeros_like(green)], axis=-1)
    overlay = merge.copy()
    overlay[find_boundaries(cells.max(axis=0), mode="outer")] = [1, 1, 1]

    n_panels = len(channel_mips) + 2  # channels + merge + overlay
    fig, ax = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 5.5))
    for i, mip in enumerate(channel_mips):
        ax[i].imshow(mip, cmap="gray")
        ax[i].set_title(f"{names[i]} (MIP)")
    ax[-2].imshow(merge)
    ax[-2].set_title("merge")
    ax[-1].imshow(overlay)
    ax[-1].set_title(f"segmentation ({n_cells} cells)")
    for a in ax:
        a.axis("off")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=95, bbox_inches="tight")
    plt.close(fig)
    return out_png


def review_run(run_dir: Path | str, *, verify_hashes: bool = True) -> list[Path]:
    """Render a review PNG for every field of a completed run.

    Expects the ``run`` layout: ``<run_dir>/images/<field>.image.ome.zarr`` and
    ``<run_dir>/labels/<field>.labels.ome.zarr``. Writes to
    ``<run_dir>/review/<field>_compare.png``.
    """
    run_dir = Path(run_dir)
    images_dir = run_dir / "images"
    labels_dir = run_dir / "labels"
    review_dir = run_dir / "review"

    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(
            f"{run_dir} does not look like a pipeline run directory "
            "(expected images/ and labels/ subdirectories)."
        )

    written: list[Path] = []
    for image_artifact in sorted(images_dir.glob("*.image.ome.zarr")):
        field_id = image_artifact.name.removesuffix(".image.ome.zarr")
        label_artifact = labels_dir / f"{field_id}.labels.ome.zarr"
        if not label_artifact.exists():
            continue
        out_png = review_dir / f"{field_id}_compare.png"
        written.append(
            render_field_review(
                image_artifact, label_artifact, out_png, verify_hashes=verify_hashes
            )
        )
    return written
