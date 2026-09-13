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


# Outline colours for the population panel. Anything not listed is drawn grey.
POPULATION_COLOURS: dict[str, tuple[float, float, float]] = {
    "Cirl-GFP": (0.2, 1.0, 0.2),
    "Cirl-mCherry": (1.0, 0.25, 0.25),
    "ambiguous": (1.0, 0.75, 0.1),
    "double_signal": (0.85, 0.3, 1.0),
    "unassigned": (0.6, 0.6, 0.6),
}


def _population_by_cell(objects_csv: Path, field_id: str) -> dict[int, str]:
    """``object_id -> population`` for one field, from a run's objects.csv.

    Only ``cell_3d`` rows are used. Returns an empty mapping when the file
    lacks a population column (no population configured for the run).
    """
    import csv

    out: dict[int, str] = {}
    with objects_csv.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "population" not in reader.fieldnames:
            return out
        for row in reader:
            if row.get("field_id") != field_id:
                continue
            if row.get("object_kind", "cell_3d") != "cell_3d":
                continue
            try:
                out[int(row["object_id"])] = row["population"] or "unassigned"
            except (KeyError, ValueError):
                continue
    return out


def render_field_review(
    image_artifact_dir: Path,
    label_artifact_dir: Path,
    out_png: Path,
    *,
    verify_hashes: bool = True,
    objects_csv: Path | None = None,
) -> Path:
    """Write one comparison PNG for a single field. Returns the path written.

    When ``objects_csv`` (a run's ``measurements/objects.csv``) is given and
    carries a ``population`` column, a sixth panel draws each outline in its
    population's colour so the assignment can be judged against the merge.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.color import label2rgb
    from skimage.segmentation import find_boundaries

    image = read_image_volume(image_artifact_dir, verify_hashes=verify_hashes)
    labels = read_label_volume(
        label_artifact_dir, verify_hashes=verify_hashes, image=image
    )
    cells = labels.cells
    n_cells = int(len(np.unique(cells)) - (1 if (cells == 0).any() else 0))

    # Per-channel MIPs, in channel order. Colour the first two green/red for the
    # merge (the common two-population layout); extra channels are shown mono.
    channel_mips = [_mip(image.data[i]) for i in range(image.data.shape[0])]
    names = [c.channel_id for c in image.channels]

    green = channel_mips[0]
    red = channel_mips[1] if len(channel_mips) > 1 else np.zeros_like(green)
    merge = np.stack([red, green, np.zeros_like(green)], axis=-1)

    # Max-label projection: at each pixel, the highest cell id in the column.
    # Enough to show instance identity for a visual quality check.
    label_mip = cells.max(axis=0)
    overlay = merge.copy()
    overlay[find_boundaries(label_mip, mode="outer")] = [1, 1, 1]
    # Filled instances in distinct colours -- makes split/merge errors obvious in
    # a way outlines alone do not.
    filled = label2rgb(label_mip, bg_label=0, bg_color=(0, 0, 0))

    population_of: dict[int, str] = {}
    if objects_csv is not None and objects_csv.exists():
        population_of = _population_by_cell(objects_csv, image.identity.field_id)

    # channels + merge + outlines + filled mask (+ population outlines)
    n_panels = len(channel_mips) + 3 + (1 if population_of else 0)
    fig, ax = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 5.5))
    for i, mip in enumerate(channel_mips):
        ax[i].imshow(mip, cmap="gray")
        ax[i].set_title(f"{names[i]} (MIP)")
    k = len(channel_mips)
    ax[k].imshow(merge)
    ax[k].set_title("merge")
    ax[k + 1].imshow(overlay)
    ax[k + 1].set_title(f"outlines ({n_cells} cells)")
    ax[k + 2].imshow(filled)
    ax[k + 2].set_title("predicted mask (filled)")
    if population_of:
        pop_overlay = merge * 0.6
        counts: dict[str, int] = {}
        for cell_id, pop in population_of.items():
            colour = POPULATION_COLOURS.get(pop, (0.6, 0.6, 0.6))
            pop_overlay[find_boundaries(label_mip == cell_id, mode="outer")] = colour
            counts[pop] = counts.get(pop, 0) + 1
        ax[k + 3].imshow(np.clip(pop_overlay, 0, 1))
        legend = ", ".join(f"{p} {n}" for p, n in sorted(counts.items()))
        ax[k + 3].set_title(f"population: {legend}", fontsize=9)
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

    objects_csv = run_dir / "measurements" / "objects.csv"
    written: list[Path] = []
    for image_artifact in sorted(images_dir.glob("*.image.ome.zarr")):
        field_id = image_artifact.name.removesuffix(".image.ome.zarr")
        label_artifact = labels_dir / f"{field_id}.labels.ome.zarr"
        if not label_artifact.exists():
            continue
        out_png = review_dir / f"{field_id}_compare.png"
        written.append(
            render_field_review(
                image_artifact, label_artifact, out_png, verify_hashes=verify_hashes,
                objects_csv=objects_csv if objects_csv.exists() else None,
            )
        )
    return written
