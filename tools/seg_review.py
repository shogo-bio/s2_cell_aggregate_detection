#!/usr/bin/env python3
"""Run a segmentation backend on one nd2 field and save reviewable output.

Produces, in the output directory:
  * ``<tag>_labels.tif``        the 3D label volume (ZYX uint16)
  * ``<tag>_compare.png``       original channels + MIP outline overlay, side by
                                side, so a human can judge the segmentation
  * ``<tag>_meta.json``         backend, model, runtime, cell count, parameters

This is a REVIEW tool, deliberately outside the tested pipeline: it uses cellpose
or stardist directly (each in its own environment) so you can eyeball quality and
timing on real data before committing to a backend. It does not write pipeline
artifacts.

Usage (from the matching environment):
  cyto3:    dev/envs/cellpose-x64/.venv/python  tools/seg_review.py cellpose FILE.nd2 out/ [--field 0]
  stardist: dev/envs/stardist-x64/.venv/python  tools/seg_review.py stardist FILE.nd2 out/ [--field 0]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import nd2
import numpy as np


def _norm(a: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.5) -> np.ndarray:
    lo, hi = np.percentile(a, lo_pct), np.percentile(a, hi_pct)
    return np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)


def load_field(path: Path, field: int) -> tuple[np.ndarray, dict]:
    """Return (Z, C, Y, X) float32 for one field, plus voxel metadata."""
    with nd2.ND2File(path) as f:
        arr = np.asarray(f.asarray())
        sizes = dict(f.sizes)
        v = f.voxel_size()
        spacing = (float(v.z), float(v.y), float(v.x))
    # collapse any P axis to the requested field
    if "P" in sizes:
        p_axis = list(sizes).index("P")
        arr = np.take(arr, field, axis=p_axis)
    return arr.astype(np.float32), {"sizes": sizes, "spacing_um_zyx": spacing}


def run_cellpose(all_cells: np.ndarray, diameter: float) -> tuple[np.ndarray, dict]:
    """2.5D stitched cyto3: segment each plane in the good XY resolution, stitch in Z."""
    from cellpose import models, version

    model = models.Cellpose(gpu=False, model_type="cyto3")
    normed = np.stack([_norm(z) for z in all_cells])
    t = time.perf_counter()
    masks, _, _, _ = model.eval(
        normed, channels=[0, 0], diameter=diameter, do_3D=False, stitch_threshold=0.3
    )
    elapsed = time.perf_counter() - t
    return masks.astype(np.uint16), {
        "backend": "cellpose",
        "model": "cyto3",
        "cellpose_version": version,
        "mode": "2.5D_stitch(stitch_threshold=0.3)",
        "diameter_px": diameter,
        "elapsed_seconds": round(elapsed, 1),
    }


def run_stardist(all_cells: np.ndarray, spacing: tuple[float, float, float]) -> tuple[np.ndarray, dict]:
    """StarDist 3D with the (nuclei) demo model. Plumbing check only -- see docs/stardist_notes.md."""
    import stardist
    from csbdeep.utils import normalize
    from stardist.models import StarDist3D

    try:
        model = StarDist3D.from_pretrained("3D_demo")
    except OSError:
        # Windows symlink-privilege fallback: load the already-extracted weights.
        cache = Path.home() / ".keras" / "models" / "StarDist3D" / "3D_demo"
        model = StarDist3D(None, name="3D_demo_extracted", basedir=str(cache))

    img = normalize(all_cells, 1.0, 99.8)
    t = time.perf_counter()
    labels, _ = model.predict_instances(img, axes="ZYX", verbose=False)
    elapsed = time.perf_counter() - t
    return labels.astype(np.uint16), {
        "backend": "stardist",
        "model": "3D_demo(PRETRAINED nuclei demo -- not trained on these cells)",
        "stardist_version": getattr(stardist, "__version__", "?"),
        "model_anisotropy": list(getattr(model.config, "anisotropy", []) or []),
        "elapsed_seconds": round(elapsed, 1),
    }


def classify_by_channel(masks: np.ndarray, field: np.ndarray) -> dict:
    """Rough green/red split by background-relative mean, for the review caption only."""
    g, r = field[:, 0], field[:, 1]

    def score(ch: np.ndarray, mask: np.ndarray, outside: np.ndarray) -> float:
        bg = ch[outside]
        med = float(np.median(bg)) if bg.size else 0.0
        mad = (float(np.median(np.abs(bg - med))) * 1.4826) or 1.0
        return (float(ch[mask].mean()) - med) / mad

    outside = masks == 0
    green = red = 0
    for lab in np.unique(masks):
        if lab == 0:
            continue
        m = masks == lab
        green += int(score(g, m, outside) >= score(r, m, outside))
        red += int(score(g, m, outside) < score(r, m, outside))
    return {"green_cells": green, "red_cells": red}


def save_comparison(field: np.ndarray, masks: np.ndarray, out_png: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries

    g_mip = _norm(field[:, 0].max(0))
    r_mip = _norm(field[:, 1].max(0))
    merge = np.stack([r_mip, g_mip, np.zeros_like(g_mip)], -1)
    overlay = merge.copy()
    overlay[find_boundaries(masks.max(0), mode="outer")] = [1, 1, 1]

    fig, ax = plt.subplots(1, 4, figsize=(22, 5.5))
    ax[0].imshow(g_mip, cmap="gray"); ax[0].set_title("FITC / green (MIP)")
    ax[1].imshow(r_mip, cmap="gray"); ax[1].set_title("mCherry / red (MIP)")
    ax[2].imshow(merge); ax[2].set_title("merge")
    ax[3].imshow(overlay); ax[3].set_title(title)
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=95, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("backend", choices=["cellpose", "stardist"])
    ap.add_argument("nd2_path", type=Path)
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--field", type=int, default=0)
    ap.add_argument("--diameter", type=float, default=16.0, help="cellpose cell diameter (px)")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    field, meta = load_field(args.nd2_path, args.field)
    all_cells = field.max(axis=1)  # any-colour cell body proxy

    if args.backend == "cellpose":
        masks, run_meta = run_cellpose(all_cells, args.diameter)
    else:
        masks, run_meta = run_stardist(all_cells, meta["spacing_um_zyx"])

    n_cells = int(masks.max())
    run_meta.update(
        {
            "file": args.nd2_path.name,
            "field": args.field,
            "spacing_um_zyx": meta["spacing_um_zyx"],
            "field_shape_zcyx": list(field.shape),
            "n_cells": n_cells,
            **classify_by_channel(masks, field),
        }
    )

    tag = f"{args.nd2_path.stem}_field{args.field}_{args.backend}"
    import tifffile
    tifffile.imwrite(args.output_dir / f"{tag}_labels.tif", masks)
    save_comparison(
        field, masks, args.output_dir / f"{tag}_compare.png",
        title=f"{run_meta['backend']} ({n_cells} cells, {run_meta['elapsed_seconds']}s)",
    )
    (args.output_dir / f"{tag}_meta.json").write_text(
        json.dumps(run_meta, indent=2), encoding="utf-8"
    )
    print(f"OK {tag}: {n_cells} cells in {run_meta['elapsed_seconds']}s -> {args.output_dir}")


if __name__ == "__main__":
    main()
