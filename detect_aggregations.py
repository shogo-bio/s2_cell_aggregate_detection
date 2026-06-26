#!/usr/bin/env python3
"""
S2 cell aggregation detection from nd2 fluorescence images.

Segmentation strategy: cell occupancy estimation (not boundary detection).
Pipeline per field: MIP → normalize → Gaussian blur → Otsu threshold
→ binary_fill_holes → morphology close → connected components → area filter.

Usage:
    python detect_aggregations.py input.nd2 output_dir/ [options]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Generator, NamedTuple

import cv2
import nd2
import numpy as np
from scipy.ndimage import binary_fill_holes


# ─── Configuration ────────────────────────────────────────────────────────────

_COLOR_RGB: dict[str, tuple[int, int, int]] = {
    "red":     (1, 0, 0),
    "green":   (0, 1, 0),
    "blue":    (0, 0, 1),
    "cyan":    (0, 1, 1),
    "magenta": (1, 0, 1),
    "yellow":  (1, 1, 0),
    "white":   (1, 1, 1),
}

CHANNEL_COLORS: dict[int, str] = {
    0: "green",
    1: "red",
    2: "blue",
}


class Config(NamedTuple):
    s2_diameter_um: float = 10.0      # S2 cell diameter in µm (typical 8–12 µm)
    aggregation_min_cells: int = 3    # aggregation threshold = this many S2 cells
    morph_close_radius_um: float = 1.65  # disk radius for morphology close in µm (converted to px at runtime)
    binary_threshold: int = 50        # fixed threshold for binarization after median (0–255)
    min_active_channels: int = 2      # reject regions with fewer active channels
    channel_colors: dict[int, str] = CHANNEL_COLORS
    save_debug: bool = False          # save intermediate mask/channel images when True


def compute_area_threshold_px(
    pixel_size_um: float,
    s2_diameter_um: float,
    min_cells: int,
) -> float:
    """Convert S2-cell-count threshold to pixel area."""
    cell_area_um2 = np.pi * (s2_diameter_um / 2.0) ** 2
    threshold_um2 = cell_area_um2 * min_cells
    return threshold_um2 / (pixel_size_um ** 2)


# ─── nd2 loading ──────────────────────────────────────────────────────────────


def load_nd2(path: Path) -> tuple[np.ndarray, dict[str, int], float]:
    """
    Load an nd2 file.

    Returns
    -------
    data : np.ndarray
    sizes : dict[str, int]  e.g. {'P': 3, 'Z': 10, 'C': 3, 'Y': 512, 'X': 512}
    pixel_size_um : float
    """
    with nd2.ND2File(path) as f:
        data = np.asarray(f.asarray())
        sizes = dict(f.sizes)
        voxel = f.voxel_size()
        pixel_size_um = float(voxel.x)
    return data, sizes, pixel_size_um


def iter_fields(
    data: np.ndarray, sizes: dict[str, int]
) -> Generator[tuple[int, np.ndarray, dict[str, int]], None, None]:
    """Yield (field_id, field_array, field_sizes) for each position."""
    dim_keys = list(sizes.keys())
    if "P" in dim_keys:
        p_axis = dim_keys.index("P")
        n_fields = sizes["P"]
        field_sizes = {k: v for k, v in sizes.items() if k != "P"}
        for fid in range(n_fields):
            yield fid, np.take(data, fid, axis=p_axis), field_sizes
    else:
        yield 0, data, sizes


# ─── Image processing helpers ─────────────────────────────────────────────────


def to_uint8(
    img: np.ndarray,
    lo: float | None = None,
    hi: float | None = None,
) -> np.ndarray:
    """Min-max normalize to uint8, using provided lo/hi if given."""
    img = img.astype(np.float32)
    if lo is None:
        lo = float(img.min())
    if hi is None:
        hi = float(img.max())
    if hi > lo:
        img = (img - lo) / (hi - lo) * 255.0
        img = np.clip(img, 0.0, 255.0)
    else:
        img = np.zeros_like(img)
    return img.astype(np.uint8)


def max_projection(stack: np.ndarray) -> np.ndarray:
    """Max-intensity projection along axis 0.  Input: (Z, Y, X)."""
    return stack.max(axis=0)


def compute_global_channel_minmax(
    data: np.ndarray, sizes: dict[str, int]
) -> list[tuple[float, float]]:
    """Global (lo, hi) per channel across all fields and Z-planes (raw counts)."""
    n_channels = sizes.get("C", 1)
    dim_keys = list(sizes.keys())
    result: list[tuple[float, float]] = []
    for ch in range(n_channels):
        arr = data
        if "C" in dim_keys:
            c_axis = dim_keys.index("C")
            arr = np.take(arr, ch, axis=c_axis)
        result.append((float(arr.min()), float(arr.max())))
    return result


def compute_median_kernel_size(pixel_size_um: float, s2_diameter_um: float) -> int:
    """Odd kernel size for median filter: ~1/4 of S2-cell diameter in pixels, minimum 3."""
    diameter_px = s2_diameter_um / pixel_size_um
    k = max(3, int(diameter_px / 4))
    return k if k % 2 == 1 else k + 1


def extract_channel_projections(
    field_data: np.ndarray,
    sizes: dict[str, int],
    global_minmax: list[tuple[float, float]] | None = None,
) -> list[np.ndarray]:
    """Return a list of uint8 MIP images per channel, normalized with global stats."""
    dim_keys = list(sizes.keys())
    n_channels = sizes.get("C", 1)
    projections: list[np.ndarray] = []

    for ch in range(n_channels):
        arr = field_data

        if "C" in dim_keys:
            c_axis = dim_keys.index("C")
            arr = np.take(arr, ch, axis=c_axis)
            remaining = [k for k in dim_keys if k != "C"]
        else:
            remaining = list(dim_keys)

        if "Z" in remaining:
            z_axis = remaining.index("Z")
            arr = arr.max(axis=z_axis)

        lo, hi = global_minmax[ch] if global_minmax else (None, None)
        projections.append(to_uint8(arr, lo=lo, hi=hi))

    return projections


def extract_center_slices(
    field_data: np.ndarray,
    sizes: dict[str, int],
    pixel_size_um: float,
    s2_diameter_um: float,
    global_minmax: list[tuple[float, float]] | None = None,
) -> list[np.ndarray]:
    """Center Z-slice per channel with adaptive median filter for channel activity detection."""
    dim_keys = list(sizes.keys())
    n_channels = sizes.get("C", 1)
    n_z = sizes.get("Z", 1)
    center_z = n_z // 2
    kernel_size = compute_median_kernel_size(pixel_size_um, s2_diameter_um)
    slices: list[np.ndarray] = []

    for ch in range(n_channels):
        arr = field_data

        if "C" in dim_keys:
            c_axis = dim_keys.index("C")
            arr = np.take(arr, ch, axis=c_axis)
            remaining = [k for k in dim_keys if k != "C"]
        else:
            remaining = list(dim_keys)

        if "Z" in remaining:
            z_axis = remaining.index("Z")
            arr = np.take(arr, center_z, axis=z_axis)

        lo, hi = global_minmax[ch] if global_minmax else (None, None)
        normalized = to_uint8(arr, lo=lo, hi=hi)
        filtered = cv2.medianBlur(normalized, kernel_size)
        slices.append(filtered)

    return slices


def merge_channels(channel_projections: list[np.ndarray]) -> np.ndarray:
    """Per-pixel maximum across channels → grayscale uint8 for segmentation."""
    return np.maximum.reduce(channel_projections)


def apply_pseudo_color(gray: np.ndarray, color: str) -> np.ndarray:
    """Map uint8 grayscale to pseudo-colored RGB (H, W, 3)."""
    weights = np.array(_COLOR_RGB[color], dtype=np.uint8)
    return gray[..., np.newaxis] * weights


def pseudo_color_merge(
    channel_projections: list[np.ndarray],
    channel_colors: dict[int, str],
) -> np.ndarray:
    """Assign pseudo-colors and merge by per-pixel maximum → uint8 RGB."""
    h, w = channel_projections[0].shape
    merged_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for i, proj in enumerate(channel_projections):
        color = channel_colors.get(i, "white")
        merged_rgb = np.maximum(merged_rgb, apply_pseudo_color(proj, color))
    return merged_rgb


# ─── Occupancy-based segmentation ────────────────────────────────────────────


def segment_occupancy(
    img: np.ndarray, cfg: Config, pixel_size_um: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cell occupancy estimation pipeline.  Expects a pre-filtered (median) image.

    Steps
    -----
    1. Fixed threshold — binarize with cfg.binary_threshold
    2. binary_fill_holes — fill dark nuclei so donut → disk
    3. morphology close  — bridge sub-pixel gaps without merging distant cells

    Returns
    -------
    mask_binary : uint8 binary after fixed threshold
    mask_filled : uint8 binary after hole filling
    mask_final  : uint8 binary after morphology close (used for detection)
    """
    _, mask_binary = cv2.threshold(img, cfg.binary_threshold, 255, cv2.THRESH_BINARY)
    # 第2引数は 0 にし、フラグに cv2.THRESH_OTSU を追加します
    # threshold_value, mask_binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Fill enclosed dark regions (e.g. nucleus inside membrane ring)
    mask_filled = binary_fill_holes(mask_binary > 0).astype(np.uint8) * 255

    r = max(1, round(cfg.morph_close_radius_um / pixel_size_um))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    mask_final = cv2.morphologyEx(mask_filled, cv2.MORPH_CLOSE, kernel)

    return mask_binary, mask_filled, mask_final


# ─── Connected components & area filtering ────────────────────────────────────


class RegionRecord(NamedTuple):
    field_id: int
    aggregation_id: int
    area_px: int
    area_um2: float
    active_channels: int


def count_active_channels(
    component_mask: np.ndarray,
    binary_center_slices: list[np.ndarray],
) -> int:
    """Count channels that have any positive pixel within the component.

    Each center slice must already be binarized (values 0 or 255).
    A channel is active when at least one white pixel overlaps the component.
    """
    return sum(
        1 for slc in binary_center_slices
        if (slc.astype(bool) & component_mask).any()
    )


def detect_aggregations(
    mask: np.ndarray,
    binary_center_slices: list[np.ndarray],
    pixel_size_um: float,
    field_id: int,
    area_threshold_px: float,
    min_active_channels: int,
) -> tuple[list[RegionRecord], np.ndarray, np.ndarray]:
    """
    Label connected components; filter by area then by active-channel check.

    A region is accepted only when:
      area_px >= area_threshold_px
      AND active_channels >= min_active_channels
      where active = any white pixel inside component in the binarized center slice

    Returns
    -------
    records       : RegionRecord for each accepted region
    valid_mask    : uint8 binary mask of accepted regions
    rejected_mask : uint8 binary mask of area-passing but insufficient-channel regions
    """
    n_labels, labeled, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    valid_mask = np.zeros_like(mask)
    rejected_mask = np.zeros_like(mask)
    records: list[RegionRecord] = []
    agg_id = 0
    pixel_area_um2 = pixel_size_um ** 2

    for label_id in range(1, n_labels):  # 0 is background
        area_px = int(stats[label_id, cv2.CC_STAT_AREA])
        if area_px < area_threshold_px:
            continue

        component_mask = labeled == label_id
        active_ch = count_active_channels(component_mask, binary_center_slices)

        if active_ch >= min_active_channels:
            valid_mask[component_mask] = 255
            records.append(RegionRecord(
                field_id=field_id,
                aggregation_id=agg_id,
                area_px=area_px,
                area_um2=round(area_px * pixel_area_um2, 4),
                active_channels=active_ch,
            ))
            agg_id += 1
        else:
            rejected_mask[component_mask] = 255

    return records, valid_mask, rejected_mask


# ─── Visualization ────────────────────────────────────────────────────────────


def _contours_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return list(contours)


def draw_contour_overlay_gray(
    gray: np.ndarray, valid_mask: np.ndarray, rejected_mask: np.ndarray
) -> np.ndarray:
    """Red = valid aggregation, blue = single-channel rejected. Returns BGR."""
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(bgr, _contours_from_mask(rejected_mask), -1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.drawContours(bgr, _contours_from_mask(valid_mask), -1, (0, 0, 255), 2, cv2.LINE_AA)
    return bgr


def draw_contour_overlay_color(
    merged_rgb: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
) -> np.ndarray:
    """Draw contours of mask on pseudo-color RGB base with the given BGR color. Returns BGR."""
    bgr = cv2.cvtColor(merged_rgb, cv2.COLOR_RGB2BGR)
    cv2.drawContours(bgr, _contours_from_mask(mask), -1, color, 2, cv2.LINE_AA)
    return bgr


# ─── Debug image saving ───────────────────────────────────────────────────────


def _imwrite(path: Path, img: np.ndarray) -> None:
    """Write an image and print an error if it fails."""
    try:
        ok = cv2.imwrite(str(path), img)
    except Exception as e:
        print(f"[ERROR] Failed to save debug image {path}: {e}")
        return
    if not ok:
        print(f"[ERROR] cv2.imwrite returned False for {path} — image may be empty or path invalid")


def save_debug_images(
    debug_dir: Path,
    prefix: str,
    channel_projections: list[np.ndarray],
    center_slices: list[np.ndarray],
    binary_center_slices: list[np.ndarray],
    merged_rgb: np.ndarray,
    gray_merged_median: np.ndarray,
    mask_binary: np.ndarray,
    mask_filled: np.ndarray,
    mask_final: np.ndarray,
    overlay_gray: np.ndarray,
    overlay_color_valid: np.ndarray,
    overlay_color_rejected: np.ndarray,
    save_debug: bool = False,
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)

    # Always saved
    _imwrite(debug_dir / f"{prefix}_color_merge.png", cv2.cvtColor(merged_rgb, cv2.COLOR_RGB2BGR))
    _imwrite(debug_dir / f"{prefix}_overlay_valid.png", overlay_color_valid)
    _imwrite(debug_dir / f"{prefix}_overlay_rejected.png", overlay_color_rejected)

    if save_debug:
        for i, proj in enumerate(channel_projections):
            _imwrite(debug_dir / f"{prefix}_ch{i}_mip.png", proj)
        for i, slc in enumerate(center_slices):
            _imwrite(debug_dir / f"{prefix}_ch{i}_center_median.png", slc)
        for i, slc in enumerate(binary_center_slices):
            _imwrite(debug_dir / f"{prefix}_ch{i}_center_binary.png", slc)
        _imwrite(debug_dir / f"{prefix}_mip_merged_median.png", gray_merged_median)
        _imwrite(debug_dir / f"{prefix}_binary_thresh.png", mask_binary)
        _imwrite(debug_dir / f"{prefix}_binary_filled.png", mask_filled)
        _imwrite(debug_dir / f"{prefix}_binary_closed.png", mask_final)
        _imwrite(debug_dir / f"{prefix}_overlay_gray.png", overlay_gray)


# ─── Per-field pipeline ───────────────────────────────────────────────────────


def process_field(
    field_data: np.ndarray,
    field_sizes: dict[str, int],
    pixel_size_um: float,
    field_id: int,
    cfg: Config,
    debug_dir: Path,
    global_minmax: list[tuple[float, float]] | None = None,
) -> list[RegionRecord]:
    """Full occupancy-based pipeline for one field of view."""
    median_k = compute_median_kernel_size(pixel_size_um, cfg.s2_diameter_um)
    channel_projections = extract_channel_projections(field_data, field_sizes, global_minmax)
    center_slices = extract_center_slices(
        field_data, field_sizes, pixel_size_um, cfg.s2_diameter_um, global_minmax
    )
    gray_merged = merge_channels(channel_projections)
    gray_merged_median = cv2.medianBlur(gray_merged, median_k)
    rgb_merged = pseudo_color_merge(channel_projections, cfg.channel_colors)

    if field_id == 0:
        pct5 = int(np.percentile(gray_merged_median, 5))
        pct95 = int(np.percentile(gray_merged_median, 95))
        print(
            f"  [field 0 merged_median] dtype={gray_merged_median.dtype}"
            f"  range=[{gray_merged_median.min()}, {gray_merged_median.max()}]"
            f"  5th–95th pct=[{pct5}, {pct95}]"
        )

    mask_binary, mask_filled, mask_final = segment_occupancy(gray_merged_median, cfg, pixel_size_um)

    binary_center_slices = [
        cv2.threshold(slc, cfg.binary_threshold, 255, cv2.THRESH_BINARY)[1]
        for slc in center_slices
    ]

    area_threshold_px = compute_area_threshold_px(
        pixel_size_um, cfg.s2_diameter_um, cfg.aggregation_min_cells
    )

    records, valid_mask, rejected_mask = detect_aggregations(
        mask_final, binary_center_slices, pixel_size_um, field_id,
        area_threshold_px, cfg.min_active_channels,
    )

    overlay_gray = draw_contour_overlay_gray(gray_merged_median, valid_mask, rejected_mask)
    # (0, 0, 255) = red in BGR;  (255, 0, 0) = blue in BGR
    overlay_color_valid = draw_contour_overlay_color(rgb_merged, valid_mask, (0, 0, 255))
    overlay_color_rejected = draw_contour_overlay_color(rgb_merged, rejected_mask, (255, 0, 0))

    prefix = f"field{field_id:02d}"
    save_debug_images(
        debug_dir, prefix, channel_projections, center_slices, binary_center_slices,
        rgb_merged, gray_merged_median, mask_binary, mask_filled, mask_final,
        overlay_gray, overlay_color_valid, overlay_color_rejected,
        save_debug=cfg.save_debug,
    )

    n_rejected = len(_contours_from_mask(rejected_mask))
    print(
        f"  Field {field_id:3d}: {len(records)} accepted, {n_rejected} rejected (single-channel)"
        f"  [area threshold {area_threshold_px:.0f} px"
        f" = {cfg.aggregation_min_cells}× S2 @ {pixel_size_um:.4f} µm/px]"
    )
    return records


# ─── CSV output ───────────────────────────────────────────────────────────────


def save_csv(records: list[RegionRecord], output_path: Path) -> None:
    if not records:
        print("No aggregations detected — CSV not written.")
        return
    fieldnames = ["field_id", "aggregation_id", "area_px", "area_um2", "active_channels"]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([r._asdict() for r in records])
    print(f"Results saved → {output_path}")


# ─── Top-level pipeline ───────────────────────────────────────────────────────


def process_nd2(nd2_path: Path, output_dir: Path, cfg: Config) -> None:
    """Load an nd2 file and run the aggregation detection pipeline on all fields."""
    print(f"Loading: {nd2_path}")
    data, sizes, pixel_size_um = load_nd2(nd2_path)
    print(f"  Dimensions : {sizes}")
    print(f"  Pixel size : {pixel_size_um:.4f} µm/pixel")

    s2_area_um2 = np.pi * (cfg.s2_diameter_um / 2.0) ** 2
    print(
        f"  Area threshold: {cfg.aggregation_min_cells} S2 cells"
        f" × {s2_area_um2:.1f} µm² = {s2_area_um2 * cfg.aggregation_min_cells:.1f} µm²"
    )

    debug_dir = output_dir / "debug" / nd2_path.stem
    all_records: list[RegionRecord] = []

    global_minmax = compute_global_channel_minmax(data, sizes)
    print(f"  Global per-channel range: {[(f'{lo:.0f}', f'{hi:.0f}') for lo, hi in global_minmax]}")

    median_k = compute_median_kernel_size(pixel_size_um, cfg.s2_diameter_um)
    morph_r_px = max(1, round(cfg.morph_close_radius_um / pixel_size_um))
    print(f"  Median filter kernel: {median_k}px  ({cfg.s2_diameter_um}µm / {pixel_size_um:.4f}µm/px ÷ 4)")
    print(f"  Morph close radius  : {morph_r_px}px  ({cfg.morph_close_radius_um}µm / {pixel_size_um:.4f}µm/px)")
    print(f"  Binary threshold    : {cfg.binary_threshold}  (pass --binary-threshold N to change)")

    for field_id, field_data, field_sizes in iter_fields(data, sizes):
        records = process_field(
            field_data, field_sizes, pixel_size_um,
            field_id, cfg, debug_dir, global_minmax,
        )
        all_records.extend(records)

    csv_path = output_dir / f"{nd2_path.stem}_aggregations.csv"
    save_csv(all_records, csv_path)


# ─── CLI entry point ──────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Detect and quantify S2 cell aggregations in nd2 fluorescence images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("nd2_path", type=Path, help="Input .nd2 file")
    p.add_argument("output_dir", type=Path, help="Directory for CSV and debug images")
    _d = Config()  # single source of truth for all defaults
    p.add_argument(
        "--s2-diameter", type=float, default=_d.s2_diameter_um,
        help="S2 cell diameter in µm (used to compute aggregation area threshold)",
    )
    p.add_argument(
        "--min-cells", type=int, default=_d.aggregation_min_cells,
        help="Minimum number of S2 cells to call a region an aggregation",
    )
    p.add_argument(
        "--morph-close-radius", type=float, default=_d.morph_close_radius_um,
        help="Disk radius (µm) for morphology close after hole filling",
    )
    p.add_argument(
        "--binary-threshold", type=int, default=_d.binary_threshold,
        help="Fixed intensity threshold (0–255) for binarization after median filter",
    )
    p.add_argument(
        "--min-active-channels", type=int, default=_d.min_active_channels,
        help="Minimum number of channels that must have a positive pixel in the component",
    )
    p.add_argument(
        "--debug", action="store_true",
        help="Save all intermediate images (per-channel MIPs, center slices, mask stages)",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config(
        s2_diameter_um=args.s2_diameter,
        aggregation_min_cells=args.min_cells,
        morph_close_radius_um=args.morph_close_radius,
        binary_threshold=args.binary_threshold,
        min_active_channels=args.min_active_channels,
        save_debug=args.debug,
    )
    process_nd2(args.nd2_path, args.output_dir, cfg)


if __name__ == "__main__":
    main()
