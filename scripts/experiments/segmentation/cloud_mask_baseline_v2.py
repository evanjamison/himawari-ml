#!/usr/bin/env python
"""
cloud_mask_baseline_v2.py

Improved heuristic cloud mask generator for Himawari imagery.
Designed explicitly as a *teacher* for U-Net training.

Outputs:
- out/masks/*.png          (binary cloud masks)
- out/cloud_metrics.csv    (training manifest)
- optional overlay images
"""

from __future__ import annotations

import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


# ----------------------------
# Utilities
# ----------------------------

def load_image(path: Path, size: int | None):
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"Failed to read {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if size is not None:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img


def earth_disk_mask(h: int, w: int, margin: int = 4) -> np.ndarray:
    """Binary mask for Earth disk (removes space background)."""
    yy, xx = np.mgrid[:h, :w]
    cy, cx = h // 2, w // 2
    r = min(cx, cy) - margin
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return (dist <= r).astype(np.uint8)


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 1
    return out


# ----------------------------
# Core baseline logic
# ----------------------------

def cloud_mask_baseline(
    rgb: np.ndarray,
    luma_thresh: float,
    sat_thresh: float,
    open_size: int,
    close_size: int,
    min_area: int,
) -> np.ndarray:
    """
    Generate cloud mask from RGB image.
    """

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    v = hsv[..., 2].astype(np.float32) / 255.0
    s = hsv[..., 1].astype(np.float32) / 255.0

    # Bright & low-sat → clouds
    cloud_raw = (v >= luma_thresh) & (s <= sat_thresh)
    cloud = cloud_raw.astype(np.uint8)

    # Morphological cleanup
    if open_size > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
        cloud = cv2.morphologyEx(cloud, cv2.MORPH_OPEN, k)

    if close_size > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
        cloud = cv2.morphologyEx(cloud, cv2.MORPH_CLOSE, k)

    # Remove tiny speckles
    cloud = remove_small_components(cloud, min_area)

    return cloud


def make_overlay(rgb: np.ndarray, mask: np.ndarray, alpha: float):
    overlay = rgb.copy()
    pink = np.zeros_like(rgb)
    pink[..., 0] = 255
    pink[..., 2] = 255
    overlay[mask == 1] = (
        (1 - alpha) * overlay[mask == 1] + alpha * pink[mask == 1]
    ).astype(np.uint8)
    return overlay


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--luma-thresh", type=float, default=0.60)
    ap.add_argument("--sat-thresh", type=float, default=0.25)
    ap.add_argument("--open-size", type=int, default=3)
    ap.add_argument("--close-size", type=int, default=7)
    ap.add_argument("--min-area", type=int, default=80)
    ap.add_argument("--overlay-alpha", type=float, default=0.45)
    ap.add_argument("--save-overlays", action="store_true")
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    outdir = Path(args.outdir)
    mask_dir = outdir / "masks"
    overlay_dir = outdir / "overlays"

    mask_dir.mkdir(parents=True, exist_ok=True)
    if args.save_overlays:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted(raw_dir.glob("*.png"))
    if args.max_frames:
        frames = frames[: args.max_frames]

    records = []

    for fp in tqdm(frames, desc="Processing"):
        rgb = load_image(fp, args.image_size)
        h, w, _ = rgb.shape

        disk = earth_disk_mask(h, w)
        cloud = cloud_mask_baseline(
            rgb,
            luma_thresh=args.luma_thresh,
            sat_thresh=args.sat_thresh,
            open_size=args.open_size,
            close_size=args.close_size,
            min_area=args.min_area,
        )

        cloud = cloud * disk

        out_mask = (cloud * 255).astype(np.uint8)
        mask_path = mask_dir / f"{fp.stem}_mask.png"
        cv2.imwrite(str(mask_path), out_mask)

        if args.save_overlays:
            overlay = make_overlay(rgb, cloud, args.overlay_alpha)
            cv2.imwrite(str(overlay_dir / f"{fp.stem}_overlay.png"),
                        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        records.append({
            "relpath": str(fp),
            "mask_path": str(mask_path),
            "cloud_frac": float(cloud.mean()),
        })

    df = pd.DataFrame.from_records(records)
    df.to_csv(outdir / "cloud_metrics.csv", index=False)

    print(f"Processed {len(df)} frames")
    print(f"Wrote masks to {mask_dir}")
    print(f"Wrote metrics to {outdir / 'cloud_metrics.csv'}")


if __name__ == "__main__":
    main()
