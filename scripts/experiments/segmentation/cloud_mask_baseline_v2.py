#!/usr/bin/env python
"""
cloud_mask_baseline_b13.py

Heuristic cloud mask generator for Himawari-9 B13 (10.4 µm thermal IR) imagery.
Designed as a teacher signal for U-Net training.

B13 is illumination-invariant -- no terminator line, no day/night split needed.
Cold cloud tops are bright, clear sky and warm ocean are dark, consistently 24/7.

Detection logic:
  - Bright pixels (V >= luma_thresh) = cloud
  - Optional adaptive Gaussian threshold for structured cloud edges
  - Optional hybrid: adaptive OR global (recommended)

Outputs:
  - out/masks/*.png         (binary cloud masks, 255=cloud 0=clear)
  - out/overlays/*.png      (optional pink overlay for visual QC)
  - out/cloud_metrics.csv   (training manifest)
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


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 1
    return out


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
# Core mask logic
# ----------------------------

def cloud_mask_b13(
    rgb: np.ndarray,
    luma_thresh: float,
    open_size: int,
    close_size: int,
    min_area: int,
    adaptive: bool,
    adaptive_block: int,
    adaptive_c: int,
    hybrid: bool,
) -> np.ndarray:
    """
    Generate cloud mask from B13 grayscale image.

    B13 is grayscale so saturation filtering is not needed.
    Bright = cold cloud tops, dark = warm clear sky/ocean.

    Modes:
      global (default): V >= luma_thresh
      adaptive:         local Gaussian threshold
      hybrid:           adaptive OR global (recommended — adaptive catches
                        cloud edges, global catches uniform overcast)
    """
    # B13 is grayscale but stored as RGB -- all channels equal, just use V
    v = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    if hybrid or adaptive:
        v_uint8 = (v * 255).astype(np.uint8)
        adp = cv2.adaptiveThreshold(
            v_uint8, 1,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=adaptive_block,
            C=adaptive_c,
        ).astype(bool)

    if hybrid:
        cloud = (adp | (v >= luma_thresh)).astype(np.uint8)
    elif adaptive:
        cloud = adp.astype(np.uint8)
    else:
        cloud = (v >= luma_thresh).astype(np.uint8)

    # Morphological cleanup
    if open_size > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
        cloud = cv2.morphologyEx(cloud, cv2.MORPH_OPEN, k)
    if close_size > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
        cloud = cv2.morphologyEx(cloud, cv2.MORPH_CLOSE, k)

    cloud = remove_small_components(cloud, min_area)
    return cloud


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Heuristic cloud mask for Himawari B13 thermal IR imagery."
    )
    ap.add_argument("--raw-dir", required=True,
                    help="Directory of input PNG frames.")
    ap.add_argument("--outdir", default="out",
                    help="Output directory for masks, overlays, and CSV.")
    ap.add_argument("--image-size", type=int, default=256,
                    help="Resize frames to this square size (default 256).")
    ap.add_argument("--luma-thresh", type=float, default=0.45,
                    help="Global brightness threshold (default 0.45). "
                         "Pixels brighter than this are flagged as cloud. "
                         "B13: cold tops are bright, clear sky is dark.")
    ap.add_argument("--open-size", type=int, default=3,
                    help="Morphological open kernel size (default 3, 0=off).")
    ap.add_argument("--close-size", type=int, default=7,
                    help="Morphological close kernel size (default 7, 0=off).")
    ap.add_argument("--min-area", type=int, default=80,
                    help="Remove connected components smaller than this (default 80px).")
    ap.add_argument("--overlay-alpha", type=float, default=0.45,
                    help="Pink overlay opacity for QC images (default 0.45).")
    ap.add_argument("--save-overlays", action="store_true",
                    help="Save pink overlay QC images to outdir/overlays/.")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="Process only the first N frames (for quick testing).")

    # Threshold mode
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--adaptive", action="store_true",
                      help="Pure local Gaussian adaptive threshold.")
    mode.add_argument("--hybrid", action="store_true",
                      help="[RECOMMENDED] Adaptive OR global threshold. "
                           "Adaptive catches cloud edges, global catches "
                           "uniform overcast with little local contrast.")

    ap.add_argument("--adaptive-block", type=int, default=101,
                    help="Neighbourhood size for adaptive threshold (must be odd, default 101).")
    ap.add_argument("--adaptive-c", type=int, default=-8,
                    help="Constant subtracted from local mean (default -8). "
                         "More negative = stricter.")

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

    if not frames:
        print(f"No PNG frames found in {raw_dir}")
        return

    records = []

    for fp in tqdm(frames, desc="Processing"):
        rgb = load_image(fp, args.image_size)

        cloud = cloud_mask_b13(
            rgb,
            luma_thresh=args.luma_thresh,
            open_size=args.open_size,
            close_size=args.close_size,
            min_area=args.min_area,
            adaptive=args.adaptive,
            adaptive_block=args.adaptive_block,
            adaptive_c=args.adaptive_c,
            hybrid=args.hybrid,
        )

        out_mask = (cloud * 255).astype(np.uint8)
        mask_path = mask_dir / f"{fp.stem}_mask.png"
        cv2.imwrite(str(mask_path), out_mask)

        if args.save_overlays:
            overlay = make_overlay(rgb, cloud, args.overlay_alpha)
            cv2.imwrite(
                str(overlay_dir / f"{fp.stem}_overlay.png"),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
            )

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
