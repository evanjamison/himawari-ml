"""
consolidate_master.py
=====================
Consolidates all ingested Himawari B13 frames from the rolling daily ingest
into a single flat master dataset ready for training.

Usage:
    python scripts/pipeline/consolidate_master.py
    python scripts/pipeline/consolidate_master.py --japan-raw data/japan_raw --outdir data/japan_master
    python scripts/pipeline/consolidate_master.py --dry-run

What it does:
    1. Scans data/japan_raw/{YYYY-MM-DD}/latest/*.png for all frames
    2. Copies them into data/japan_master/frames/ (flat, no subdirs)
    3. Runs cloud_mask_baseline_v2.py over all frames to produce masks
    4. Writes data/japan_master/cloud_metrics.csv (the training manifest)
    5. Prints a summary: total frames, date range, cloud fraction stats

Output structure:
    data/japan_master/
        frames/          <- all raw PNG frames, flat
        masks/           <- all binary cloud masks
        overlays/        <- optional QC overlays
        cloud_metrics.csv

The cloud_metrics.csv is the file you pass to train_unet_pixelmask.py
and train_convlstm.py via --metrics-csv.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd


REPO_ROOT = Path(__file__).resolve()
while not (REPO_ROOT / "pyproject.toml").exists() and not (REPO_ROOT / ".git").exists():
    if REPO_ROOT == REPO_ROOT.parent:
        break
    REPO_ROOT = REPO_ROOT.parent

BASELINE_SCRIPT = REPO_ROOT / "scripts" / "experiments" / "segmentation" / "cloud_mask_baseline_v2.py"


def find_all_frames(japan_raw: Path) -> list[Path]:
    """
    Scan japan_raw/{YYYY-MM-DD}/latest/*.png and return sorted list of all frames.
    Also checks japan_raw/{YYYY-MM-DD}/*.png for any flat-stored frames.
    """
    frames = []
    if not japan_raw.exists():
        return frames

    for day_dir in sorted(japan_raw.iterdir()):
        if not day_dir.is_dir():
            continue
        # Try latest/ subdir first (standard ingest layout)
        latest = day_dir / "latest"
        if latest.exists():
            frames.extend(sorted(latest.glob("*.png")))
        else:
            # Fall back to flat layout
            frames.extend(sorted(day_dir.glob("*.png")))

    return sorted(frames, key=lambda p: p.stem)


def copy_frames(frames: list[Path], dest: Path, dry_run: bool) -> list[Path]:
    """Copy all frames into dest/ (flat). Returns list of destination paths."""
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    skipped = 0

    for src in frames:
        dst = dest / src.name
        if dst.exists():
            skipped += 1
            copied.append(dst)
            continue
        if not dry_run:
            shutil.copy2(src, dst)
        copied.append(dst)

    print(f"  Frames: {len(frames)} found, {skipped} already present, "
          f"{len(frames) - skipped} newly copied")
    return sorted(copied, key=lambda p: p.stem)


def run_baseline(frames_dir: Path, outdir: Path, dry_run: bool, luma_thresh: float,
                 adaptive_c: int, adaptive_block: int) -> Path:
    """Run cloud_mask_baseline_v2.py over frames_dir and write output to outdir."""
    if not BASELINE_SCRIPT.exists():
        print(f"ERROR: baseline script not found at {BASELINE_SCRIPT}")
        sys.exit(1)

    cmd = [
        sys.executable, str(BASELINE_SCRIPT),
        "--raw-dir",        str(frames_dir),
        "--outdir",         str(outdir),
        "--image-size",     "256",
        "--save-overlays",
        "--hybrid",
        "--luma-thresh",    str(luma_thresh),
        "--adaptive-block", str(adaptive_block),
        "--adaptive-c",     str(adaptive_c),
        "--min-area",       "60",
    ]

    print(f"\n  Running baseline mask over {frames_dir}")
    print(f"  Command: {' '.join(cmd)}\n")

    if dry_run:
        print("  [DRY RUN] skipping baseline execution")
        return outdir / "cloud_metrics.csv"

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"ERROR: baseline script exited with code {result.returncode}")
        sys.exit(result.returncode)

    return outdir / "cloud_metrics.csv"


def print_summary(metrics_csv: Path, frames_dir: Path) -> None:
    """Print dataset summary stats."""
    if not metrics_csv.exists():
        print("  (metrics CSV not found, skipping summary)")
        return

    df = pd.read_csv(metrics_csv)
    total = len(df)

    # Extract dates from filenames
    stems = [Path(p).stem for p in df["relpath"].tolist()]
    dates = []
    for s in stems:
        try:
            # himawari_20260301T010000Z
            dt = datetime.strptime(s.split("_")[1], "%Y%m%dT%H%M%SZ")
            dates.append(dt)
        except Exception:
            pass

    print("\n" + "=" * 55)
    print("  MASTER DATASET SUMMARY")
    print("=" * 55)
    print(f"  Total frames      : {total}")
    if dates:
        print(f"  Date range        : {min(dates).strftime('%Y-%m-%d %H:%MZ')} "
              f"→ {max(dates).strftime('%Y-%m-%d %H:%MZ')}")
        unique_days = len(set(d.date() for d in dates))
        print(f"  Unique days       : {unique_days}")
    print(f"  Mean cloud frac   : {df['cloud_frac'].mean():.3f}")
    print(f"  Std  cloud frac   : {df['cloud_frac'].std():.3f}")
    print(f"  Min  cloud frac   : {df['cloud_frac'].min():.3f}")
    print(f"  Max  cloud frac   : {df['cloud_frac'].max():.3f}")
    print(f"  Frames dir        : {frames_dir}")
    print(f"  Metrics CSV       : {metrics_csv}")
    print("=" * 55)
    print()
    print("  Ready to train. Use:")
    print()
    print("  python ml/train_unet_pixelmask.py \\")
    print(f"    --frames-dir {frames_dir} \\")
    print(f"    --masks-dir  {metrics_csv.parent / 'masks'} \\")
    print("    --outdir out \\")
    print("    --split-by-time \\")
    print("    --augment \\")
    print("    --epochs 30")
    print()
    print("  python ml/train_convlstm.py \\")
    print(f"    --metrics-csv {metrics_csv} \\")
    print("    --outdir out \\")
    print("    --seq-len 6 --rollout-k 2 \\")
    print("    --epochs 30")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Consolidate all ingested B13 frames into a flat master training dataset."
    )
    ap.add_argument("--japan-raw", default=str(REPO_ROOT / "data" / "japan_raw"),
                    help="Root of daily ingest folders (default: data/japan_raw).")
    ap.add_argument("--outdir", default=str(REPO_ROOT / "data" / "japan_master"),
                    help="Output master directory (default: data/japan_master).")
    ap.add_argument("--luma-thresh", type=float, default=0.35,
                    help="Luma threshold for baseline mask (default: 0.35).")
    ap.add_argument("--adaptive-c", type=int, default=-6,
                    help="Adaptive threshold constant (default: -6).")
    ap.add_argument("--adaptive-block", type=int, default=101,
                    help="Adaptive block size (default: 101).")
    ap.add_argument("--skip-baseline", action="store_true",
                    help="Skip baseline mask generation (just consolidate frames).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be done without copying or running anything.")
    args = ap.parse_args()

    japan_raw = Path(args.japan_raw)
    outdir    = Path(args.outdir)
    frames_dir = outdir / "frames"
    masks_dir  = outdir / "masks"

    print(f"\nConsolidating B13 master dataset")
    print(f"  Source : {japan_raw}")
    print(f"  Output : {outdir}")
    if args.dry_run:
        print("  Mode   : DRY RUN\n")

    # 1. Find all frames
    print("\n[1/3] Scanning for frames...")
    all_frames = find_all_frames(japan_raw)
    if not all_frames:
        print(f"ERROR: No frames found under {japan_raw}")
        print("  Make sure you're on the data branch and have pulled the latest.")
        sys.exit(1)
    print(f"  Found {len(all_frames)} frames across "
          f"{len(set(p.parent.parent.name for p in all_frames))} day(s)")

    # 2. Copy frames into flat master/frames/
    print("\n[2/3] Copying frames to master...")
    copy_frames(all_frames, frames_dir, dry_run=args.dry_run)

    # 3. Run baseline mask
    if args.skip_baseline:
        print("\n[3/3] Skipping baseline mask generation (--skip-baseline)")
        metrics_csv = outdir / "cloud_metrics.csv"
    else:
        print("\n[3/3] Running baseline mask generation...")
        metrics_csv = run_baseline(
            frames_dir=frames_dir,
            outdir=outdir,
            dry_run=args.dry_run,
            luma_thresh=args.luma_thresh,
            adaptive_c=args.adaptive_c,
            adaptive_block=args.adaptive_block,
        )

    # 4. Summary
    print_summary(metrics_csv, frames_dir)


if __name__ == "__main__":
    main()
