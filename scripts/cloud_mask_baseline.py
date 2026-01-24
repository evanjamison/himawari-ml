from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR_DEFAULT = REPO_ROOT / "data" / "raw" / "sequence"


TS_RE = re.compile(r"(\d{8})_(\d{6})Z")  # e.g. 20260122_051000Z


@dataclass
class MaskParams:
    # “Cloud-like” heuristic: bright + low saturation
    luma_thresh: float = 0.60  # [0..1] on luma
    sat_thresh: float = 0.25   # [0..1] proxy saturation (lower = whiter)
    # Morphology-ish cleanup using PIL filters (odd integers; 0 disables)
    open_size: int = 3         # opening removes speckle (min then max)
    close_size: int = 5        # closing fills holes (max then min)


def parse_timestamp_from_name(name: str) -> pd.Timestamp | None:
    """
    Parses timestamps like ..._YYYYMMDD_HHMMSSZ...
    Returns UTC pandas Timestamp or None if not found.
    """
    m = TS_RE.search(name)
    if not m:
        return None
    ymd, hms = m.group(1), m.group(2)
    # YYYYMMDDHHMMSS
    s = f"{ymd}{hms}"
    ts = pd.to_datetime(s, format="%Y%m%d%H%M%S", errors="coerce", utc=True)
    if pd.isna(ts):
        return None
    return ts


def list_images(raw_dir: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    paths = [p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    # Sort by parsed timestamp if possible, else by name
    def key(p: Path):
        ts = parse_timestamp_from_name(p.name)
        return (pd.Timestamp.min.tz_localize("UTC") if ts is None else ts, p.name)
    return sorted(paths, key=key)


def pil_to_np_rgb(im: Image.Image) -> np.ndarray:
    im = im.convert("RGB")
    arr = np.asarray(im, dtype=np.float32) / 255.0  # HxWx3
    return arr


def compute_cloud_mask(rgb: np.ndarray, params: MaskParams) -> np.ndarray:
    """
    rgb: float32 [0..1], shape (H,W,3)
    Returns boolean mask (H,W) where True = cloud.
    """
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]

    # Luma (Rec. 709)
    luma = 0.2126 * r + 0.7152 * g + 0.0722 * b

    # Saturation proxy: (max - min) / (max + eps)
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    sat = (mx - mn) / (mx + 1e-6)

    # Cloud heuristic: bright and low-saturation (“white-ish”)
    mask = (luma >= params.luma_thresh) & (sat <= params.sat_thresh)

    return mask


def apply_morphology(mask_bool: np.ndarray, params: MaskParams) -> np.ndarray:
    """
    Uses PIL MinFilter/MaxFilter to do opening/closing-like cleanup.
    """
    m = (mask_bool.astype(np.uint8) * 255)
    im = Image.fromarray(m, mode="L")

    # Opening: Min -> Max (removes small bright specks)
    if params.open_size and params.open_size >= 3:
        if params.open_size % 2 == 0:
            raise ValueError("--open-size must be odd (or 0)")
        im = im.filter(ImageFilter.MinFilter(params.open_size))
        im = im.filter(ImageFilter.MaxFilter(params.open_size))

    # Closing: Max -> Min (fills small holes)
    if params.close_size and params.close_size >= 3:
        if params.close_size % 2 == 0:
            raise ValueError("--close-size must be odd (or 0)")
        im = im.filter(ImageFilter.MaxFilter(params.close_size))
        im = im.filter(ImageFilter.MinFilter(params.close_size))

    out = (np.asarray(im, dtype=np.uint8) > 127)
    return out


def overlay_mask(im_rgb: Image.Image, mask_bool: np.ndarray, alpha: float = 0.35) -> Image.Image:
    """
    Red overlay where mask is True.
    """
    base = im_rgb.convert("RGBA")
    h, w = mask_bool.shape
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    # red pixels where cloud
    red = Image.new("RGBA", (w, h), (255, 0, 0, int(255 * alpha)))
    # mask as L
    m = Image.fromarray((mask_bool.astype(np.uint8) * 255), mode="L")
    overlay.paste(red, (0, 0), m)
    return Image.alpha_composite(base, overlay).convert("RGB")


def make_contact_sheet(images: list[Image.Image], cols: int = 4, thumb_px: int = 256) -> Image.Image:
    if not images:
        raise ValueError("No images for contact sheet")
    thumbs = []
    for im in images:
        im2 = im.copy()
        im2.thumbnail((thumb_px, thumb_px))
        thumbs.append(im2)

    cols = max(1, cols)
    rows = (len(thumbs) + cols - 1) // cols
    w = cols * thumb_px
    h = rows * thumb_px
    sheet = Image.new("RGB", (w, h), (255, 255, 255))

    for i, im in enumerate(thumbs):
        r, c = divmod(i, cols)
        x0 = c * thumb_px + (thumb_px - im.width) // 2
        y0 = r * thumb_px + (thumb_px - im.height) // 2
        sheet.paste(im, (x0, y0))
    return sheet


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=str(RAW_DIR_DEFAULT), help="Directory containing raw frames (default: data/raw/sequence)")
    ap.add_argument("--outdir", default=str(REPO_ROOT / "out"), help="Base output directory (default: out/)")
    ap.add_argument("--luma-thresh", type=float, default=0.60, help="Cloud luma threshold in [0..1]")
    ap.add_argument("--sat-thresh", type=float, default=0.25, help="Cloud saturation proxy threshold in [0..1] (lower = whiter)")
    ap.add_argument("--open-size", type=int, default=3, help="Odd kernel size for opening (0 disables)")
    ap.add_argument("--close-size", type=int, default=5, help="Odd kernel size for closing (0 disables)")
    ap.add_argument("--max-frames", type=int, default=0, help="If >0, process only the most recent N frames")
    ap.add_argument("--save-overlays", action="store_true", help="Save per-frame overlay images")
    ap.add_argument("--overlay-alpha", type=float, default=0.35, help="Overlay alpha [0..1]")
    ap.add_argument("--contact-sheet-n", type=int, default=12, help="How many latest overlays to include in contact sheet")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir).resolve()
    out_root = Path(args.outdir).resolve()
    out_masks = out_root / "masks"
    out_viz = out_root / "viz"
    out_overlays = out_viz / "mask_overlays"

    out_masks.mkdir(parents=True, exist_ok=True)
    out_viz.mkdir(parents=True, exist_ok=True)
    if args.save_overlays:
        out_overlays.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        raise SystemExit(f"Raw dir not found: {raw_dir}")

    params = MaskParams(
        luma_thresh=float(args.luma_thresh),
        sat_thresh=float(args.sat_thresh),
        open_size=int(args.open_size),
        close_size=int(args.close_size),
    )

    img_paths = list_images(raw_dir)
    if not img_paths:
        raise SystemExit(f"No images found under: {raw_dir}")

    # Keep only most recent N frames if requested
    if args.max_frames and args.max_frames > 0:
        img_paths = img_paths[-int(args.max_frames):]

    rows = []
    overlays_for_sheet: list[Image.Image] = []

    for p in img_paths:
        relpath = p.relative_to(REPO_ROOT).as_posix()
        ts = parse_timestamp_from_name(p.name)

        try:
            im = Image.open(p).convert("RGB")
            rgb = pil_to_np_rgb(im)
        except Exception:
            # Skip unreadable
            rows.append(
                {
                    "relpath": relpath,
                    "timestamp_utc": ts,
                    "img_ok": 0,
                    "cloud_fraction": np.nan,
                    "mean_luma": np.nan,
                    "std_luma": np.nan,
                }
            )
            continue

        # base stats
        luma = (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2])
        mean_luma = float(np.mean(luma) * 255.0)
        std_luma = float(np.std(luma) * 255.0)

        # mask + cleanup
        mask0 = compute_cloud_mask(rgb, params)
        mask = apply_morphology(mask0, params)

        cloud_frac = float(mask.mean())

        # Save mask
        mask_png = out_masks / (p.stem + "_mask.png")
        Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(mask_png)

        # Optional overlay
        if args.save_overlays:
            ov = overlay_mask(im, mask, alpha=float(args.overlay_alpha))
            ov_path = out_overlays / (p.stem + "_overlay.png")
            ov.save(ov_path)

            # Collect for contact sheet (latest N only)
            overlays_for_sheet.append(ov)

        rows.append(
            {
                "relpath": relpath,
                "timestamp_utc": ts,
                "img_ok": 1,
                "cloud_fraction": cloud_frac,
                "mean_luma": mean_luma,
                "std_luma": std_luma,
                "mask_path": mask_png.relative_to(REPO_ROOT).as_posix(),
            }
        )

    df = pd.DataFrame(rows)
    # Ensure timestamp sorting
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
        df = df.sort_values(["timestamp_utc", "relpath"], na_position="last")

    # Write metrics CSV
    metrics_csv = out_root / "cloud_metrics.csv"
    df.to_csv(metrics_csv, index=False)

    # Plot cloud fraction timeseries (only for readable + timestamped)
    dff = df[(df["img_ok"] == 1) & df["timestamp_utc"].notna()].copy()
    if not dff.empty:
        plt.figure()
        plt.plot(dff["timestamp_utc"], dff["cloud_fraction"])
        plt.title("Cloud fraction over time (baseline mask)")
        plt.xlabel("Timestamp (UTC)")
        plt.ylabel("cloud_fraction")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        ts_png = out_viz / "cloud_fraction_timeseries.png"
        plt.savefig(ts_png, dpi=200, bbox_inches="tight")
        plt.close()
    else:
        ts_png = None

    # Contact sheet of latest overlays (if overlays were saved)
    sheet_png = None
    if args.save_overlays and overlays_for_sheet:
        n = int(args.contact_sheet_n)
        overlays_for_sheet = overlays_for_sheet[-n:]
        sheet = make_contact_sheet(overlays_for_sheet, cols=4, thumb_px=256)
        sheet_png = out_viz / "cloud_mask_overlays_contact_sheet.png"
        sheet.save(sheet_png)

    # Print summary
    ok = int((df["img_ok"] == 1).sum())
    total = len(df)
    print(f"Processed: {ok}/{total} readable frames from {raw_dir}")
    print(f"Wrote: {metrics_csv}")
    print(f"Wrote masks to: {out_masks}")
    if ts_png is not None:
        print(f"Wrote: {ts_png}")
    if sheet_png is not None:
        print(f"Wrote: {sheet_png}")
    if args.save_overlays:
        print(f"Wrote overlays to: {out_overlays}")
    print(f"Mask params: luma_thresh={params.luma_thresh}, sat_thresh={params.sat_thresh}, open={params.open_size}, close={params.close_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
