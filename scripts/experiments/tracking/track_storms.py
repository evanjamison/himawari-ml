from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]


# -----------------------------
# Helpers
# -----------------------------
def _norm_path(s: str) -> str:
    return str(s).replace("\\", "/")


def _safe_ts(x) -> pd.Timestamp | pd.NaT:
    return pd.to_datetime(x, errors="coerce", utc=True)


def _load_frames_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Frames CSV not found: {path}")

    df = pd.read_csv(path)

    required = ["timestamp_utc", "relpath", "mask_path", "n_objects", "largest_centroid_x", "largest_centroid_y"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    df["relpath"] = df["relpath"].astype(str).map(_norm_path)
    df["mask_path"] = df["mask_path"].astype(str).map(_norm_path)
    df["timestamp_utc"] = df["timestamp_utc"].map(_safe_ts)

    # Drop rows without timestamp or centroid
    df = df.dropna(subset=["timestamp_utc"]).copy()
    df = df.reset_index(drop=True)

    # Ensure numeric
    for c in ["largest_centroid_x", "largest_centroid_y", "n_objects"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    return df


def _euclid(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).sum()))


def _load_rgb(relpath: str) -> Optional[Image.Image]:
    if not relpath:
        return None
    p = REPO_ROOT / relpath
    if not p.exists():
        return None
    return Image.open(p).convert("RGB")


def _draw_track_on_image(
    img: Image.Image,
    xs: List[float],
    ys: List[float],
    radius: int = 5,
) -> Image.Image:
    out = img.copy().convert("RGBA")
    d = ImageDraw.Draw(out, "RGBA")

    # Draw polyline
    pts = [(float(x), float(y)) for x, y in zip(xs, ys) if np.isfinite(x) and np.isfinite(y)]
    if len(pts) >= 2:
        d.line(pts, fill=(255, 60, 60, 220), width=4)

    # Draw points
    for i, (x, y) in enumerate(pts):
        r = radius
        d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 200, 60, 230), outline=(255, 255, 255, 240), width=2)
        # label every few points to avoid clutter
        if i % 3 == 0:
            d.text((x + r + 2, y + r + 2), f"{i}", fill=(255, 255, 255, 240))

    return out.convert("RGB")


def _make_contact_sheet(images: List[Image.Image], cols: int = 4, thumb_px: int = 256) -> Image.Image:
    thumbs = []
    for im in images:
        im2 = im.copy()
        im2.thumbnail((thumb_px, thumb_px))
        thumbs.append(im2)

    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_px, rows * thumb_px), (255, 255, 255))

    for i, im in enumerate(thumbs):
        r, c = divmod(i, cols)
        x0 = c * thumb_px + (thumb_px - im.width) // 2
        y0 = r * thumb_px + (thumb_px - im.height) // 2
        sheet.paste(im, (x0, y0))
    return sheet


# -----------------------------
# Core tracking
# -----------------------------
@dataclass
class TrackParams:
    max_jump_px: float = 120.0     # reject centroid jumps larger than this between frames
    min_objects: int = 1           # require at least this many objects in frame (largest exists)
    smooth_window: int = 3         # rolling mean window for centroid smoothing (odd recommended)
    save_contact_sheet: bool = True
    contact_sheet_n: int = 12


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-csv", default=str(REPO_ROOT / "out" / "cloud_objects_frames.csv"),
                    help="Output of scripts/cloud_objects.py (default: out/cloud_objects_frames.csv)")
    ap.add_argument("--outdir", default=str(REPO_ROOT / "out"), help="Base output directory (default: out/)")
    ap.add_argument("--max-jump-px", type=float, default=120.0, help="Reject frame-to-frame centroid jumps above this")
    ap.add_argument("--smooth-window", type=int, default=3, help="Rolling mean smoothing window for centroids")
    ap.add_argument("--save-contact-sheet", action="store_true", help="Save per-frame overlay contact sheet")
    ap.add_argument("--contact-sheet-n", type=int, default=12, help="Number of latest frames in contact sheet")
    ap.add_argument("--max-frames", type=int, default=0, help="If >0, only use most recent N frames")
    args = ap.parse_args()

    params = TrackParams(
        max_jump_px=float(args.max_jump_px),
        smooth_window=int(args.smooth_window),
        save_contact_sheet=bool(args.save_contact_sheet),
        contact_sheet_n=int(args.contact_sheet_n),
    )

    out_root = Path(args.outdir).resolve()
    out_viz = out_root / "viz"
    out_viz.mkdir(parents=True, exist_ok=True)

    df = _load_frames_csv(Path(args.frames_csv).resolve())

    if args.max_frames and int(args.max_frames) > 0:
        df = df.tail(int(args.max_frames)).copy().reset_index(drop=True)

    # Track uses "largest_centroid_*" (dominant object per frame)
    x = df["largest_centroid_x"].to_numpy(dtype=float)
    y = df["largest_centroid_y"].to_numpy(dtype=float)
    t = df["timestamp_utc"].to_numpy()

    # Valid frame: has objects and finite centroid
    valid = (df["n_objects"].fillna(0) >= 1) & np.isfinite(x) & np.isfinite(y)
    df["track_valid"] = valid.astype(int)

    # Reject huge jumps (data glitches)
    jump_ok = np.ones(len(df), dtype=bool)
    last_idx = None
    for i in range(len(df)):
        if not valid.iloc[i]:
            jump_ok[i] = False
            continue
        if last_idx is None:
            last_idx = i
            continue
        a = np.array([x[last_idx], y[last_idx]], dtype=float)
        b = np.array([x[i], y[i]], dtype=float)
        dist = _euclid(a, b)
        if dist > params.max_jump_px:
            jump_ok[i] = False
        else:
            last_idx = i

    df["jump_ok"] = jump_ok.astype(int)
    df["use_for_track"] = (df["track_valid"].astype(bool) & df["jump_ok"].astype(bool)).astype(int)

    # Apply smoothing to the usable track points only
    xs = pd.Series(x).where(df["use_for_track"].astype(bool))
    ys = pd.Series(y).where(df["use_for_track"].astype(bool))

    # Rolling mean with min_periods=1 so it fills early points; center=True for symmetric smoothing
    w = max(1, params.smooth_window)
    xs_s = xs.rolling(window=w, min_periods=1, center=True).mean()
    ys_s = ys.rolling(window=w, min_periods=1, center=True).mean()

    df["track_x"] = xs_s
    df["track_y"] = ys_s

    # Compute velocities between successive usable points
    speed_px_per_hr = [np.nan] * len(df)
    vx_px_per_hr = [np.nan] * len(df)
    vy_px_per_hr = [np.nan] * len(df)

    last_i = None
    for i in range(len(df)):
        if not bool(df.loc[i, "use_for_track"]):
            continue
        if last_i is None:
            last_i = i
            continue

        dt = (df.loc[i, "timestamp_utc"] - df.loc[last_i, "timestamp_utc"]).total_seconds()
        if dt <= 0:
            last_i = i
            continue

        dx = float(df.loc[i, "track_x"] - df.loc[last_i, "track_x"])
        dy = float(df.loc[i, "track_y"] - df.loc[last_i, "track_y"])
        v = math.sqrt(dx * dx + dy * dy) / (dt / 3600.0)

        speed_px_per_hr[i] = v
        vx_px_per_hr[i] = dx / (dt / 3600.0)
        vy_px_per_hr[i] = dy / (dt / 3600.0)
        last_i = i

    df["vx_px_per_hr"] = vx_px_per_hr
    df["vy_px_per_hr"] = vy_px_per_hr
    df["speed_px_per_hr"] = speed_px_per_hr

    # Save track CSV
    out_track_csv = out_root / "storm_track.csv"
    keep_cols = [
        "timestamp_utc",
        "relpath",
        "mask_path",
        "n_objects",
        "total_cloud_frac",
        "largest_area_px",
        "largest_area_frac",
        "track_valid",
        "jump_ok",
        "use_for_track",
        "track_x",
        "track_y",
        "vx_px_per_hr",
        "vy_px_per_hr",
        "speed_px_per_hr",
    ]
    for c in keep_cols:
        if c not in df.columns:
            df[c] = np.nan
    df[keep_cols].to_csv(out_track_csv, index=False)

    # -----------------------------
    # Plots (matplotlib)
    # -----------------------------
    import matplotlib.pyplot as plt

    # Timeseries plot
    fig = plt.figure(figsize=(10, 6))
    ax1 = plt.gca()

    # centroid x/y
    ax1.plot(df["timestamp_utc"], df["track_x"], label="centroid_x (smoothed)")
    ax1.plot(df["timestamp_utc"], df["track_y"], label="centroid_y (smoothed)")
    ax1.set_xlabel("Timestamp (UTC)")
    ax1.set_ylabel("pixels")
    ax1.set_title("Storm centroid track (dominant cloud object)")
    ax1.legend(loc="best")
    plt.tight_layout()
    out_ts_png = out_viz / "storm_track_timeseries.png"
    plt.savefig(out_ts_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Speed plot
    fig = plt.figure(figsize=(10, 4))
    ax = plt.gca()
    ax.plot(df["timestamp_utc"], df["speed_px_per_hr"], label="speed (px/hr)")
    ax.set_xlabel("Timestamp (UTC)")
    ax.set_ylabel("px/hr")
    ax.set_title("Storm motion speed (dominant object)")
    ax.legend(loc="best")
    plt.tight_layout()
    out_speed_png = out_viz / "storm_speed_timeseries.png"
    plt.savefig(out_speed_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Overlay track on the LAST RGB frame available
    # Find last usable relpath with file existing
    last_rgb = None
    last_rel = None
    for rp in reversed(df["relpath"].tolist()):
        img = _load_rgb(rp)
        if img is not None:
            last_rgb = img
            last_rel = rp
            break

    out_overlay_png = None
    if last_rgb is not None:
        xs_pts = df["track_x"].to_list()
        ys_pts = df["track_y"].to_list()
        overlay = _draw_track_on_image(last_rgb, xs_pts, ys_pts, radius=6)
        out_overlay_png = out_viz / "storm_track_overlay.png"
        overlay.save(out_overlay_png)

    # Optional per-frame contact sheet of overlays (last N)
    out_sheet_png = None
    if params.save_contact_sheet:
        overlays = []
        n = min(params.contact_sheet_n, len(df))
        tail = df.tail(n)

        for _, r in tail.iterrows():
            img = _load_rgb(r["relpath"])
            if img is None:
                continue
            # Draw a single point (current centroid)
            ov = img.copy().convert("RGBA")
            d = ImageDraw.Draw(ov, "RGBA")
            if bool(r["use_for_track"]) and np.isfinite(r["track_x"]) and np.isfinite(r["track_y"]):
                x0, y0 = float(r["track_x"]), float(r["track_y"])
                rr = 7
                d.ellipse([x0 - rr, y0 - rr, x0 + rr, y0 + rr], fill=(255, 60, 60, 220), outline=(255, 255, 255, 240), width=2)
            overlays.append(ov.convert("RGB"))

        if overlays:
            sheet = _make_contact_sheet(overlays, cols=4, thumb_px=256)
            out_sheet_png = out_viz / "storm_track_contact_sheet.png"
            sheet.save(out_sheet_png)

    # Print summary
    used = int(df["use_for_track"].sum())
    total = len(df)
    print(f"Frames: {total}")
    print(f"Track points used: {used}")
    print(f"Wrote: {out_track_csv}")
    print(f"Wrote: {out_ts_png}")
    print(f"Wrote: {out_speed_png}")
    if out_overlay_png is not None:
        print(f"Wrote: {out_overlay_png} (base frame: {last_rel})")
    if out_sheet_png is not None:
        print(f"Wrote: {out_sheet_png}")
    print(f"Params: max_jump_px={params.max_jump_px}, smooth_window={params.smooth_window}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
