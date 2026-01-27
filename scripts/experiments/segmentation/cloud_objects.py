from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


# -----------------------------
# Robust repo root detection
# -----------------------------
def _find_repo_root(start: Path) -> Path:
    """
    Walk upward until we find pyproject.toml (preferred) or .git.
    This makes scripts resilient to folder reorganizations.
    """
    p = start.resolve()
    for _ in range(20):
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            return p
        if p == p.parent:
            break
        p = p.parent
    # fallback: 2 parents up from file (better than crashing)
    return start.resolve().parents[2]


REPO_ROOT = _find_repo_root(Path(__file__))


TS_RE = re.compile(r"(\d{8})_(\d{6})Z")  # e.g. 20260122_051000Z


# -----------------------------
# Connected components (robust)
# -----------------------------
def _label_connected_components(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    mask: bool (H,W)
    Returns (labels, n_labels), where labels are 0..n (0 = background)
    Uses scipy.ndimage if available, else skimage, else pure python BFS fallback.
    """
    try:
        from scipy.ndimage import label  # type: ignore

        labels, n = label(mask.astype(np.uint8))
        return labels.astype(np.int32), int(n)
    except Exception:
        pass

    try:
        from skimage.measure import label as sk_label  # type: ignore

        labels = sk_label(mask.astype(np.uint8), connectivity=2)
        n = int(labels.max())
        return labels.astype(np.int32), n
    except Exception:
        pass

    # Fallback: BFS (works fine for 256–1024 images)
    H, W = mask.shape
    labels = np.zeros((H, W), dtype=np.int32)
    n = 0

    # 8-connectivity
    neigh = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]

    for y in range(H):
        for x in range(W):
            if not mask[y, x] or labels[y, x] != 0:
                continue
            n += 1
            stack = [(y, x)]
            labels[y, x] = n
            while stack:
                cy, cx = stack.pop()
                for dy, dx in neigh:
                    ny, nx = cy + dy, cx + dx
                    if (
                        0 <= ny < H
                        and 0 <= nx < W
                        and mask[ny, nx]
                        and labels[ny, nx] == 0
                    ):
                        labels[ny, nx] = n
                        stack.append((ny, nx))

    return labels, n


# -----------------------------
# Helpers
# -----------------------------
def parse_timestamp_from_name(name: str) -> pd.Timestamp | None:
    m = TS_RE.search(name)
    if not m:
        return None
    ymd, hms = m.group(1), m.group(2)
    ts = pd.to_datetime(
        f"{ymd}{hms}", format="%Y%m%d%H%M%S", errors="coerce", utc=True
    )
    if pd.isna(ts):
        return None
    return ts


def _norm_relpath(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace("\\", "/", regex=False)


def _resolve_path(s: str, *, out_root: Path) -> Path:
    """
    Resolve a path string that might be:
      - absolute
      - repo-relative (e.g., out/masks/xxx.png)
      - out-relative (e.g., masks/xxx.png)
    """
    p = Path(s)

    if p.is_absolute():
        return p

    # First try repo-relative
    cand1 = (REPO_ROOT / p).resolve()
    if cand1.exists():
        return cand1

    # Then try outdir-relative
    cand2 = (out_root / p).resolve()
    if cand2.exists():
        return cand2

    # Common fallback: out/masks by filename
    cand3 = (out_root / "masks" / p.name).resolve()
    return cand3


def _load_metrics(metrics_csv: Path, *, out_root: Path) -> pd.DataFrame:
    if not metrics_csv.exists():
        raise FileNotFoundError(f"metrics_csv not found: {metrics_csv}")
    df = pd.read_csv(metrics_csv)

    # Normalize stored slashes
    for c in ["relpath", "mask_path"]:
        if c in df.columns:
            df[c] = _norm_relpath(df[c])

    # Parse timestamp if present
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)

    # If timestamp missing, try parse from filename
    if "timestamp_utc" not in df.columns or df["timestamp_utc"].isna().all():
        def _infer_ts(rp: str):
            return parse_timestamp_from_name(Path(rp).name)
        if "relpath" in df.columns:
            df["timestamp_utc"] = df["relpath"].astype(str).map(_infer_ts)

    # Keep readable frames if flagged
    if "img_ok" in df.columns:
        df = df[df["img_ok"] == 1].copy()

    # Require mask_path
    if "mask_path" not in df.columns:
        raise ValueError("cloud_metrics.csv must contain 'mask_path' (run cloud_mask_baseline.py).")

    # Resolve mask paths robustly + keep only existing masks
    resolved_masks: list[str] = []
    exists_flags: list[bool] = []

    for mp in df["mask_path"].astype(str).tolist():
        p = _resolve_path(mp, out_root=out_root)
        resolved_masks.append(str(p))
        exists_flags.append(p.exists())

    df["mask_path"] = resolved_masks
    df = df[pd.Series(exists_flags, index=df.index)].copy().reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("No rows with existing mask files. Check out/cloud_metrics.csv and out/masks/.")

    # Sort by time if available
    if "timestamp_utc" in df.columns:
        df = df.sort_values(["timestamp_utc", "mask_path"], na_position="last")

    return df


def _load_mask(mask_path: str) -> np.ndarray:
    p = Path(mask_path)
    im = Image.open(p).convert("L")
    arr = np.asarray(im, dtype=np.uint8)
    return arr > 127


def _overlay_objects_on_rgb(
    rgb: Image.Image, objects: List[Dict[str, Any]], alpha: int = 120
) -> Image.Image:
    """
    Draw bounding boxes + centroids on an RGB image.
    """
    base = rgb.convert("RGBA")
    draw = ImageDraw.Draw(base, "RGBA")

    # simple palette
    colors = [
        (255, 80, 80, alpha),
        (80, 255, 80, alpha),
        (80, 160, 255, alpha),
        (255, 200, 80, alpha),
        (200, 80, 255, alpha),
        (80, 255, 220, alpha),
    ]

    for i, obj in enumerate(objects):
        x0, y0, x1, y1 = obj["bbox"]
        cx, cy = obj["centroid_xy"]
        col = colors[i % len(colors)]

        draw.rectangle([x0, y0, x1, y1], outline=col[:3] + (255,), width=3)
        r = 4
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col[:3] + (255,))
        draw.text((x0 + 4, y0 + 4), f"#{obj['object_id']}", fill=col[:3] + (255,))

    return base.convert("RGB")


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
# Main extraction
# -----------------------------
@dataclass
class Params:
    min_area_px: int = 250          # filter specks
    max_objects_per_frame: int = 5  # keep top-K largest components
    save_overlays: bool = True
    overlay_alpha: int = 120
    contact_sheet_n: int = 12


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--metrics-csv",
        default=str(REPO_ROOT / "out" / "cloud_metrics.csv"),
        help="CSV from cloud_mask_baseline.py (default: out/cloud_metrics.csv)",
    )
    ap.add_argument("--outdir", default=str(REPO_ROOT / "out"), help="Base output directory (default: out/)")
    ap.add_argument("--min-area-px", type=int, default=250, help="Minimum component area to keep (pixels)")
    ap.add_argument("--max-objects", type=int, default=5, help="Keep top-K largest objects per frame")
    ap.add_argument("--max-frames", type=int, default=0, help="If >0, only process most recent N frames")
    ap.add_argument("--save-overlays", action="store_true", help="Save per-frame overlay images")
    ap.add_argument("--overlay-alpha", type=int, default=120, help="Overlay alpha 0..255")
    ap.add_argument("--contact-sheet-n", type=int, default=12, help="How many latest overlays to include in contact sheet")
    args = ap.parse_args()

    params = Params(
        min_area_px=int(args.min_area_px),
        max_objects_per_frame=int(args.max_objects),
        save_overlays=bool(args.save_overlays),
        overlay_alpha=int(args.overlay_alpha),
        contact_sheet_n=int(args.contact_sheet_n),
    )

    metrics_csv = Path(args.metrics_csv).resolve()
    out_root = Path(args.outdir).resolve()
    out_viz = out_root / "viz"
    out_objs_dir = out_viz / "cloud_objects_overlays"
    out_viz.mkdir(parents=True, exist_ok=True)
    if params.save_overlays:
        out_objs_dir.mkdir(parents=True, exist_ok=True)

    df = _load_metrics(metrics_csv, out_root=out_root)

    if args.max_frames and int(args.max_frames) > 0:
        df = df.tail(int(args.max_frames)).copy().reset_index(drop=True)

    object_rows: List[Dict[str, Any]] = []
    frame_rows: List[Dict[str, Any]] = []

    overlays_for_sheet: List[Image.Image] = []

    for _, row in df.iterrows():
        mask_path = str(row["mask_path"])
        relpath = str(row["relpath"]) if "relpath" in row and pd.notna(row["relpath"]) else ""
        ts = row["timestamp_utc"] if "timestamp_utc" in row else None

        mask = _load_mask(mask_path)
        H, W = mask.shape
        labels, n = _label_connected_components(mask)

        if n == 0:
            frame_rows.append(
                {
                    "timestamp_utc": ts,
                    "relpath": relpath,
                    "mask_path": mask_path,
                    "n_objects": 0,
                    "total_cloud_frac": float(mask.mean()),
                    "largest_area_px": 0,
                    "largest_area_frac": 0.0,
                    "largest_centroid_x": np.nan,
                    "largest_centroid_y": np.nan,
                }
            )
            continue

        comps = []
        for lab in range(1, n + 1):
            ys, xs = np.where(labels == lab)
            area = int(len(xs))
            if area < params.min_area_px:
                continue
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            cx = float(xs.mean())
            cy = float(ys.mean())
            comps.append(
                {
                    "object_id": lab,
                    "area_px": area,
                    "area_frac": area / float(H * W),
                    "bbox": (x0, y0, x1, y1),
                    "centroid_xy": (cx, cy),
                }
            )

        comps = sorted(comps, key=lambda d: d["area_px"], reverse=True)[: params.max_objects_per_frame]

        for rank, obj in enumerate(comps, start=1):
            object_rows.append(
                {
                    "timestamp_utc": ts,
                    "relpath": relpath,
                    "mask_path": mask_path,
                    "object_rank": rank,
                    "object_id": obj["object_id"],
                    "area_px": obj["area_px"],
                    "area_frac": obj["area_frac"],
                    "bbox_x0": obj["bbox"][0],
                    "bbox_y0": obj["bbox"][1],
                    "bbox_x1": obj["bbox"][2],
                    "bbox_y1": obj["bbox"][3],
                    "centroid_x": obj["centroid_xy"][0],
                    "centroid_y": obj["centroid_xy"][1],
                }
            )

        if comps:
            largest = comps[0]
            frame_rows.append(
                {
                    "timestamp_utc": ts,
                    "relpath": relpath,
                    "mask_path": mask_path,
                    "n_objects": len(comps),
                    "total_cloud_frac": float(mask.mean()),
                    "largest_area_px": int(largest["area_px"]),
                    "largest_area_frac": float(largest["area_frac"]),
                    "largest_centroid_x": float(largest["centroid_xy"][0]),
                    "largest_centroid_y": float(largest["centroid_xy"][1]),
                }
            )
        else:
            frame_rows.append(
                {
                    "timestamp_utc": ts,
                    "relpath": relpath,
                    "mask_path": mask_path,
                    "n_objects": 0,
                    "total_cloud_frac": float(mask.mean()),
                    "largest_area_px": 0,
                    "largest_area_frac": 0.0,
                    "largest_centroid_x": np.nan,
                    "largest_centroid_y": np.nan,
                }
            )

        # Optional overlay on the RGB image (if relpath exists)
        if params.save_overlays and relpath:
            rgb_path = _resolve_path(relpath, out_root=out_root)
            if rgb_path.exists():
                rgb = Image.open(rgb_path).convert("RGB")
                if rgb.size != (W, H):
                    rgb = rgb.resize((W, H), resample=Image.Resampling.BILINEAR)

                ov = _overlay_objects_on_rgb(rgb, comps, alpha=params.overlay_alpha)
                out_overlay = out_objs_dir / (Path(relpath).stem + "_objects.png")
                ov.save(out_overlay)
                overlays_for_sheet.append(ov)

    objs_df = pd.DataFrame(object_rows)
    frames_df = pd.DataFrame(frame_rows)

    if "timestamp_utc" in frames_df.columns:
        frames_df["timestamp_utc"] = pd.to_datetime(frames_df["timestamp_utc"], errors="coerce", utc=True)
        frames_df = frames_df.sort_values(["timestamp_utc", "mask_path"], na_position="last")

    if "timestamp_utc" in objs_df.columns and not objs_df.empty:
        objs_df["timestamp_utc"] = pd.to_datetime(objs_df["timestamp_utc"], errors="coerce", utc=True)
        objs_df = objs_df.sort_values(["timestamp_utc", "object_rank"], na_position="last")

    out_objects_csv = out_root / "cloud_objects.csv"
    out_frames_csv = out_root / "cloud_objects_frames.csv"
    objs_df.to_csv(out_objects_csv, index=False)
    frames_df.to_csv(out_frames_csv, index=False)

    sheet_png = None
    if params.save_overlays and overlays_for_sheet:
        n = min(params.contact_sheet_n, len(overlays_for_sheet))
        last = overlays_for_sheet[-n:]
        sheet = _make_contact_sheet(last, cols=4, thumb_px=256)
        sheet_png = out_viz / "cloud_objects_contact_sheet.png"
        sheet.save(sheet_png)

    print(f"Read frames: {len(df)}")
    print(f"Wrote: {out_objects_csv}")
    print(f"Wrote: {out_frames_csv}")
    if params.save_overlays:
        print(f"Wrote overlays to: {out_objs_dir}")
    if sheet_png is not None:
        print(f"Wrote: {sheet_png}")
    print(f"Params: min_area_px={params.min_area_px}, max_objects_per_frame={params.max_objects_per_frame}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
