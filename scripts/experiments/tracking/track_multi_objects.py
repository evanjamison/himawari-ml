from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]


# -----------------------------
# Utils
# -----------------------------
def _norm_path(s: str) -> str:
    return str(s).replace("\\", "/")


def _safe_ts(x) -> pd.Timestamp | pd.NaT:
    return pd.to_datetime(x, errors="coerce", utc=True)


def bbox_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))


def try_hungarian(cost: np.ndarray) -> Optional[List[Tuple[int, int]]]:
    """
    Return list of (row_idx, col_idx) assignments, or None if scipy not available.
    """
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore
        r, c = linear_sum_assignment(cost)
        return list(zip(r.tolist(), c.tolist()))
    except Exception:
        return None


def greedy_assignment(cost: np.ndarray) -> List[Tuple[int, int]]:
    """
    Greedy lowest-cost matching (fallback when scipy isn't installed).
    """
    pairs = []
    used_r = set()
    used_c = set()
    flat = [(cost[i, j], i, j) for i in range(cost.shape[0]) for j in range(cost.shape[1])]
    flat.sort(key=lambda x: x[0])
    for _, i, j in flat:
        if i in used_r or j in used_c:
            continue
        used_r.add(i)
        used_c.add(j)
        pairs.append((i, j))
    return pairs


def load_objects_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"objects csv not found: {path}")

    df = pd.read_csv(path)

    required = [
        "timestamp_utc", "relpath",
        "object_rank", "area_px",
        "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1",
        "centroid_x", "centroid_y",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    df["timestamp_utc"] = df["timestamp_utc"].map(_safe_ts)
    df = df.dropna(subset=["timestamp_utc"]).copy()

    df["relpath"] = df["relpath"].astype(str).map(_norm_path)

    num_cols = ["object_rank", "area_px", "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1", "centroid_x", "centroid_y"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["centroid_x", "centroid_y", "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1"]).copy()

    df = df.sort_values(["timestamp_utc", "object_rank"]).reset_index(drop=True)
    return df


def load_rgb(relpath: str) -> Optional[Image.Image]:
    if not relpath:
        return None
    p = REPO_ROOT / relpath
    if not p.exists():
        return None
    return Image.open(p).convert("RGB")


# -----------------------------
# Tracking
# -----------------------------
@dataclass
class TrackParams:
    top_k: int = 5
    max_dist_px: float = 180.0
    min_iou: float = 0.05
    iou_weight: float = 120.0  # how strongly IoU affects matching
    allow_match_if_either: bool = True  # if True: accept if dist ok OR iou ok (useful for big clouds)


def track_multi(df: pd.DataFrame, params: TrackParams) -> pd.DataFrame:
    # Keep only top-K objects per frame
    df = df[df["object_rank"] <= params.top_k].copy()
    df = df.sort_values(["timestamp_utc", "object_rank"]).reset_index(drop=True)

    # Group by timestamp
    times = sorted(df["timestamp_utc"].unique())

    next_track_id = 1
    active: Dict[int, Dict[str, object]] = {}  # track_id -> last state

    rows_out = []

    for t in times:
        frame = df[df["timestamp_utc"] == t].copy().reset_index(drop=True)
        if len(frame) == 0:
            continue

        # Build "detections" list
        det_centroids = [(float(frame.loc[i, "centroid_x"]), float(frame.loc[i, "centroid_y"])) for i in range(len(frame))]
        det_bboxes = [(float(frame.loc[i, "bbox_x0"]), float(frame.loc[i, "bbox_y0"]),
                       float(frame.loc[i, "bbox_x1"]), float(frame.loc[i, "bbox_y1"])) for i in range(len(frame))]

        # Build "tracks" list from active
        track_ids = list(active.keys())
        tr_centroids = []
        tr_bboxes = []
        for tid in track_ids:
            tr_centroids.append(active[tid]["centroid"])  # type: ignore
            tr_bboxes.append(active[tid]["bbox"])         # type: ignore

        # Compute matching cost matrix
        matches: List[Tuple[int, int]] = []
        if len(track_ids) > 0:
            cost = np.full((len(track_ids), len(frame)), 1e9, dtype=float)

            for i, tid in enumerate(track_ids):
                tc = tr_centroids[i]  # (x,y)
                tb = tr_bboxes[i]     # (x0,y0,x1,y1)
                for j in range(len(frame)):
                    dc = det_centroids[j]
                    db = det_bboxes[j]

                    d = dist(tc, dc)
                    iou = bbox_iou(tb, db)

                    dist_ok = d <= params.max_dist_px
                    iou_ok = iou >= params.min_iou
                    ok = (dist_ok or iou_ok) if params.allow_match_if_either else (dist_ok and iou_ok)

                    if not ok:
                        continue

                    # cost: centroid distance + penalty for low IoU
                    cost[i, j] = d + params.iou_weight * (1.0 - iou)

            # Solve assignment
            hung = try_hungarian(cost)
            pairs = hung if hung is not None else greedy_assignment(cost)

            # keep only finite, feasible pairs
            for i, j in pairs:
                if cost[i, j] >= 1e8:
                    continue
                matches.append((i, j))

        matched_tracks = set()
        matched_dets = set()

        # Apply matches: update existing tracks
        for i, j in matches:
            tid = track_ids[i]
            matched_tracks.add(tid)
            matched_dets.add(j)

            frame.loc[j, "track_id"] = tid

            # update active state
            active[tid] = {
                "centroid": det_centroids[j],
                "bbox": det_bboxes[j],
                "last_time": t,
            }

        # Create new tracks for unmatched detections
        for j in range(len(frame)):
            if j in matched_dets:
                continue
            tid = next_track_id
            next_track_id += 1
            frame.loc[j, "track_id"] = tid
            active[tid] = {
                "centroid": det_centroids[j],
                "bbox": det_bboxes[j],
                "last_time": t,
            }

        # Optional: prune tracks not seen recently (simple)
        # Since we process every timestamp in order, any unmatched track can be kept,
        # but for cleanliness we can drop tracks that weren't matched in this frame.
        drop = [tid for tid in list(active.keys()) if active[tid]["last_time"] != t]  # type: ignore
        for tid in drop:
            # keep them if you want persistence; for now, end tracks when not matched
            del active[tid]

        # Store output rows
        frame["track_id"] = frame["track_id"].astype(int)
        rows_out.append(frame)

    out = pd.concat(rows_out, ignore_index=True) if rows_out else df.assign(track_id=np.nan)
    out = out.sort_values(["track_id", "timestamp_utc", "object_rank"]).reset_index(drop=True)
    return out


def draw_tracks_overlay(df_tracks: pd.DataFrame, out_png: Path) -> Optional[str]:
    # Find last frame with an image
    last_rel = None
    last_ts = None
    for ts in reversed(sorted(df_tracks["timestamp_utc"].unique())):
        sub = df_tracks[df_tracks["timestamp_utc"] == ts]
        for rp in sub["relpath"].tolist():
            img = load_rgb(rp)
            if img is not None:
                last_rel = rp
                last_ts = ts
                base = img
                break
        if last_rel is not None:
            break

    if last_rel is None:
        return None

    base_rgba = base.convert("RGBA")
    d = ImageDraw.Draw(base_rgba, "RGBA")

    # collect track polylines
    sub_all = df_tracks.copy()
    # only points that exist on the base canvas size (sanity)
    W, H = base.size

    def clamp(p):
        x, y = float(p[0]), float(p[1])
        return (max(0, min(W - 1, x)), max(0, min(H - 1, y)))

    # palette
    colors = [
        (255, 60, 60, 220),
        (60, 255, 120, 220),
        (80, 170, 255, 220),
        (255, 200, 60, 220),
        (200, 80, 255, 220),
        (80, 255, 240, 220),
    ]

    for k, tid in enumerate(sorted(sub_all["track_id"].unique())):
        tdf = sub_all[sub_all["track_id"] == tid].sort_values("timestamp_utc")
        pts = [clamp((x, y)) for x, y in zip(tdf["centroid_x"].tolist(), tdf["centroid_y"].tolist())]
        if len(pts) < 2:
            continue
        col = colors[k % len(colors)]
        d.line(pts, fill=col, width=4)
        # mark last point
        lx, ly = pts[-1]
        r = 6
        d.ellipse([lx - r, ly - r, lx + r, ly + r], fill=col, outline=(255, 255, 255, 240), width=2)
        d.text((lx + r + 2, ly + r + 2), f"T{int(tid)}", fill=(255, 255, 255, 240))

    base_rgba.convert("RGB").save(out_png)
    return f"{last_rel} @ {last_ts}"


def summarize_tracks(df_tracks: pd.DataFrame) -> pd.DataFrame:
    g = df_tracks.groupby("track_id", as_index=False).agg(
        n_frames=("timestamp_utc", "count"),
        t_start=("timestamp_utc", "min"),
        t_end=("timestamp_utc", "max"),
        mean_area_px=("area_px", "mean"),
        max_area_px=("area_px", "max"),
    )
    g = g.sort_values(["n_frames", "max_area_px"], ascending=[False, False]).reset_index(drop=True)
    return g


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects-csv", default=str(REPO_ROOT / "out" / "cloud_objects.csv"),
                    help="Per-object CSV from scripts/cloud_objects.py (default: out/cloud_objects.csv)")
    ap.add_argument("--outdir", default=str(REPO_ROOT / "out"), help="Base output directory (default: out/)")
    ap.add_argument("--top-k", type=int, default=5, help="Track only top-K largest objects per frame")
    ap.add_argument("--max-dist-px", type=float, default=180.0, help="Max centroid distance allowed for matching")
    ap.add_argument("--min-iou", type=float, default=0.05, help="Min IoU allowed for matching")
    ap.add_argument("--iou-weight", type=float, default=120.0, help="Cost penalty weight for low IoU")
    ap.add_argument("--require-both", action="store_true",
                    help="If set, require BOTH distance and IoU thresholds (more strict). Default is OR.")
    args = ap.parse_args()

    out_root = Path(args.outdir).resolve()
    out_viz = out_root / "viz"
    out_viz.mkdir(parents=True, exist_ok=True)

    df = load_objects_csv(Path(args.objects_csv).resolve())

    params = TrackParams(
        top_k=int(args.top_k),
        max_dist_px=float(args.max_dist_px),
        min_iou=float(args.min_iou),
        iou_weight=float(args.iou_weight),
        allow_match_if_either=not bool(args.require_both),
    )

    tracked = track_multi(df, params)
    out_tracks_csv = out_root / "storm_tracks_multi.csv"
    tracked.to_csv(out_tracks_csv, index=False)

    summary = summarize_tracks(tracked)
    out_summary_csv = out_root / "storm_tracks_summary.csv"
    summary.to_csv(out_summary_csv, index=False)

    out_overlay = out_viz / "storm_tracks_multi_overlay.png"
    base_info = draw_tracks_overlay(tracked, out_overlay)

    print(f"Read objects: {len(df)} rows")
    print(f"Wrote: {out_tracks_csv}")
    print(f"Wrote: {out_summary_csv}")
    if base_info is not None:
        print(f"Wrote: {out_overlay} (base: {base_info})")
    else:
        print("Overlay skipped: no RGB frames found on disk for relpath.")
    print(f"Params: top_k={params.top_k}, max_dist_px={params.max_dist_px}, min_iou={params.min_iou}, "
          f"iou_weight={params.iou_weight}, require_both={bool(args.require_both)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
