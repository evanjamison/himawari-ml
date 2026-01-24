from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_FEATURES = [
    "mean_r",
    "mean_g",
    "mean_b",
    "mean_luma",
    "std_luma",
    "aspect_ratio",
    "width",
    "height",
]


def _norm_relpath(s: pd.Series) -> pd.Series:
    # Normalize slashes so merges work on Windows.
    return s.astype(str).str.replace("\\", "/", regex=False)


def _load_and_filter(pred_csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(pred_csv_path)

    # Some pipelines save img_ok; keep only readable if present
    if "img_ok" in df.columns:
        df = df[df["img_ok"] == 1].copy()

    # This script expects predictions.csv produced by train_baseline --save-predictions
    if "cluster" not in df.columns:
        raise ValueError("CSV must include a 'cluster' column. Run train_baseline with --save-predictions.")

    if "relpath" not in df.columns:
        raise ValueError("CSV must include a 'relpath' column to link rows to images (and to anomaly labels).")

    df["relpath"] = _norm_relpath(df["relpath"])
    return df


def _merge_anomaly_labels(df: pd.DataFrame, labeled_csv: Path) -> pd.DataFrame:
    """
    Merge is_bad_frame (+ anomaly_z if available) from viz_timeseries output.
    Safe behavior: if labeled csv missing or incompatible, just add is_bad_frame=False.
    """
    out = df.copy()

    out["is_bad_frame"] = False
    out["anomaly_z"] = pd.NA

    if not labeled_csv or not labeled_csv.exists():
        return out

    labeled = pd.read_csv(labeled_csv)
    if "relpath" not in labeled.columns:
        return out

    labeled["relpath"] = _norm_relpath(labeled["relpath"])

    keep = [c for c in ["relpath", "is_bad_frame", "anomaly_z"] if c in labeled.columns]
    labeled = labeled[keep].copy()

    merged = out.merge(labeled, on="relpath", how="left", suffixes=("", "_lbl"))

    if "is_bad_frame_lbl" in merged.columns:
        merged["is_bad_frame"] = merged["is_bad_frame_lbl"]
        merged = merged.drop(columns=["is_bad_frame_lbl"])

    if "anomaly_z_lbl" in merged.columns:
        merged["anomaly_z"] = merged["anomaly_z_lbl"]
        merged = merged.drop(columns=["anomaly_z_lbl"])

    merged["is_bad_frame"] = merged["is_bad_frame"].fillna(False).astype(bool)
    return merged


def _prepare_X(df: pd.DataFrame, features: list[str]):
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = df[features].copy()
    for c in features:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    mask = X.notna().all(axis=1)
    df = df.loc[mask].copy()
    X = X.loc[mask].copy()
    return df, X


def plot_pca_scatter(df: pd.DataFrame, X: pd.DataFrame, out_path: Path):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(Xs)

    # Ensure boolean flag exists
    if "is_bad_frame" not in df.columns:
        df["is_bad_frame"] = False
    is_bad = df["is_bad_frame"].fillna(False).astype(bool).to_numpy()
    is_good = ~is_bad

    # clusters as ints for coloring
    clusters = pd.to_numeric(df["cluster"], errors="coerce").fillna(-1).astype(int).to_numpy()

    plt.figure()

    # Normal points first (colored by cluster)
    plt.scatter(
        Z[is_good, 0],
        Z[is_good, 1],
        c=clusters[is_good],
        s=30,
        alpha=0.85,
        label="Normal",
    )

    # Anomalies on top as red X
    if is_bad.any():
        plt.scatter(
            Z[is_bad, 0],
            Z[is_bad, 1],
            s=120,
            marker="x",
            linewidths=2.5,
            color="red",
            label=f"Anomaly ({int(is_bad.sum())})",
        )

    plt.title("Baseline clusters (PCA projection) + anomaly overlay")
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def make_thumbnail_grid(
    df: pd.DataFrame,
    images_root: Path,
    out_dir: Path,
    max_per_cluster: int = 12,
    thumb_px: int = 128,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    for cluster_id, g in df.groupby("cluster"):
        g = g.head(max_per_cluster)

        thumbs = []
        for rel in g["relpath"].astype(str):
            img_path = images_root / rel
            try:
                im = Image.open(img_path).convert("RGB")
                im.thumbnail((thumb_px, thumb_px))
                thumbs.append(im)
            except Exception:
                continue

        if not thumbs:
            continue

        cols = min(6, len(thumbs))
        rows = (len(thumbs) + cols - 1) // cols

        grid = Image.new("RGB", (cols * thumb_px, rows * thumb_px), (255, 255, 255))
        for i, im in enumerate(thumbs):
            r, c = divmod(i, cols)
            x = c * thumb_px + (thumb_px - im.width) // 2
            y = r * thumb_px + (thumb_px - im.height) // 2
            grid.paste(im, (x, y))

        out_path = out_dir / f"cluster_{int(cluster_id)}_grid.png"
        grid.save(out_path)


def make_anomaly_grid(
    df: pd.DataFrame,
    images_root: Path,
    out_path: Path,
    max_images: int = 24,
    thumb_px: int = 128,
):
    """Optional: one grid of anomalies only."""
    if "is_bad_frame" not in df.columns:
        return
    g = df[df["is_bad_frame"]].copy()
    if g.empty:
        return

    g = g.head(max_images)
    thumbs = []
    for rel in g["relpath"].astype(str):
        img_path = images_root / rel
        try:
            im = Image.open(img_path).convert("RGB")
            im.thumbnail((thumb_px, thumb_px))
            thumbs.append(im)
        except Exception:
            continue

    if not thumbs:
        return

    cols = min(6, len(thumbs))
    rows = (len(thumbs) + cols - 1) // cols
    grid = Image.new("RGB", (cols * thumb_px, rows * thumb_px), (255, 255, 255))

    for i, im in enumerate(thumbs):
        r, c = divmod(i, cols)
        x = c * thumb_px + (thumb_px - im.width) // 2
        y = r * thumb_px + (thumb_px - im.height) // 2
        grid.paste(im, (x, y))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-csv", required=True, help="Predictions CSV produced by train_baseline --save-predictions")
    ap.add_argument("--labeled-csv", default=str(REPO_ROOT / "out" / "viz" / "features_labeled.csv"),
                    help="CSV from viz_timeseries.py with is_bad_frame (merged by relpath)")
    ap.add_argument("--images-root", default=str(REPO_ROOT), help="Root to resolve relpath (default: repo root)")
    ap.add_argument("--outdir", default=str(REPO_ROOT / "out" / "viz"), help="Output directory")
    ap.add_argument("--features", default=",".join(DEFAULT_FEATURES), help="Comma-separated feature columns")
    ap.add_argument("--max-per-cluster", type=int, default=12, help="Max thumbnails per cluster")
    ap.add_argument("--thumb-px", type=int, default=128, help="Thumbnail size in px")
    args = ap.parse_args()

    pred_csv = Path(args.pred_csv).resolve()
    labeled_csv = Path(args.labeled_csv).resolve() if args.labeled_csv else None
    images_root = Path(args.images_root).resolve()
    outdir = Path(args.outdir).resolve()
    features = [c.strip() for c in args.features.split(",") if c.strip()]

    df = _load_and_filter(pred_csv)
    df = _merge_anomaly_labels(df, labeled_csv) if labeled_csv else df
    df, X = _prepare_X(df, features)

    # PCA scatter (with anomaly overlay)
    scatter_path = outdir / "clusters_pca_scatter.png"
    plot_pca_scatter(df, X, scatter_path)

    # Thumbnail grids per cluster
    make_thumbnail_grid(df, images_root, outdir, max_per_cluster=args.max_per_cluster, thumb_px=args.thumb_px)

    # Optional anomalies-only grid
    anomaly_grid_path = outdir / "anomalies_grid.png"
    make_anomaly_grid(df, images_root, anomaly_grid_path, thumb_px=args.thumb_px)

    # Prints
    n_bad = int(df["is_bad_frame"].sum()) if "is_bad_frame" in df.columns else 0
    print(f"Wrote: {scatter_path} (anomalies={n_bad})")
    if n_bad > 0 and anomaly_grid_path.exists():
        print(f"Wrote: {anomaly_grid_path}")
    print(f"Wrote thumbnail grids to: {outdir}")


if __name__ == "__main__":
    main()

