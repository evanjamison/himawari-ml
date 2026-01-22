from __future__ import annotations

import argparse
from pathlib import Path

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


def _load_and_filter(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "img_ok" in df.columns:
        df = df[df["img_ok"] == 1].copy()
    if "cluster" not in df.columns:
        raise ValueError("CSV must include a 'cluster' column. Run train_baseline with --save-predictions.")
    return df


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


def plot_pca_scatter(df: pd.DataFrame, X, out_path: Path):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(Xs)

    plt.figure()
    plt.scatter(Z[:, 0], Z[:, 1], c=df["cluster"].astype(int), s=30)
    plt.title("Baseline clusters (PCA projection)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def make_thumbnail_grid(df: pd.DataFrame, images_root: Path, out_dir: Path, max_per_cluster: int = 12, thumb_px: int = 128):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-csv", required=True, help="Predictions CSV produced by train_baseline --save-predictions")
    ap.add_argument("--images-root", default=str(REPO_ROOT), help="Root to resolve relpath (default: repo root)")
    ap.add_argument("--outdir", default=str(REPO_ROOT / "out" / "viz"), help="Output directory")
    ap.add_argument("--features", default=",".join(DEFAULT_FEATURES), help="Comma-separated feature columns")
    ap.add_argument("--max-per-cluster", type=int, default=12, help="Max thumbnails per cluster")
    args = ap.parse_args()

    pred_csv = Path(args.pred_csv).resolve()
    images_root = Path(args.images_root).resolve()
    outdir = Path(args.outdir).resolve()
    features = [c.strip() for c in args.features.split(",") if c.strip()]

    df = _load_and_filter(pred_csv)
    df, X = _prepare_X(df, features)

    # Scatter plot
    plot_pca_scatter(df, X, outdir / "clusters_pca_scatter.png")

    # Thumbnail grids
    make_thumbnail_grid(df, images_root, outdir, max_per_cluster=args.max_per_cluster)

    print(f"Wrote: {outdir / 'clusters_pca_scatter.png'}")
    print(f"Wrote thumbnail grids to: {outdir}")


if __name__ == "__main__":
    main()
