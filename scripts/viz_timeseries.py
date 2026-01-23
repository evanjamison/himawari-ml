from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_features_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Features CSV not found: {path}")
    df = pd.read_csv(path)

    # Require these columns to do time series
    needed = ["timestamp_utc", "mean_luma"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in features CSV: {missing}")

    # Keep only parsed timestamps if available
    if "parsed_ok" in df.columns:
        df = df[df["parsed_ok"] == 1].copy()

    # Keep only readable images if available
    if "img_ok" in df.columns:
        df = df[df["img_ok"] == 1].copy()

    # Parse timestamp + numeric luma
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
    df["mean_luma"] = pd.to_numeric(df["mean_luma"], errors="coerce")

    df = df.dropna(subset=["timestamp_utc", "mean_luma"]).copy()
    df = df.sort_values("timestamp_utc")
    return df


def maybe_merge_clusters(df: pd.DataFrame, pred_csv: Path | None) -> pd.DataFrame:
    if pred_csv is None:
        return df

    if not pred_csv.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {pred_csv}")

    pred = pd.read_csv(pred_csv)

    if "relpath" not in df.columns or "relpath" not in pred.columns:
        raise ValueError("Both features CSV and predictions CSV must have a 'relpath' column to merge.")

    if "cluster" not in pred.columns:
        raise ValueError("Predictions CSV must include a 'cluster' column.")

    pred["cluster"] = pd.to_numeric(pred["cluster"], errors="coerce").astype("Int64")
    out = df.merge(pred[["relpath", "cluster"]], on="relpath", how="left")
    return out


def plot_timeseries(df: pd.DataFrame, out_png: Path, title: str = "Mean luma over time") -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(df["timestamp_utc"], df["mean_luma"])
    plt.title(title)
    plt.xlabel("Timestamp (UTC)")
    plt.ylabel("mean_luma")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def plot_timeseries_with_clusters(df: pd.DataFrame, out_png: Path, title: str = "Mean luma over time (colored by cluster)") -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()

    # Always show the line for context
    plt.plot(df["timestamp_utc"], df["mean_luma"])

    # Overlay cluster-colored points if cluster exists
    if "cluster" in df.columns and df["cluster"].notna().any():
        plt.scatter(
            df["timestamp_utc"],
            df["mean_luma"],
            c=df["cluster"].fillna(-1),
            s=35,
        )

    plt.title(title)
    plt.xlabel("Timestamp (UTC)")
    plt.ylabel("mean_luma")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--features-csv",
        required=True,
        help="Dataset CSV produced by build_dataset.py (must include timestamp_utc + mean_luma)",
    )
    ap.add_argument(
        "--pred-csv",
        default="",
        help="Optional predictions CSV produced by train_baseline.py --save-predictions (adds cluster coloring)",
    )
    ap.add_argument(
        "--outdir",
        default=str(REPO_ROOT / "out" / "viz"),
        help="Output directory for plots",
    )
    ap.add_argument(
        "--max-points",
        type=int,
        default=0,
        help="If >0, downsample to at most this many points (uniformly) for readability",
    )
    args = ap.parse_args()

    features_csv = Path(args.features_csv).resolve()
    pred_csv = Path(args.pred_csv).resolve() if args.pred_csv.strip() else None
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_features_csv(features_csv)
    if len(df) == 0:
        raise SystemExit("No usable rows after filtering. Need parsed_ok=1, img_ok=1, valid timestamp_utc, mean_luma.")

    # Optional downsample (uniform)
    if args.max_points and args.max_points > 0 and len(df) > args.max_points:
        step = max(1, len(df) // args.max_points)
        df = df.iloc[::step].copy()

    # Base plot (no clusters)
    base_png = outdir / "luma_timeseries.png"
    plot_timeseries(df, base_png)

    # Optional cluster overlay
    if pred_csv is not None:
        df2 = maybe_merge_clusters(df, pred_csv)
        cluster_png = outdir / "luma_timeseries_clusters.png"
        plot_timeseries_with_clusters(df2, cluster_png)

        print(f"Wrote: {base_png}")
        print(f"Wrote: {cluster_png}")
    else:
        print(f"Wrote: {base_png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
