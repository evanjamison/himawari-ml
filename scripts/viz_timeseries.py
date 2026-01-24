from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
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
    df = df.reset_index(drop=True)
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

    # Normalize relpath slashes to be safe on Windows
    df2 = df.copy()
    df2["relpath"] = df2["relpath"].astype(str).str.replace("\\", "/", regex=False)
    pred["relpath"] = pred["relpath"].astype(str).str.replace("\\", "/", regex=False)

    out = df2.merge(pred[["relpath", "cluster"]], on="relpath", how="left")
    return out


def add_anomaly_labels(
    df: pd.DataFrame,
    z_thresh: float = 2.5,
    window: int = 25,
    use_diff: bool = True,
) -> pd.DataFrame:
    """
    Adds:
      - anomaly_signal: the raw signal we score
          * diff(mean_luma) if use_diff=True  (change anomalies)
          * mean_luma       if use_diff=False (level anomalies)
      - anomaly_z: z-score of that signal
          * rolling z-score for diff mode
          * global z-score for level mode (more reliable for one-off black/corrupt frames)
      - is_bad_frame: bool flag where |z| >= z_thresh

    This is intentionally simple + deterministic.
    """
    out = df.copy()

    # 1) define the signal
    if use_diff:
        sig = out["mean_luma"].diff()
    else:
        sig = out["mean_luma"].copy()

    out["anomaly_signal"] = pd.to_numeric(sig, errors="coerce")

    # 2) compute z-score
    if use_diff:
        # Rolling z-score for local "jumps"
        r = out["anomaly_signal"].rolling(window=window, min_periods=max(5, window // 3))
        mu = r.mean()
        sd = r.std(ddof=0)
    else:
        # ✅ GLOBAL z-score for absolute-level outliers (e.g., near-black frame)
        mu = out["anomaly_signal"].mean()
        sd = out["anomaly_signal"].std(ddof=0)

    # avoid division by zero
    if isinstance(sd, pd.Series):
        sd = sd.replace(0, np.nan)
    else:
        sd = np.nan if sd == 0 else sd

    z = (out["anomaly_signal"] - mu) / sd
    out["anomaly_z"] = z

    # 3) flag anomalies (NaNs -> False)
    out["is_bad_frame"] = out["anomaly_z"].abs().ge(z_thresh).fillna(False).astype(bool)
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


def plot_timeseries_with_anomalies(
    df: pd.DataFrame,
    out_png: Path,
    title: str = "Mean luma over time (anomalies overlaid)",
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(df["timestamp_utc"], df["mean_luma"], label="Mean luma")

    if "is_bad_frame" in df.columns:
        is_bad = df["is_bad_frame"].fillna(False).astype(bool)
        if is_bad.any():
            plt.scatter(
                df.loc[is_bad, "timestamp_utc"],
                df.loc[is_bad, "mean_luma"],
                s=70,
                marker="x",
                linewidths=2,
                label=f"Anomaly ({int(is_bad.sum())})",
            )

    plt.title(title)
    plt.xlabel("Timestamp (UTC)")
    plt.ylabel("mean_luma")
    plt.xticks(rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def plot_timeseries_with_clusters(
    df: pd.DataFrame,
    out_png: Path,
    title: str = "Mean luma over time (colored by cluster)",
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(df["timestamp_utc"], df["mean_luma"])

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
        help="Output directory for plots + labeled CSV",
    )
    ap.add_argument(
        "--max-points",
        type=int,
        default=0,
        help="If >0, downsample to at most this many points (uniformly) for readability",
    )

    # anomaly labeling controls
    ap.add_argument(
        "--z",
        type=float,
        default=2.5,
        help="Z threshold for is_bad_frame (|z| >= z). Example: 2.0 or 2.5",
    )
    ap.add_argument(
        "--window",
        type=int,
        default=25,
        help="Rolling window length for z-score on the anomaly signal (diff mode)",
    )
    ap.add_argument(
        "--no-diff",
        action="store_true",
        help="If set, score anomalies on raw mean_luma (level anomalies). Uses GLOBAL z-score.",
    )
    ap.add_argument(
        "--save-labeled-csv",
        action="store_true",
        help="Write a CSV alongside plots that includes is_bad_frame + anomaly_z",
    )
    args = ap.parse_args()

    features_csv = Path(args.features_csv).resolve()
    pred_csv = Path(args.pred_csv).resolve() if args.pred_csv.strip() else None
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_features_csv(features_csv)
    if len(df) == 0:
        raise SystemExit("No usable rows after filtering. Need parsed_ok=1, img_ok=1, valid timestamp_utc, mean_luma.")

    if args.max_points and args.max_points > 0 and len(df) > args.max_points:
        step = max(1, len(df) // args.max_points)
        df = df.iloc[::step].copy()
        df = df.reset_index(drop=True)

    # Add anomaly labels
    df = add_anomaly_labels(
        df,
        z_thresh=float(args.z),
        window=int(args.window),
        use_diff=not bool(args.no_diff),
    )

    base_png = outdir / "luma_timeseries.png"
    plot_timeseries(df, base_png)

    anom_png = outdir / "luma_timeseries_anoms.png"
    plot_timeseries_with_anomalies(df, anom_png)

    if pred_csv is not None:
        df2 = maybe_merge_clusters(df, pred_csv)
        cluster_png = outdir / "luma_timeseries_clusters.png"
        plot_timeseries_with_clusters(df2, cluster_png)
        print(f"Wrote: {cluster_png}")

    if args.save_labeled_csv:
        labeled_csv = outdir / "features_labeled.csv"
        df.to_csv(labeled_csv, index=False)
        print(f"Wrote: {labeled_csv} (is_bad_frame={int(df['is_bad_frame'].sum())})")

    print(f"Wrote: {base_png}")
    print(f"Wrote: {anom_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
