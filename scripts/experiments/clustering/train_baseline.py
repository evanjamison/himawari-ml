# scripts/train_baseline.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_IN = REPO_ROOT / "data" / "derived" / "samples.csv"
DEFAULT_OUTDIR = REPO_ROOT / "out" / "models"

FEATURES_DEFAULT = [
    "mean_r",
    "mean_g",
    "mean_b",
    "mean_luma",
    "std_luma",
    "aspect_ratio",
    "width",
    "height",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    return pd.read_csv(path)


def select_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in dataset: {missing}")

    # Use only readable images if that column exists
    if "img_ok" in df.columns:
        df = df[df["img_ok"] == 1].copy()

    X = df[features].copy()
    # Coerce to numeric and drop rows with any NaN in features
    for c in features:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    mask = X.notna().all(axis=1)
    df = df.loc[mask].copy()
    X = X.loc[mask].copy()
    return df, X


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default=str(DEFAULT_IN), help="Input CSV path")
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR), help="Output directory for artifacts")
    ap.add_argument("--k", type=int, default=3, help="Number of clusters (KMeans)")
    ap.add_argument(
        "--features",
        default=",".join(FEATURES_DEFAULT),
        help="Comma-separated feature columns",
    )
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--save-predictions", action="store_true", help="Write a CSV with cluster assignments")
    args = ap.parse_args()

    in_path = Path(args.in_path).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    features = [c.strip() for c in args.features.split(",") if c.strip()]
    if args.k < 2:
        print("Error: --k must be >= 2", file=sys.stderr)
        return 2

    try:
        df = load_dataset(in_path)
        df_sel, X = select_features(df, features)
    except Exception as e:
        print(f"Training failed during data prep: {e}", file=sys.stderr)
        return 2

    if len(X) == 0:
        print(
            "Training failed: no usable rows after filtering/NaN removal. "
            "Check img_ok and feature extraction.",
            file=sys.stderr,
        )
        return 2

    # Standardize features for KMeans stability
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=args.k, random_state=args.seed, n_init="auto")
    model.fit(X_scaled)

    labels = model.labels_.tolist()

    # Save artifacts
    stamp = _utc_now_iso().replace(":", "").replace("-", "")
    model_path = outdir / f"baseline_kmeans_k{args.k}_{stamp}.joblib"
    metrics_path = outdir / f"baseline_kmeans_k{args.k}_{stamp}.metrics.json"

    joblib.dump({"scaler": scaler, "model": model, "features": features}, model_path)

    metrics = {
        "timestamp_utc": _utc_now_iso(),
        "input_csv": str(in_path),
        "rows_total": int(len(df)),
        "rows_used": int(len(df_sel)),
        "k": int(args.k),
        "seed": int(args.seed),
        "features": features,
        "inertia": float(model.inertia_),
        "model_path": str(model_path),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if args.save_predictions:
        pred_path = outdir / f"baseline_kmeans_k{args.k}_{stamp}.predictions.csv"
        out_df = df_sel.copy()
        out_df["cluster"] = labels
        out_df.to_csv(pred_path, index=False)
        metrics["predictions_path"] = str(pred_path)

    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
