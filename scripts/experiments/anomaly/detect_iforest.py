from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]

# Good default set (matches your project’s feature style)
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
    return s.astype(str).str.replace("\\", "/", regex=False)


def load_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    df = pd.read_csv(path)

    # Normalize relpath if present (helps merges later)
    if "relpath" in df.columns:
        df["relpath"] = _norm_relpath(df["relpath"])

    # Optional: keep only readable images if available
    if "img_ok" in df.columns:
        df = df[df["img_ok"] == 1].copy()

    return df


def pick_features(df: pd.DataFrame, features: list[str]) -> list[str]:
    # If user passed explicit features, validate them
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing requested feature columns: {missing}")
    return features


def build_X(df: pd.DataFrame, feat_cols: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    X = df[feat_cols].copy()
    for c in feat_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # Drop rows with any missing feature
    mask = X.notna().all(axis=1)
    df2 = df.loc[mask].copy()
    X2 = X.loc[mask].to_numpy(dtype=float)

    if len(df2) == 0:
        raise ValueError("No rows left after dropping NaNs in features. Check your dataset/features.")

    return df2, X2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in-csv",
        default=str(REPO_ROOT / "out" / "dataset.csv"),
        help="Input feature dataset CSV (e.g., out/dataset.csv or out/viz/features_labeled.csv)",
    )
    ap.add_argument(
        "--out-csv",
        default=str(REPO_ROOT / "out" / "viz" / "features_labeled_iforest.csv"),
        help="Output CSV with is_bad_iforest + iforest_score (and any existing columns preserved)",
    )
    ap.add_argument(
        "--features",
        default=",".join(DEFAULT_FEATURES),
        help="Comma-separated feature columns to use for IsolationForest",
    )
    ap.add_argument(
        "--contamination",
        type=float,
        default=0.05,
        help="Expected anomaly fraction (e.g., 0.01–0.10). Higher flags more anomalies.",
    )
    ap.add_argument(
        "--n-estimators",
        type=int,
        default=300,
        help="Number of trees (more = stabler, slower)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    ap.add_argument(
        "--no-scale",
        action="store_true",
        help="Disable StandardScaler (not recommended unless all features already comparable)",
    )
    args = ap.parse_args()

    in_path = Path(args.in_csv).resolve()
    out_path = Path(args.out_csv).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_df(in_path)

    feat_cols = [c.strip() for c in args.features.split(",") if c.strip()]
    feat_cols = pick_features(df, feat_cols)

    df_fit, X = build_X(df, feat_cols)

    # Scale features (recommended for mixed units like width/height vs luma)
    if not args.no_scale:
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
    else:
        Xs = X

    model = IsolationForest(
        n_estimators=args.n_estimators,
        contamination=args.contamination,
        random_state=args.seed,
        n_jobs=-1,
    )
    model.fit(Xs)

    # sklearn convention: predict = 1 normal, -1 anomaly
    pred = model.predict(Xs)

    # decision_function: higher is more normal
    # invert so "higher = more anomalous" is intuitive
    normal_score = model.decision_function(Xs)
    iforest_score = -normal_score

    # Write results back into the original df (align by index)
    out_df = df.copy()
    out_df["is_bad_iforest"] = False
    out_df["iforest_score"] = np.nan

    out_df.loc[df_fit.index, "is_bad_iforest"] = (pred == -1)
    out_df.loc[df_fit.index, "iforest_score"] = iforest_score

    out_df["is_bad_iforest"] = out_df["is_bad_iforest"].fillna(False).astype(bool)

    out_df.to_csv(out_path, index=False)

    n_bad = int(out_df["is_bad_iforest"].sum())
    print(f"Wrote: {out_path} (is_bad_iforest={n_bad}, contamination={args.contamination})")
    print(f"Features used: {feat_cols}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
