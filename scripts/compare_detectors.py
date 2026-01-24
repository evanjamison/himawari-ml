from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    return s.astype(str).str.replace("\\", "/", regex=False)


def _load_csv(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    return pd.read_csv(path)


def _prepare_base(df: pd.DataFrame) -> pd.DataFrame:
    # Keep only readable images if available
    if "img_ok" in df.columns:
        df = df[df["img_ok"] == 1].copy()

    # Normalize relpath if present
    if "relpath" in df.columns:
        df["relpath"] = _norm_relpath(df["relpath"])

    # Timestamp parsing (nice for reports)
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)

    return df


def _merge_labels(
    base: pd.DataFrame,
    z_labeled: pd.DataFrame | None,
    if_labeled: pd.DataFrame | None,
) -> pd.DataFrame:
    out = base.copy()

    # Default columns
    out["is_bad_z"] = False
    out["z_score"] = np.nan
    out["is_bad_if"] = False
    out["if_score"] = np.nan

    if z_labeled is not None and "relpath" in out.columns and "relpath" in z_labeled.columns:
        z = z_labeled.copy()
        z["relpath"] = _norm_relpath(z["relpath"])
        keep = [c for c in ["relpath", "is_bad_frame", "anomaly_z"] if c in z.columns]
        z = z[keep].copy()
        out = out.merge(z, on="relpath", how="left")
        if "is_bad_frame" in out.columns:
            out["is_bad_z"] = out["is_bad_frame"].fillna(False).astype(bool)
        if "anomaly_z" in out.columns:
            out["z_score"] = pd.to_numeric(out["anomaly_z"], errors="coerce")
        out = out.drop(columns=[c for c in ["is_bad_frame", "anomaly_z"] if c in out.columns])

    if if_labeled is not None and "relpath" in out.columns and "relpath" in if_labeled.columns:
        f = if_labeled.copy()
        f["relpath"] = _norm_relpath(f["relpath"])
        keep = [c for c in ["relpath", "is_bad_iforest", "iforest_score"] if c in f.columns]
        f = f[keep].copy()
        out = out.merge(f, on="relpath", how="left")
        if "is_bad_iforest" in out.columns:
            out["is_bad_if"] = out["is_bad_iforest"].fillna(False).astype(bool)
        if "iforest_score" in out.columns:
            out["if_score"] = pd.to_numeric(out["iforest_score"], errors="coerce")
        out = out.drop(columns=[c for c in ["is_bad_iforest", "iforest_score"] if c in out.columns])

    return out


def _compute_pca(df: pd.DataFrame, feat_cols: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing PCA feature columns: {missing}")

    X = df[feat_cols].copy()
    for c in feat_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    mask = X.notna().all(axis=1)
    dff = df.loc[mask].copy()
    X = X.loc[mask].to_numpy(dtype=float)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(Xs)
    return dff, Z


def _agreement_table(df: pd.DataFrame) -> pd.DataFrame:
    z = df["is_bad_z"].astype(bool)
    f = df["is_bad_if"].astype(bool)

    both = (z & f).sum()
    z_only = (z & ~f).sum()
    if_only = (~z & f).sum()
    neither = (~z & ~f).sum()

    tbl = pd.DataFrame(
        {
            "count": [int(both), int(z_only), int(if_only), int(neither)],
        },
        index=["both_bad", "z_only", "if_only", "neither"],
    )
    return tbl


def _write_disagreements(df: pd.DataFrame, out_csv: Path) -> None:
    z = df["is_bad_z"].astype(bool)
    f = df["is_bad_if"].astype(bool)
    disagree = df[z ^ f].copy()

    cols = []
    for c in ["timestamp_utc", "relpath", "filename", "mean_luma", "std_luma"]:
        if c in disagree.columns:
            cols.append(c)
    cols += ["is_bad_z", "z_score", "is_bad_if", "if_score"]

    disagree = disagree[cols].copy()

    # Helpful ordering: IF-only first (often more interesting), then Z-only
    disagree["disagree_type"] = np.where(
        disagree["is_bad_if"] & ~disagree["is_bad_z"], "if_only", "z_only"
    )
    disagree = disagree.sort_values(["disagree_type", "if_score", "z_score"], ascending=[True, False, False])

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    disagree.to_csv(out_csv, index=False)


def _plot_pca_two_markers(df: pd.DataFrame, Z: np.ndarray, out_png: Path, title: str) -> None:
    is_z = df["is_bad_z"].astype(bool).to_numpy()
    is_if = df["is_bad_if"].astype(bool).to_numpy()

    both = is_z & is_if
    z_only = is_z & ~is_if
    if_only = ~is_z & is_if
    normal = ~is_z & ~is_if

    plt.figure()

    # Base cloud (normal)
    plt.scatter(Z[normal, 0], Z[normal, 1], s=28, alpha=0.75, label=f"Normal ({int(normal.sum())})")

    # Two detectors, two markers
    if z_only.any():
        plt.scatter(Z[z_only, 0], Z[z_only, 1], s=140, marker="x", linewidths=2.8, label=f"Z-only ({int(z_only.sum())})")
    if if_only.any():
        plt.scatter(Z[if_only, 0], Z[if_only, 1], s=120, marker="^", alpha=0.95, label=f"IF-only ({int(if_only.sum())})")

    # Overlap marker on top
    if both.any():
        plt.scatter(Z[both, 0], Z[both, 1], s=220, marker="*", label=f"Both ({int(both.sum())})")

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base-csv",
        default=str(REPO_ROOT / "out" / "dataset.csv"),
        help="Base dataset CSV (features). Typically out/dataset.csv",
    )
    ap.add_argument(
        "--z-labeled-csv",
        default=str(REPO_ROOT / "out" / "viz" / "features_labeled.csv"),
        help="Z-score labeled CSV from viz_timeseries.py (contains is_bad_frame/anomaly_z)",
    )
    ap.add_argument(
        "--if-labeled-csv",
        default=str(REPO_ROOT / "out" / "viz" / "features_labeled_iforest.csv"),
        help="IsolationForest labeled CSV (contains is_bad_iforest/iforest_score)",
    )
    ap.add_argument(
        "--outdir",
        default=str(REPO_ROOT / "out" / "viz"),
        help="Output directory",
    )
    ap.add_argument(
        "--features",
        default=",".join(DEFAULT_FEATURES),
        help="Comma-separated features for PCA",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()
    base = _prepare_base(_load_csv(Path(args.base_csv).resolve(), "base_csv"))

    z_df = None
    if args.z_labeled_csv:
        zp = Path(args.z_labeled_csv).resolve()
        if zp.exists():
            z_df = _prepare_base(_load_csv(zp, "z_labeled_csv"))

    if_df = None
    if args.if_labeled_csv:
        ip = Path(args.if_labeled_csv).resolve()
        if ip.exists():
            if_df = _prepare_base(_load_csv(ip, "if_labeled_csv"))

    df = _merge_labels(base, z_df, if_df)

    # Agreement table + disagreements CSV
    agree = _agreement_table(df)
    disagree_csv = outdir / "detector_disagreements.csv"
    _write_disagreements(df, disagree_csv)

    # PCA plot with two markers
    feat_cols = [c.strip() for c in args.features.split(",") if c.strip()]
    df_pca, Z = _compute_pca(df, feat_cols)

    pca_png = outdir / "detector_compare_pca.png"
    _plot_pca_two_markers(
        df_pca,
        Z,
        pca_png,
        title="Detector comparison on PCA (Z-score vs IsolationForest)",
    )

    # Print a readable summary
    print("Agreement counts:")
    print(agree.to_string())
    print()
    print(f"Wrote: {disagree_csv}")
    print(f"Wrote: {pca_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
