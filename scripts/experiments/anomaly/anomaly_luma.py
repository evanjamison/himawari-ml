from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-csv", required=True, help="CSV from build_dataset (must include timestamp_utc, mean_luma)")
    ap.add_argument("--outdir", default="out/viz", help="Output directory")
    ap.add_argument("--z", type=float, default=3.0, help="Z-score threshold for anomalies")
    args = ap.parse_args()

    features_csv = Path(args.features_csv).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(features_csv)

    # Filter + parse
    if "parsed_ok" in df.columns:
        df = df[df["parsed_ok"] == 1].copy()
    if "img_ok" in df.columns:
        df = df[df["img_ok"] == 1].copy()

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
    df["mean_luma"] = pd.to_numeric(df["mean_luma"], errors="coerce")
    df = df.dropna(subset=["timestamp_utc", "mean_luma"]).sort_values("timestamp_utc").reset_index(drop=True)

    if len(df) < 3:
        raise SystemExit("Need at least 3 points for anomaly detection.")

    # ---- 5-line anomaly detector (delta luma z-score) ----
    mu = df["mean_luma"].mean()
    sd = df["mean_luma"].std() or 1.0
    df["z_luma"] = (df["mean_luma"] - mu) / sd
    df["is_anomaly"] = df["z_luma"].abs() >= float(args.z)

    # ------------------------------------------------------

    # Plot
    plt.figure()
    plt.plot(df["timestamp_utc"], df["mean_luma"])
    anom = df[df["is_anomaly"]].copy()
    if len(anom):
        plt.scatter(anom["timestamp_utc"], anom["mean_luma"], s=70)

    plt.title(f"Mean luma + anomalies (|z(Δluma)| ≥ {args.z:g})")
    plt.xlabel("Timestamp (UTC)")
    plt.ylabel("mean_luma")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    out_png = outdir / "luma_anomalies.png"
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

    # Save a small anomalies report
    out_csv = outdir / "luma_anomalies.csv"
    cols = [c for c in ["timestamp_utc", "filename", "relpath", "mean_luma", "delta_luma", "z_delta", "is_anomaly"] if c in df.columns]
    df.loc[df["is_anomaly"], cols].to_csv(out_csv, index=False)

    print(f"Wrote: {out_png}")
    print(f"Wrote: {out_csv} (rows={len(anom)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
