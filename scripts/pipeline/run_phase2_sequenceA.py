from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(cmd: list[str]) -> None:
    print("\n" + "=" * 88)
    print(" ".join(str(c) for c in cmd))
    print("=" * 88)
    subprocess.check_call(cmd)


def _dataset_slug_from_raw_dir(raw_dir: Path) -> str:
    """
    Create a stable output subfolder name from a raw_dir.
    Example: data/sample/raw/sequenceA -> sequenceA
    """
    # Prefer last folder name (most intuitive)
    name = raw_dir.name.strip()
    return name if name else "dataset"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Phase 2 runner (baseline + anomaly + clustering + viz) for ANY image dataset folder."
    )

    # --- INPUTS ---
    ap.add_argument(
        "--raw-dir",
        default=None,
        help=(
            "Directory containing raw frames (png/jpg). "
            "Example: data/sample/raw/sequenceA or any folder you have."
        ),
    )
    ap.add_argument(
        "--metrics-csv",
        default=None,
        help=(
            "Optional: path to an existing cloud_metrics.csv. "
            "If provided, we skip baseline generation unless --ensure-baseline is also set."
        ),
    )

    # --- OUTPUTS ---
    ap.add_argument(
        "--outdir",
        default=None,
        help=(
            "Output directory. If omitted, defaults to out/<dataset_name>/ when --raw-dir is given, "
            "or out/ when only --metrics-csv is provided."
        ),
    )

    # --- CONTROL FLAGS ---
    ap.add_argument(
        "--ensure-baseline",
        action="store_true",
        help="If set, runs the baseline pipeline first (mask -> objects -> tracking) to produce cloud_metrics.csv.",
    )

    # --- PARAMS ---
    ap.add_argument(
        "--features",
        default="mean_luma,cloud_fraction",
        help="Comma-separated features to use for IF/KMeans/Viz (must exist in cloud_metrics.csv).",
    )
    ap.add_argument("--k", type=int, default=3, help="KMeans K for baseline clustering.")
    ap.add_argument("--z", type=float, default=3.0, help="Z threshold for anomaly_luma.")
    ap.add_argument("--contamination", type=float, default=0.05, help="IsolationForest contamination.")

    args = ap.parse_args()

    py = sys.executable

    # -----------------------------
    # Resolve raw_dir / metrics_csv
    # -----------------------------
    raw_dir: Path | None = Path(args.raw_dir).resolve() if args.raw_dir else None
    metrics_csv: Path | None = Path(args.metrics_csv).resolve() if args.metrics_csv else None

    if raw_dir is None and metrics_csv is None:
        raise SystemExit(
            "ERROR: You must provide at least one of:\n"
            "  --raw-dir <folder_of_images>\n"
            "  --metrics-csv <path_to_cloud_metrics.csv>\n"
        )

    if raw_dir is not None and not raw_dir.exists():
        raise FileNotFoundError(f"--raw-dir not found: {raw_dir}")

    # -----------------------------
    # Pick output directory
    # -----------------------------
    if args.outdir:
        outdir = Path(args.outdir).resolve()
    else:
        if raw_dir is not None:
            slug = _dataset_slug_from_raw_dir(raw_dir)
            outdir = (REPO_ROOT / "out" / slug).resolve()
        else:
            outdir = (REPO_ROOT / "out").resolve()

    outdir.mkdir(parents=True, exist_ok=True)

    # If metrics_csv wasn't provided, assume it should live in outdir
    if metrics_csv is None:
        metrics_csv = outdir / "cloud_metrics.csv"

    # -----------------------------
    # Step 0: optionally generate baseline metrics
    # -----------------------------
    # Baseline run is required if:
    #  - user demanded it via --ensure-baseline
    #  - OR metrics_csv doesn't exist yet and we have raw_dir
    need_baseline = args.ensure_baseline or (not metrics_csv.exists())

    if need_baseline:
        if raw_dir is None:
            raise SystemExit(
                "ERROR: Baseline generation requires --raw-dir, but you only provided --metrics-csv.\n"
                "Either provide --raw-dir or remove --ensure-baseline."
            )

        _run(
            [
                py,
                str(REPO_ROOT / "scripts" / "pipeline" / "run_sample_pipeline.py"),
                "--raw-dir",
                str(raw_dir),
                "--outdir",
                str(outdir),
                "--save-overlays",
            ]
        )

    if not metrics_csv.exists():
        raise FileNotFoundError(
            f"Expected metrics csv not found: {metrics_csv}\n"
            f"Try:\n"
            f"  python scripts\\pipeline\\run_phase2_sequenceA.py --raw-dir <YOUR_DIR> --ensure-baseline\n"
        )

    features = args.features.strip()

    # -----------------------------
    # A) Simple anomaly: luma delta z-score
    # anomaly_luma.py expects: --features-csv, --outdir, --z
    # -----------------------------
    _run(
        [
            py,
            str(REPO_ROOT / "scripts" / "experiments" / "anomaly" / "anomaly_luma.py"),
            "--features-csv",
            str(metrics_csv),
            "--outdir",
            str(outdir),
            "--z",
            str(args.z),
        ]
    )

    # -----------------------------
    # B) IsolationForest anomaly detector
    # detect_iforest.py expects: --in-csv, --out-csv, --features, [--contamination], ...
    # -----------------------------
    iforest_out = outdir / "iforest_out.csv"
    _run(
        [
            py,
            str(REPO_ROOT / "scripts" / "experiments" / "anomaly" / "detect_iforest.py"),
            "--in-csv",
            str(metrics_csv),
            "--out-csv",
            str(iforest_out),
            "--features",
            features,
            "--contamination",
            str(args.contamination),
        ]
    )

    # -----------------------------
    # C) Baseline clustering (KMeans) + save predictions
    # train_baseline.py expects: --in, --outdir, --k, --features, --save-predictions
    # -----------------------------
    _run(
        [
            py,
            str(REPO_ROOT / "scripts" / "experiments" / "clustering" / "train_baseline.py"),
            "--in",
            str(metrics_csv),
            "--outdir",
            str(outdir),
            "--k",
            str(args.k),
            "--features",
            features,
            "--save-predictions",
        ]
    )

    # Find the newest predictions CSV produced by train_baseline
    pred_files = sorted(outdir.glob("baseline_kmeans_k*_*.predictions.csv"))
    if not pred_files:
        raise FileNotFoundError(
            f"No predictions CSV found in {outdir}.\n"
            f"Expected something like baseline_kmeans_k{args.k}_*.predictions.csv"
        )
    pred_csv = pred_files[-1]

    # -----------------------------
    # D) Visualize clusters using the predictions csv
    # viz_clusters.py expects: --pred-csv, --outdir, --images-root, [--features]
    # NOTE: This needs raw_dir to show thumbnails/links to images.
    # -----------------------------
    if raw_dir is None:
        print(
            "\nWARNING: Skipping viz_clusters because you didn't provide --raw-dir (only --metrics-csv).\n"
            "If you want cluster thumbnails, rerun with --raw-dir pointing to the images."
        )
    else:
        _run(
            [
                py,
                str(REPO_ROOT / "scripts" / "analysis" / "viz_clusters.py"),
                "--pred-csv",
                str(pred_csv),
                "--outdir",
                str(outdir),
                "--images-root",
                str(raw_dir),
                "--features",
                features,
            ]
        )

    # -----------------------------
    # E) Compare detectors (PCA overlay + agreement/disagreement)
    # compare_detectors.py expects: --base-csv, --z-labeled-csv, --if-labeled-csv, --outdir, --features
    # -----------------------------
    z_labeled = outdir / "luma_anomalies.csv"
    _run(
        [
            py,
            str(REPO_ROOT / "scripts" / "analysis" / "compare_detectors.py"),
            "--base-csv",
            str(metrics_csv),
            "--z-labeled-csv",
            str(z_labeled),
            "--if-labeled-csv",
            str(iforest_out),
            "--outdir",
            str(outdir / "viz"),
            "--features",
            features,
        ]
    )

    print("\nDONE (Phase 2)")
    if raw_dir is not None:
        print(f"Raw frames: {raw_dir}")
    print(f"Out dir:    {outdir}")
    print(f"Metrics:    {metrics_csv}")
    print(f"IForest:    {iforest_out}")
    print(f"KMeans:     {pred_csv}")
    print(f"Viz dir:    {outdir / 'viz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
