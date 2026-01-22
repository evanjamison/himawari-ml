from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
from PIL import Image, UnidentifiedImageError, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_SAMPLE_RAW = REPO_ROOT / "data" / "sample" / "raw"
DATA_DERIVED = REPO_ROOT / "data" / "derived"

# Now that we extract real features, require them by default.
DEFAULT_REQUIRED_COLS = [
    "relpath",
    "filename",
    "suffix",
    "bytes",
    "width",
    "height",
    "aspect_ratio",
    "mean_r",
    "mean_g",
    "mean_b",
    "mean_luma",
    "std_luma",
]


def find_images(raw_dir: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    if not raw_dir.exists():
        return []
    return sorted(p for p in raw_dir.rglob("*") if p.suffix.lower() in exts)


def _safe_image_features(path: Path) -> dict:
    """
    Extract lightweight features. If image can't be read, returns NaNs and an error flag.
    Keeps pipeline robust (especially in CI).
    """
    feat = {
        "width": pd.NA,
        "height": pd.NA,
        "aspect_ratio": pd.NA,
        "mean_r": pd.NA,
        "mean_g": pd.NA,
        "mean_b": pd.NA,
        "mean_luma": pd.NA,
        "std_luma": pd.NA,
        "img_ok": 0,
        "img_error": "",
    }

    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)  # handle rotated JPEGs safely
            w, h = im.size
            feat["width"] = int(w)
            feat["height"] = int(h)
            feat["aspect_ratio"] = float(w) / float(h) if h else pd.NA

            # RGB means
            rgb = im.convert("RGB")
            # ImageStat is fast and avoids numpy dependency
            from PIL import ImageStat

            stat_rgb = ImageStat.Stat(rgb)
            r_mean, g_mean, b_mean = stat_rgb.mean
            feat["mean_r"] = float(r_mean)
            feat["mean_g"] = float(g_mean)
            feat["mean_b"] = float(b_mean)

            # Luma (grayscale) mean/std (simple brightness/contrast)
            gray = rgb.convert("L")
            stat_l = ImageStat.Stat(gray)
            feat["mean_luma"] = float(stat_l.mean[0])
            feat["std_luma"] = float(stat_l.stddev[0])

            feat["img_ok"] = 1
            return feat

    except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
        feat["img_error"] = type(e).__name__
        return feat


def build_dataframe(image_paths: list[Path]) -> pd.DataFrame:
    rows = []
    for p in image_paths:
        base = {
            "relpath": str(p.relative_to(REPO_ROOT)).replace("\\", "/"),
            "filename": p.name,
            "suffix": p.suffix.lower(),
            "bytes": p.stat().st_size,
        }
        base.update(_safe_image_features(p))
        rows.append(base)

    df = pd.DataFrame(rows)

    # Use nullable numeric dtypes where possible (nice for missing values)
    for col in ["width", "height", "img_ok"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    for col in ["aspect_ratio", "mean_r", "mean_g", "mean_b", "mean_luma", "std_luma"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def validate_dataset(
    df: pd.DataFrame,
    *,
    require_nonempty: bool,
    required_cols: list[str],
    mode: str,
    raw_dir: Path,
) -> None:
    errors: list[str] = []

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")

    if require_nonempty and len(df) == 0:
        errors.append(
            "Dataset is empty but non-empty is required. "
            f"(mode={mode}, raw_dir={raw_dir})"
        )

    # In sample mode, we also want at least one readable image
    if mode == "sample" and require_nonempty and "img_ok" in df.columns:
        ok_count = int(pd.to_numeric(df["img_ok"], errors="coerce").fillna(0).sum())
        if ok_count == 0:
            errors.append(
                "Sample dataset has zero readable images (img_ok=0 for all rows). "
                "Your sample image may be corrupt/unsupported."
            )

    if errors:
        raise ValueError("Dataset validation failed:\n- " + "\n- ".join(errors))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=None, help="Input directory (default: data/raw)")
    ap.add_argument("--sample", action="store_true", help="Use sample raw data (data/sample/raw)")
    ap.add_argument("--out", default=str(DATA_DERIVED / "samples.csv"), help="Output dataset path (CSV only)")
    ap.add_argument("--limit", type=int, default=0, help="If >0, only include first N files")

    # Validation controls
    ap.add_argument(
        "--require-nonempty",
        action="store_true",
        help="Fail if dataset is empty (default: ON for --sample, OFF otherwise)",
    )
    ap.add_argument(
        "--no-require-nonempty",
        action="store_true",
        help="Override to allow empty dataset",
    )
    ap.add_argument(
        "--require-cols",
        default=",".join(DEFAULT_REQUIRED_COLS),
        help="Comma-separated required columns",
    )

    args = ap.parse_args()

    # Resolve raw directory precedence
    if args.sample:
        raw_dir = DATA_SAMPLE_RAW
        mode = "sample"
    elif args.raw_dir is not None:
        raw_dir = Path(args.raw_dir)
        mode = "custom"
    else:
        raw_dir = DATA_RAW
        mode = "full"

    raw_dir = raw_dir.resolve()

    out_path = Path(args.out).resolve()
    if out_path.suffix.lower() != ".csv":
        out_path = out_path.with_suffix(".csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    images = find_images(raw_dir)
    if args.limit and args.limit > 0:
        images = images[: args.limit]

    df = build_dataframe(images)

    # Default nonempty behavior: ON for sample, OFF otherwise
    require_nonempty = args.require_nonempty or (args.sample and not args.no_require_nonempty)
    if args.no_require_nonempty:
        require_nonempty = False

    required_cols = [c.strip() for c in args.require_cols.split(",") if c.strip()]

    try:
        validate_dataset(
            df,
            require_nonempty=require_nonempty,
            required_cols=required_cols,
            mode=mode,
            raw_dir=raw_dir,
        )
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2

    df.to_csv(out_path, index=False)

    summary = {
        "mode": mode,
        "raw_dir": str(raw_dir),
        "num_files": len(images),
        "num_readable_images": int(pd.to_numeric(df.get("img_ok", 0), errors="coerce").fillna(0).sum())
        if len(df) > 0
        else 0,
        "columns": list(df.columns),
        "output": str(out_path),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
