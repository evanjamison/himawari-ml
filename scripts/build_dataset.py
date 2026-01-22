from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_SAMPLE_RAW = REPO_ROOT / "data" / "sample" / "raw"
DATA_DERIVED = REPO_ROOT / "data" / "derived"

DEFAULT_REQUIRED_COLS = ["relpath", "filename", "suffix", "bytes"]


def find_images(raw_dir: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    if not raw_dir.exists():
        return []
    return sorted(p for p in raw_dir.rglob("*") if p.suffix.lower() in exts)


def build_dataframe(image_paths: list[Path]) -> pd.DataFrame:
    rows = []
    for p in image_paths:
        rows.append(
            {
                "relpath": str(p.relative_to(REPO_ROOT)).replace("\\", "/"),
                "filename": p.name,
                "suffix": p.suffix.lower(),
                "bytes": p.stat().st_size,
            }
        )
    return pd.DataFrame(rows)


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
            "Dataset is empty but --require-nonempty is enabled. "
            f"(mode={mode}, raw_dir={raw_dir})"
        )

    if errors:
        msg = "Dataset validation failed:\n- " + "\n- ".join(errors)
        raise ValueError(msg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--raw-dir",
        default=None,
        help="Input directory (default: data/raw)",
    )
    ap.add_argument(
        "--sample",
        action="store_true",
        help="Use sample raw data (data/sample/raw)",
    )
    ap.add_argument(
        "--out",
        default=str(DATA_DERIVED / "samples.csv"),
        help="Output dataset path (CSV only)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, only include first N files",
    )

    # Validation flags
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
        help=f"Comma-separated required columns (default: {','.join(DEFAULT_REQUIRED_COLS)})",
    )

    args = ap.parse_args()

    # Resolve raw directory precedence:
    # 1) --sample
    # 2) --raw-dir
    # 3) default data/raw
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

    # Determine require_nonempty default:
    # - sample mode: ON
    # - otherwise: OFF
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
        # Print a clean error and return non-zero so CI fails.
        print(str(e), file=sys.stderr)
        return 2

    df.to_csv(out_path, index=False)

    summary = {
        "mode": mode,
        "raw_dir": str(raw_dir),
        "num_files": len(images),
        "columns": list(df.columns),
        "required_cols": required_cols,
        "require_nonempty": require_nonempty,
        "output": str(out_path),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
