from __future__ import annotations

import argparse
from pathlib import Path
import sys
import json

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_DERIVED = REPO_ROOT / "data" / "derived"


def find_images(raw_dir: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    if not raw_dir.exists():
        return []
    return sorted([p for p in raw_dir.rglob("*") if p.suffix.lower() in exts])


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=str(DATA_RAW), help="Input directory (default: data/raw)")
    ap.add_argument("--out", default=str(DATA_DERIVED / "samples.parquet"), help="Output dataset path")
    ap.add_argument("--limit", type=int, default=0, help="If >0, only include first N files")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    images = find_images(raw_dir)
    if args.limit and args.limit > 0:
        images = images[: args.limit]

    df = build_dataframe(images)

    # Write parquet if possible; otherwise fallback to CSV
    wrote = None
    if out_path.suffix.lower() == ".parquet":
        try:
            df.to_parquet(out_path, index=False)
            wrote = str(out_path)
        except Exception as e:
            # fallback
            csv_path = out_path.with_suffix(".csv")
            df.to_csv(csv_path, index=False)
            wrote = f"{csv_path} (parquet failed: {type(e).__name__})"
    else:
        df.to_csv(out_path, index=False)
        wrote = str(out_path)

    summary = {
        "raw_dir": str(raw_dir),
        "num_files": len(images),
        "columns": list(df.columns),
        "output": wrote,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
