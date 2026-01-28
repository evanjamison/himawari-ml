# scripts/pipeline/filter_black_frames.py
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageStat


def is_black_frame(
    path: Path,
    mean_max: float,
    std_max: float,
) -> tuple[bool, float, float]:
    """
    Returns (is_black, mean_luma, std_luma)
    - mean_luma: average grayscale brightness in [0,255]
    - std_luma: brightness spread; near-black images also have tiny std
    """
    try:
        with Image.open(path) as im:
            im = im.convert("L")  # grayscale
            st = ImageStat.Stat(im)
            mean = float(st.mean[0])
            std = float(st.stddev[0])
            return (mean <= mean_max and std <= std_max), mean, std
    except Exception:
        # If PIL can't open it, treat as "bad" (you can change this to False if you prefer)
        return True, -1.0, -1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, help="Directory of frames (recursive scan)")
    ap.add_argument("--outdir", required=True, help="Directory to write filtered frames (keeps good frames)")
    ap.add_argument(
        "--quarantine",
        default="",
        help="If set, black frames are moved here (preserving relative paths). "
             "If omitted, black frames are NOT copied to outdir and remain in place.",
    )
    ap.add_argument("--pattern", default="*.png", help="Glob pattern (default: *.png)")
    ap.add_argument("--mean-max", type=float, default=6.0, help="Max mean brightness to consider black (0-255)")
    ap.add_argument("--std-max", type=float, default=6.0, help="Max std brightness to consider black (0-255)")
    ap.add_argument("--report", default="black_frame_report.csv", help="CSV report filename")
    ap.add_argument("--mode", choices=["copy", "move"], default="copy", help="How to place good frames into outdir")
    args = ap.parse_args()

    inroot = Path(args.inputs)
    outroot = Path(args.outdir)
    qroot = Path(args.quarantine).resolve() if args.quarantine else None
    outroot.mkdir(parents=True, exist_ok=True)

    files = sorted(inroot.rglob(args.pattern))
    if not files:
        print(f"ERROR: no files found under {inroot} matching {args.pattern}")
        return 1

    report_path = outroot / args.report

    n_total = 0
    n_black = 0
    n_good = 0

    with report_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["relpath", "is_black", "mean_luma", "std_luma"])

        for p in files:
            if not p.is_file():
                continue
            n_total += 1
            rel = p.relative_to(inroot)

            black, mean, std = is_black_frame(p, args.mean_max, args.std_max)
            w.writerow([str(rel).replace("\\", "/"), int(black), f"{mean:.3f}", f"{std:.3f}"])

            if black:
                n_black += 1
                if qroot is not None:
                    dest = qroot / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    # move the bad frame (remove from input)
                    p.replace(dest)
                continue

            # good frame => copy/move to outroot preserving relative paths
            dest = outroot / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if args.mode == "move":
                p.replace(dest)
            else:
                dest.write_bytes(p.read_bytes())
            n_good += 1

    print(f"Scanned: {n_total}  Good: {n_good}  Black: {n_black}")
    print(f"Report: {report_path}")
    if qroot:
        print(f"Quarantine: {qroot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
