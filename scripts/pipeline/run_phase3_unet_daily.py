# scripts/pipeline/run_phase3_unet_daily.py
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import subprocess
import sys


def yday_utc() -> str:
    return (datetime.now(timezone.utc).date() - timedelta(days=1)).strftime("%Y-%m-%d")


def run(cmd: list[str]) -> None:
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd)


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 3: run U-Net inference on yesterday's daily master frames.")
    ap.add_argument("--day", default=None, help="UTC day YYYY-MM-DD (default: yesterday).")
    ap.add_argument("--checkpoint", default="models/unet/unet_cloudmask_latest.pt", help="U-Net checkpoint path.")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--device", default=None, help="cpu|cuda (default: auto)")
    args = ap.parse_args()

    day = args.day or yday_utc()

    frames_dir = Path(f"data/raw/{day}/master/frames")
    if not frames_dir.exists():
        raise SystemExit(f"Missing frames_dir: {frames_dir} (run daily rollup first)")

    outdir = Path(f"out/{day}/phase3_unet")
    outdir.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, "-m", "ml.scripts.infer_unet",
           "--frames", str(frames_dir),
           "--outdir", str(outdir),
           "--checkpoint", str(Path(args.checkpoint)),
           "--recursive",
           "--threshold", str(args.threshold),
           "--size", str(args.size)]
    if args.device:
        cmd += ["--device", args.device]

    run(cmd)
    print(f"✅ Phase 3 complete -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
