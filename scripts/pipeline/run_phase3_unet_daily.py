# scripts/pipeline/run_phase3_unet_daily.py
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print(">>", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--day", required=True, help="YYYY-MM-DD")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    day = args.day
    frames_dir = Path(f"data/raw/{day}/master/frames")
    outdir = Path(f"out/{day}/phase3_unet")
    outdir.mkdir(parents=True, exist_ok=True)

    if not frames_dir.exists():
        raise SystemExit(f"ERROR: frames_dir not found: {frames_dir}")

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise SystemExit(f"ERROR: checkpoint not found: {ckpt}")

    # ✅ IMPORTANT: call by FILE PATH, not `python -m ...`
    infer_script = Path("ml") / "infer_unet.py"
    if not infer_script.exists():
        # fallback if your file is named slightly differently
        infer_script = Path("ml") / "infer_unet_masks.py"

    if not infer_script.exists():
        raise SystemExit(
            "ERROR: Could not find ml/infer_unet.py or ml/infer_unet_masks.py. "
            "Check your repo's ml/ folder filenames."
        )

    cmd = [
        sys.executable,
        str(infer_script),
        "--frames",
        str(frames_dir),
        "--outdir",
        str(outdir),
        "--checkpoint",
        str(ckpt),
        "--recursive",
        "--threshold",
        str(args.threshold),
        "--size",
        str(args.size),
        "--device",
        str(args.device),
    ]

    # Optional: let infer script discover config if it uses env vars
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    run(cmd)


if __name__ == "__main__":
    main()
