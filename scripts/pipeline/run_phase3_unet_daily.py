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


def find_repo_root(start: Path) -> Path:
    p = start.resolve()
    while p != p.parent:
        if (p / ".git").exists() or (p / "pyproject.toml").exists():
            return p
        p = p.parent
    return start.resolve()


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 3: run U-Net inference on daily master frames.")
    ap.add_argument("--day", default=None, help="UTC day YYYY-MM-DD (default: yesterday).")
    ap.add_argument("--checkpoint", default="out/models/unet_cloudmask_latest.pt", help="U-Net checkpoint path.")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--device", default=None, help="cpu|cuda (default: auto)")
    args = ap.parse_args()

    repo = find_repo_root(Path(__file__))

    day = args.day or yday_utc()

    frames_dir = repo / f"data/raw/{day}/master/frames"
    if not frames_dir.exists():
        raise SystemExit(f"Missing frames_dir: {frames_dir} (run daily rollup first)")

    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = (repo / ckpt).resolve()
    if not ckpt.exists():
        raise SystemExit(f"Missing checkpoint: {ckpt}")

    outdir = repo / f"out/{day}/phase3_unet"
    outdir.mkdir(parents=True, exist_ok=True)

    # ✅ Robust infer script detection (handles both infer_unet.py and infer_unet_masks.py)
    candidates = [
        repo / "ml" / "infer_unet.py",
        repo / "ml" / "infer_unet_masks.py",
        repo / "ml" / "scripts" / "infer_unet.py",
        repo / "ml" / "scripts" / "infer_unet_masks.py",
    ]
    infer_py = next((p for p in candidates if p.exists()), None)
    if infer_py is None:
        raise SystemExit(
            "Missing infer script. Expected one of:\n"
            + "\n".join([f"  - {p}" for p in candidates])
        )

    print(f"Using infer script: {infer_py}")

    # Build command depending on which script we found
    if infer_py.name == "infer_unet_masks.py":
        # infer_unet_masks.py expects --ckpt and often uses --image-size
        cmd = [
            sys.executable,
            str(infer_py),
            "--frames",
            str(frames_dir),
            "--outdir",
            str(outdir),
            "--ckpt",
            str(ckpt),
            "--recursive",
            "--threshold",
            str(args.threshold),
            "--image-size",
            str(args.size),
        ]
        if args.device:
            cmd += ["--device", args.device]
    else:
        # infer_unet.py expects --checkpoint and uses --size
        cmd = [
            sys.executable,
            str(infer_py),
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
        ]
        if args.device:
            cmd += ["--device", args.device]

    run(cmd)
    print(f"✅ Phase 3 complete -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
