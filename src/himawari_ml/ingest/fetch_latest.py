from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

from himawari_ml.utils.io import data_dir, ensure_dir
from himawari_ml.utils.logging import get_logger

logger = get_logger()

# ---- BASES -------------------------------------------------

S3_BASE = "https://himawari8.s3.amazonaws.com"
NICT_BASES = [
    "https://himawari8-dl.nict.go.jp/himawari8/img",
    "https://himawari8.nict.go.jp/himawari8/img",
]

PRODUCT = "D531106"
LEVEL = "4d"
TILE_PX = 550
GRID_N = 4


# ---- HELPERS ----------------------------------------------

def fetch(url: str, timeout=30) -> bytes:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "himawari-ml"})
    r.raise_for_status()
    return r.content


def try_bases(path: str) -> bytes:
    # 1️⃣ S3 first (works on GitHub Actions)
    try:
        return fetch(f"{S3_BASE}{path}")
    except Exception as e:
        logger.warning(f"S3 failed: {e}")

    # 2️⃣ NICT fallback (local only)
    for base in NICT_BASES:
        try:
            return fetch(f"{base}{path}")
        except Exception as e:
            logger.warning(f"NICT failed {base}: {e}")

    raise RuntimeError(f"All sources failed for {path}")


def tile_path(ts: datetime, x: int, y: int) -> str:
    yyyy = ts.strftime("%Y")
    mm = ts.strftime("%m")
    dd = ts.strftime("%d")
    hhmm = ts.strftime("%H%M")
    return f"/{PRODUCT}/{LEVEL}/{TILE_PX}/{yyyy}/{mm}/{dd}/{hhmm}_00_{x}_{y}.png"


def stitch(ts: datetime) -> Image.Image:
    canvas = Image.new("RGB", (TILE_PX * GRID_N, TILE_PX * GRID_N))
    for y in range(GRID_N):
        for x in range(GRID_N):
            raw = try_bases(tile_path(ts, x, y))
            tile = Image.open(BytesIO(raw)).convert("RGB")
            canvas.paste(tile, (x * TILE_PX, y * TILE_PX))
    return canvas


def is_blank(img: Image.Image) -> bool:
    small = img.resize((64, 64))
    black = sum(1 for p in small.getdata() if sum(p) < 10)
    return black / (64 * 64) > 0.97


# ---- MAIN --------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback-hours", type=int, default=48)
    ap.add_argument("--max-frames", type=int, default=96)
    args = ap.parse_args()

    out_dir = ensure_dir(data_dir() / "raw")
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)

    saved = 0
    for i in range(args.lookback_hours * 6):
        if saved >= args.max_frames:
            break

        ts = now - timedelta(minutes=10 * i)
        out = out_dir / f"himawari_{ts.strftime('%Y%m%d_%H%M')}Z.png"
        if out.exists():
            continue

        try:
            img = stitch(ts)
            if is_blank(img):
                continue
            img.save(out)
            saved += 1
            logger.info(f"Saved {out.name}")
        except Exception as e:
            logger.warning(f"{ts}: {e}")
            time.sleep(2)

    logger.info(f"Done. Saved {saved} frames.")


if __name__ == "__main__":
    main()

