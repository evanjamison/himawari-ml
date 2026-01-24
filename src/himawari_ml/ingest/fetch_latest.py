from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from typing import Optional

import requests
from PIL import Image

from himawari_ml.utils.io import data_dir, ensure_dir
from himawari_ml.utils.logging import get_logger

logger = get_logger()

# ----------------------------
# CONFIG — HTTP ONLY (NICT supports this)
# ----------------------------
@dataclass(frozen=True)
class HimawariConfig:
    base: str = "http://himawari8-dl.nict.go.jp/himawari8/img"
    product: str = "D531106"
    level: str = "4d"
    tile_px: int = 550

    timeout_s: int = 30
    retries: int = 4
    backoff_s: float = 1.5


CFG = HimawariConfig()


def _grid_n(level: str) -> int:
    return int("".join(c for c in level if c.isdigit()))


def _request_bytes(url: str) -> bytes:
    headers = {
        "User-Agent": "himawari-ml/1.0",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }

    last_err: Optional[Exception] = None
    for i in range(CFG.retries):
        try:
            r = requests.get(url, timeout=CFG.timeout_s, headers=headers)
            r.raise_for_status()
            return r.content
        except Exception as e:
            last_err = e
            sleep = CFG.backoff_s * (2**i)
            logger.warning(f"Fetch failed ({i+1}/{CFG.retries}) {url} → {e}; sleeping {sleep:.1f}s")
            time.sleep(sleep)

    raise RuntimeError(f"Failed after {CFG.retries} retries: {url}") from last_err


def _latest_timestamp_from_json() -> Optional[datetime]:
    bust = int(time.time())
    url = f"{CFG.base}/{CFG.product}/latest.json?cb={bust}"
    try:
        raw = _request_bytes(url)
        obj = json.loads(raw.decode("utf-8", errors="replace"))
        date_str = obj.get("date")
        if not date_str:
            return None
        dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        return dt.replace(tzinfo=timezone.utc)
    except Exception as e:
        logger.warning(f"Could not fetch latest.json: {e}")
        return None


def _round_down(dt: datetime, minutes: int) -> datetime:
    dt = dt.replace(second=0, microsecond=0)
    return dt.replace(minute=(dt.minute // minutes) * minutes)


def _candidate_times(anchor: Optional[datetime], lookback_hours: int, step_minutes: int):
    if anchor is None:
        anchor = datetime.now(timezone.utc)
    anchor = _round_down(anchor, step_minutes)
    n = int((lookback_hours * 60) / step_minutes)
    return [anchor - timedelta(minutes=i * step_minutes) for i in range(n + 1)]


def _tile_url(ts: datetime, x: int, y: int) -> str:
    yyyy, mm, dd = ts.strftime("%Y"), ts.strftime("%m"), ts.strftime("%d")
    hhmmss = ts.strftime("%H%M%S")
    return (
        f"{CFG.base}/{CFG.product}/{CFG.level}/{CFG.tile_px}/"
        f"{yyyy}/{mm}/{dd}/{hhmmss}_{x}_{y}.png"
    )


def _download_and_stitch(ts: datetime) -> Image.Image:
    n = _grid_n(CFG.level)
    canvas = Image.new("RGB", (CFG.tile_px * n, CFG.tile_px * n))

    for y in range(n):
        for x in range(n):
            try:
                raw = _request_bytes(_tile_url(ts, x, y))
                tile = Image.open(BytesIO(raw)).convert("RGB")
                canvas.paste(tile, (x * CFG.tile_px, y * CFG.tile_px))
            except Exception as e:
                logger.warning(f"Tile failed ({x},{y}): {e}")

    return canvas


def _looks_real(img: Image.Image) -> bool:
    small = img.resize((128, 128))
    dark = sum(1 for p in small.getdata() if sum(p) < 15)
    return dark / (128 * 128) < 0.98


def _out_name(ts: datetime) -> str:
    return f"himawari_{CFG.product}_{CFG.level}_{ts.strftime('%Y%m%d_%H%M%S')}Z.png"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(data_dir() / "raw"))
    ap.add_argument("--lookback-hours", type=int, default=48)
    ap.add_argument("--step-minutes", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=96)
    args = ap.parse_args()

    out_dir = ensure_dir(Path(args.out_dir))
    anchor = _latest_timestamp_from_json()
    times = _candidate_times(anchor, args.lookback_hours, args.step_minutes)

    saved = 0
    for ts in times:
        if saved >= args.max_frames:
            break

        out = out_dir / _out_name(ts)
        if out.exists():
            continue

        logger.info(f"Trying {ts.isoformat()}")
        img = _download_and_stitch(ts)
        if not _looks_real(img):
            logger.info("Blank frame, skipping")
            continue

        img.save(out)
        logger.info(f"Saved {out}")
        saved += 1

    logger.info(f"Done. saved={saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
