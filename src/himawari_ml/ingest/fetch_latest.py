from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

import certifi
import requests
import urllib3
from PIL import Image

from himawari_ml.utils.io import data_dir, ensure_dir
from himawari_ml.utils.logging import get_logger

logger = get_logger()


# ----------------------------
# Config (simple + explicit)
# ----------------------------
@dataclass(frozen=True)
class HimawariConfig:
    # NICT tile server base
    base: str = "https://himawari8-dl.nict.go.jp/himawari8/img"

    # Product folder on NICT tiles
    product: str = "D531106"

    # Tile grid size: 1d->1x1, 2d->2x2, 4d->4x4, 8d->8x8, ...
    level: str = "4d"

    # Each tile is width x width pixels (commonly 550)
    tile_px: int = 550

    # Request timeouts and retry behavior
    timeout_s: int = 30
    retries: int = 4
    backoff_s: float = 1.5

    # Dev-only override
    allow_insecure_ssl: bool = bool(int(os.getenv("ALLOW_INSECURE_SSL", "0") or "0"))


CFG = HimawariConfig(
    base=os.getenv("HIMAWARI_BASE", HimawariConfig.base),
    product=os.getenv("HIMAWARI_PRODUCT", HimawariConfig.product),
    level=os.getenv("HIMAWARI_LEVEL", HimawariConfig.level),
    tile_px=int(os.getenv("HIMAWARI_TILE_PX", str(HimawariConfig.tile_px))),
)


if CFG.allow_insecure_ssl:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ----------------------------
# Helpers
# ----------------------------
def _grid_n(level: str) -> int:
    digits = "".join([c for c in level if c.isdigit()])
    if not digits:
        raise ValueError(f"Bad level: {level}")
    return int(digits)


def _request_bytes(url: str, timeout_s: int, retries: int, backoff_s: float) -> bytes:
    """
    Download bytes with retry.
    - Uses no-cache headers to avoid stale latest.json / CDN caching.
    - Secure default: verify with certifi CA bundle
    - Dev-only override: verify=False when ALLOW_INSECURE_SSL=1
    """
    last_err: Optional[Exception] = None
    verify = False if CFG.allow_insecure_ssl else certifi.where()

    headers = {
        "User-Agent": "himawari-ml/1.0 (+https://github.com/)",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }

    for i in range(retries):
        try:
            r = requests.get(url, timeout=timeout_s, verify=verify, headers=headers)
            r.raise_for_status()
            return r.content
        except Exception as e:
            last_err = e
            sleep = backoff_s * (2**i)
            logger.warning(f"Fetch failed ({i+1}/{retries}) for {url} -> {e}. Sleeping {sleep:.1f}s")
            time.sleep(sleep)

    raise RuntimeError(f"Failed to fetch after {retries} retries: {url}") from last_err


def _latest_timestamp_from_json() -> Optional[datetime]:
    """
    NICT provides a 'latest.json' for some products (e.g., D531106).
    If reachable, this is the cleanest way to know what timestamp to request.
    We add a cache-buster query param and no-cache headers above.
    """
    # cache buster to avoid GH runner / CDN returning stale json
    bust = int(time.time())
    url = f"{CFG.base}/{CFG.product}/latest.json?cb={bust}"

    try:
        raw = _request_bytes(url, CFG.timeout_s, CFG.retries, CFG.backoff_s)
        obj = json.loads(raw.decode("utf-8", errors="replace"))

        date_str = obj.get("date") or obj.get("datetime") or obj.get("time")
        if not date_str:
            logger.warning(f"latest.json missing recognizable datetime field: keys={list(obj.keys())}")
            return None

        date_str = date_str.replace("T", " ").replace("Z", "").strip()
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.replace(tzinfo=timezone.utc)
            except ValueError:
                continue

        logger.warning(f"Could not parse date format from latest.json: {date_str}")
        return None

    except Exception as e:
        logger.warning(f"Could not fetch/parse latest.json ({url}): {e}")
        return None


def _round_down(dt: datetime, minutes: int) -> datetime:
    dt = dt.replace(second=0, microsecond=0)
    minute = (dt.minute // minutes) * minutes
    return dt.replace(minute=minute)


def _candidate_times(
    anchor: Optional[datetime],
    lookback_hours: int,
    step_minutes: int,
) -> list[datetime]:
    """
    Generate timestamps going backwards from anchor in step_minutes increments.
    """
    if anchor is None:
        anchor = datetime.now(timezone.utc)
    anchor = _round_down(anchor, step_minutes)

    n_steps = max(1, int((lookback_hours * 60) // step_minutes))
    return [anchor - timedelta(minutes=step_minutes * k) for k in range(n_steps + 1)]


def _tile_url(ts: datetime, x: int, y: int) -> str:
    """
    Tile URL format:
      {base}/{product}/{level}/{tile_px}/{YYYY}/{MM}/{DD}/{HHMMSS}_{x}_{y}.png
    """
    yyyy = ts.strftime("%Y")
    mm = ts.strftime("%m")
    dd = ts.strftime("%d")
    hhmmss = ts.strftime("%H%M%S")
    return f"{CFG.base}/{CFG.product}/{CFG.level}/{CFG.tile_px}/{yyyy}/{mm}/{dd}/{hhmmss}_{x}_{y}.png"


def _download_and_stitch(ts: datetime) -> Image.Image:
    n = _grid_n(CFG.level)
    canvas = Image.new("RGB", (CFG.tile_px * n, CFG.tile_px * n), color=(0, 0, 0))

    for y in range(n):
        for x in range(n):
            url = _tile_url(ts, x, y)
            try:
                b = _request_bytes(url, CFG.timeout_s, CFG.retries, CFG.backoff_s)
                tile = Image.open(BytesIO(b)).convert("RGB")
                canvas.paste(tile, (x * CFG.tile_px, y * CFG.tile_px))
            except Exception as e:
                # Missing tiles happen; still produce mosaic, but may be blank-ish.
                logger.warning(f"Tile missing/failed: ({x},{y}) {url} -> {e}")

    return canvas


def _looks_like_real_image(img: Image.Image) -> bool:
    """
    Reject images that are almost entirely black (helps avoid saving blank mosaics).
    """
    small = img.resize((128, 128))
    px = small.getdata()
    near_black = 0
    total = 128 * 128
    for r, g, b in px:
        if (r + g + b) < 15:
            near_black += 1
    return (near_black / total) < 0.98


def _out_name(ts: datetime) -> str:
    return f"himawari_{CFG.product}_{CFG.level}_{ts.strftime('%Y%m%d_%H%M%S')}Z.png"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(data_dir() / "raw"), help="Directory to save mosaics")
    ap.add_argument("--lookback-hours", type=int, default=48, help="How far back to search for frames")
    ap.add_argument("--step-minutes", type=int, default=10, help="Timestamp step in minutes (Himawari is usually 10)")
    ap.add_argument("--max-frames", type=int, default=96, help="Max frames to save per run (caps repo growth)")
    ap.add_argument("--no-latest-json", action="store_true", help="Skip latest.json and use now() as anchor")
    args = ap.parse_args()

    raw_dir = ensure_dir(Path(args.out_dir))
    raw_dir.mkdir(parents=True, exist_ok=True)

    anchor = None
    if not args.no_latest_json:
        anchor = _latest_timestamp_from_json()

    candidates = _candidate_times(anchor, lookback_hours=args.lookback_hours, step_minutes=args.step_minutes)

    saved = 0
    checked = 0
    for t in candidates:
        if saved >= args.max_frames:
            break

        out_path = raw_dir / _out_name(t)
        if out_path.exists():
            continue  # already have it

        checked += 1
        logger.info(f"[{checked}] Trying {t.isoformat()} -> {out_path.name}")
        img = _download_and_stitch(t)

        if not _looks_like_real_image(img):
            logger.info("Looks blank-ish; skipping.")
            continue

        img.save(out_path)
        saved += 1
        logger.info(f"Saved: {out_path}")

    logger.info(f"Done. saved={saved} checked={checked} out_dir={raw_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
