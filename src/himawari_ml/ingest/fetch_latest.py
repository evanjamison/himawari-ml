from __future__ import annotations

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

    # Product folder (commonly used full-disk product on NICT tiles)
    product: str = "D531106"

    # Tile grid size: 1d->1x1, 2d->2x2, 4d->4x4, 8d->8x8, ...
    level: str = "4d"

    # Each tile is width x width pixels (commonly 550)
    tile_px: int = 550

    # Request timeouts and retry behavior
    timeout_s: int = 30
    retries: int = 3
    backoff_s: float = 1.5

    # Dev-only: allow insecure SSL (workaround for SSL-intercepted networks)
    allow_insecure_ssl: bool = os.getenv("ALLOW_INSECURE_SSL", "0") == "1"


CFG = HimawariConfig()

# If insecure mode, silence warnings (dev only!)
if CFG.allow_insecure_ssl:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    logger.warning("ALLOW_INSECURE_SSL=1 -> SSL verification DISABLED (dev-only).")


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
    - Secure default: verify with certifi CA bundle
    - Dev-only override: verify=False when ALLOW_INSECURE_SSL=1
    """
    last_err: Optional[Exception] = None
    verify = False if CFG.allow_insecure_ssl else certifi.where()

    for i in range(retries):
        try:
            r = requests.get(url, timeout=timeout_s, verify=verify)
            r.raise_for_status()
            return r.content
        except Exception as e:
            last_err = e
            sleep = backoff_s * (2**i)
            logger.warning(
                f"Fetch failed ({i+1}/{retries}) for {url} -> {e}. Sleeping {sleep:.1f}s"
            )
            time.sleep(sleep)

    raise RuntimeError(f"Failed to fetch after {retries} retries: {url}") from last_err


def _latest_timestamp_from_json() -> Optional[datetime]:
    """
    NICT provides a 'latest.json' for some products (e.g., D531106).
    If reachable, this is the cleanest way to know what timestamp to request.
    """
    url = f"{CFG.base}/{CFG.product}/latest.json"
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


def _round_down_to_10min(dt: datetime) -> datetime:
    dt = dt.replace(second=0, microsecond=0)
    minute = (dt.minute // 10) * 10
    return dt.replace(minute=minute)


def _candidate_times_fallback(n_back: int = 18) -> list[datetime]:
    """
    If latest.json fails, try now (UTC) rounded down to 10 minutes,
    then step backwards in 10-minute increments.
    """
    now = datetime.now(timezone.utc)
    t0 = _round_down_to_10min(now)
    return [t0 - timedelta(minutes=10 * k) for k in range(n_back)]


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


def main() -> int:
    raw_dir = ensure_dir(data_dir() / "raw")

    ts = _latest_timestamp_from_json()
    candidates = [ts] if ts else _candidate_times_fallback(n_back=18)

    last_img: Optional[Image.Image] = None
    last_ts: Optional[datetime] = None

    for t in candidates:
        logger.info(f"Attempting timestamp: {t.isoformat()}")
        img = _download_and_stitch(t)
        last_img, last_ts = img, t

        if _looks_like_real_image(img):
            logger.info("Image sanity check passed.")
            break
        logger.warning("Image looks mostly blank; trying an earlier timestamp...")

    if last_img is None or last_ts is None:
        raise SystemExit("Failed to build any mosaic image (no candidates worked).")

    out_name = f"himawari_{CFG.product}_{CFG.level}_{last_ts.strftime('%Y%m%d_%H%M%S')}Z.png"
    out_path = raw_dir / out_name
    last_img.save(out_path)
    logger.info(f"Saved stitched image: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
