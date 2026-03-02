"""
fetch_japan.py — Himawari-9 Japan-area ingest via JMA MSC
==========================================================
Source: Japan Meteorological Agency Meteorological Satellite Center
URL:    https://www.data.jma.go.jp/mscweb/data/himawari/img/jpn/jpn_b13_{HHMM}.jpg

Coverage: 115°E–155°E, 22°N–48°N  (Japan perfectly centered)
Band:     B13 (10.4 µm thermal IR) — illumination-invariant, works 24/7,
          no terminator line, cold cloud tops = bright, clear sky = dark.
Cadence:  10-minute, 24-hour rolling archive (144 frames/day)

Preprocessing:
  - Bottom 28px cropped to remove burned-in timestamp bar
  - Saved as PNG for lossless storage

Env vars (same contract as fetch_latest.py):
  HIMAWARI_OUT_DIR   — output directory (default: himawari_frames)
  TARGET_SAVED       — stop after saving this many frames (default: 12)
  LOOKBACK_MINUTES   — how far back to search (default: 120)
  FETCH_DELAY        — seconds between requests (default: 1.0)
  CROP_TIMESTAMP     — set to "0" to disable bottom crop (default: "1")
"""

import io
import os
import logging
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests
from PIL import Image

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
OUT_DIR        = Path(os.environ.get("HIMAWARI_OUT_DIR", "himawari_frames"))
TARGET_SAVED   = int(os.environ.get("TARGET_SAVED", 12))
LOOKBACK_MIN   = int(os.environ.get("LOOKBACK_MINUTES", 120))
FETCH_DELAY    = float(os.environ.get("FETCH_DELAY", 1.0))
CROP_TIMESTAMP = os.environ.get("CROP_TIMESTAMP", "1") == "1"

BASE_URL      = "https://www.data.jma.go.jp/mscweb/data/himawari/img/jpn"
TIMESTAMP_PX  = 28   # pixels to crop from bottom to remove timestamp bar
MIN_BYTES     = 10_000

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "himawari-ml-research/1.0"})


def snap_10min(dt: datetime) -> datetime:
    """Floor datetime to nearest 10-minute boundary."""
    return dt.replace(minute=(dt.minute // 10) * 10, second=0, microsecond=0)


def frame_url(dt: datetime) -> str:
    return f"{BASE_URL}/jpn_b13_{dt.hour:02d}{dt.minute:02d}.jpg"


def frame_path(dt: datetime) -> Path:
    ts = dt.strftime("%Y%m%dT%H%M%SZ")
    return OUT_DIR / f"himawari_{ts}.png"


def fetch_frame(dt: datetime) -> bool:
    """Download, optionally crop, and save one frame. Returns True if saved."""
    out = frame_path(dt)
    if out.exists():
        log.debug(f"exists: {out.name}")
        return False

    url = frame_url(dt)
    try:
        r = SESSION.get(url, timeout=20)
        if r.status_code != 200:
            log.debug(f"HTTP {r.status_code}: {url}")
            return False
        if len(r.content) < MIN_BYTES:
            log.debug(f"too small ({len(r.content)}B), skipping: {url}")
            return False

        img = Image.open(io.BytesIO(r.content)).convert("RGB")

        if CROP_TIMESTAMP:
            w, h = img.size
            img = img.crop((0, 0, w, h - TIMESTAMP_PX))

        out.parent.mkdir(parents=True, exist_ok=True)
        img.save(out, "PNG")
        log.info(f"saved {out.name}  ({img.size[0]}x{img.size[1]})")
        return True

    except Exception as e:
        log.warning(f"fetch error {url}: {e}")
        return False


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build slot list: walk back from (now - 15min) in 10-min steps
    # 15-min lag gives JMA time to publish; archive goes back 24h
    latest = snap_10min(datetime.now(timezone.utc) - timedelta(minutes=15))
    cutoff = latest - timedelta(minutes=LOOKBACK_MIN)

    slots = []
    t = latest
    while t >= cutoff:
        slots.append(t)
        t -= timedelta(minutes=10)

    log.info(
        f"Japan-area ingest (JMA B13) | outdir={OUT_DIR} | target={TARGET_SAVED} "
        f"| lookback={LOOKBACK_MIN} min | slots={len(slots)} "
        f"| crop_timestamp={CROP_TIMESTAMP} "
        f"| latest={latest.strftime('%Y-%m-%dT%H:%MZ')}"
    )

    saved = 0
    skipped = 0
    for dt in slots:
        if saved >= TARGET_SAVED:
            break
        ok = fetch_frame(dt)
        if ok:
            saved += 1
        else:
            skipped += 1
        if FETCH_DELAY > 0:
            time.sleep(FETCH_DELAY)

    log.info(f"Done. saved={saved}  skipped={skipped}")
    if saved == 0:
        log.error("No frames saved — check JMA connectivity or that Pillow is installed.")


if __name__ == "__main__":
    main()
