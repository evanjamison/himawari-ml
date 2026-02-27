# src/himawari_ml/ingest/fetch_latest.py
from __future__ import annotations

import os
import time
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timedelta, timezone
from io import BytesIO

import requests
from PIL import Image

from himawari_ml.utils.paths import raw_latest_dir, frame_filename

log = logging.getLogger("fetch_latest")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")

IN_GHA = os.getenv("GITHUB_ACTIONS", "false").lower() == "true"

# Silence urllib3 "Unverified HTTPS request" warnings when verify=False
from urllib3.exceptions import InsecureRequestWarning  # type: ignore
warnings.filterwarnings("ignore", category=InsecureRequestWarning)

# ---------------------------------------------------------------------------
# Mirrors — tried in order, first success wins.
# If NICT blocks GitHub Actions IPs, contact osn-system@ml.nict.go.jp
# to request whitelisting of GitHub Actions IP ranges for research use.
# ---------------------------------------------------------------------------
BASES = [
    "https://himawari8-dl.nict.go.jp/himawari8/img",
    "https://himawari8.nict.go.jp/himawari8/img",
]

HEADERS = {"User-Agent": "himawari-ml/1.0 (research; GitHub Actions)"}

# ---- Actions-friendly limits (override via env) ----
TARGET_SAVED = int(os.getenv("TARGET_SAVED", "24" if IN_GHA else "48"))
MAX_FRAMES_DEFAULT = int(os.getenv("MAX_FRAMES", "96" if IN_GHA else "192"))
LOOKBACK_HOURS_DEFAULT = int(os.getenv("LOOKBACK_HOURS", "48" if IN_GHA else "72"))

# Image size expected by your ML (keep 256 by default)
DEFAULT_IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "256"))

# TLS verification override
_HV = os.getenv("HIMAWARI_VERIFY", "").strip().lower()
if _HV in ("0", "false", "no"):
    VERIFY_SSL = False
elif _HV in ("1", "true", "yes"):
    VERIFY_SSL = True
else:
    VERIFY_SSL = (not IN_GHA)

# Tiling config for full disk (4d / 550 -> 4x4 tiles)
LEVEL = 4
TILE_SIZE = 550


@dataclass
class FetchCfg:
    retries: int = 3
    timeout_s: int = 15
    backoff_base_s: float = 1.0


PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _is_png_bytes(b: bytes | None) -> bool:
    if not b:
        return False
    return b.startswith(PNG_MAGIC)


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def _get(s: requests.Session, url: str, cfg: FetchCfg) -> bytes | None:
    last_err: Exception | None = None
    for i in range(cfg.retries):
        try:
            r = s.get(url, timeout=cfg.timeout_s, verify=VERIFY_SSL)
            if r.status_code == 403:
                log.warning(
                    f"403 Forbidden: {url} — NICT is blocking this IP. "
                    f"Contact osn-system@ml.nict.go.jp to request whitelisting."
                )
                return None  # don't retry 403 — it won't recover
            if r.status_code == 404:
                return None
            r.raise_for_status()

            b = r.content
            if not _is_png_bytes(b):
                return None
            return b
        except Exception as e:
            last_err = e
            if i == cfg.retries - 1:
                log.warning(f"Fetch failed: {url} -> {e}")
            time.sleep(cfg.backoff_base_s * (2**i))
    if last_err:
        return None
    return None


def latest_timestamp_10min() -> datetime:
    now = datetime.now(timezone.utc)
    minute = (now.minute // 10) * 10
    return now.replace(minute=minute, second=0, microsecond=0)


def _center_crop_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def fetch_frame(ts: datetime, out_dir: Path, image_size: int, cfg: FetchCfg) -> bool:
    y, m, d = ts.strftime("%Y"), ts.strftime("%m"), ts.strftime("%d")
    hms = ts.strftime("%H%M%S")
    rel_prefix = f"D531106/{LEVEL}d/{TILE_SIZE}/{y}/{m}/{d}/{hms}"

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / frame_filename(ts)

    if out_path.exists():
        log.info(f"Already exists, skipping: {out_path.name}")
        return True

    s = _session()

    for base in BASES:
        try:
            tiles: list[list[Image.Image]] = []
            failed = False

            for ty in range(LEVEL):
                if failed:
                    break
                row: list[Image.Image] = []
                for tx in range(LEVEL):
                    url = f"{base}/{rel_prefix}_{tx}_{ty}.png"
                    b = _get(s, url, cfg)
                    if b is None:
                        log.warning(f"Base failed: {base} -> tile missing/corrupt ({tx},{ty})")
                        failed = True
                        break
                    im = Image.open(BytesIO(b)).convert("RGB")
                    row.append(im)
                if not failed:
                    tiles.append(row)

            if failed:
                continue  # try next mirror

            full = Image.new("RGB", (LEVEL * TILE_SIZE, LEVEL * TILE_SIZE))
            for ty in range(LEVEL):
                for tx in range(LEVEL):
                    full.paste(tiles[ty][tx], (tx * TILE_SIZE, ty * TILE_SIZE))

            cropped = _center_crop_square(full)
            resized = cropped.resize((image_size, image_size), Image.BILINEAR)

            resized.save(out_path)
            log.info(f"Saved {out_path.name} (mirror: {base})")
            return True

        except Exception as e:
            log.warning(f"Base failed: {base} -> {e}")

    log.warning(f"All sources failed for {rel_prefix}_x_y.png (stitched)")
    return False


def main(
    lookback_hours: int = LOOKBACK_HOURS_DEFAULT,
    max_frames: int = MAX_FRAMES_DEFAULT,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> int:
    out = (
        Path(os.getenv("HIMAWARI_OUT_DIR", "")).expanduser()
        if os.getenv("HIMAWARI_OUT_DIR")
        else raw_latest_dir()
    )

    ts = latest_timestamp_10min()
    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    cfg = FetchCfg()

    log.info(f"NICT Himawari ingest: lookback={lookback_hours}h target={TARGET_SAVED} out={out}")

    saved = 0
    checked = 0

    while checked < max_frames and ts >= cutoff and saved < TARGET_SAVED:
        checked += 1
        if fetch_frame(ts, out, image_size=image_size, cfg=cfg):
            saved += 1
        ts -= timedelta(minutes=10)

    log.info(
        f"Done. Saved {saved} frames (checked {checked}). "
        f"out={out} verify_ssl={VERIFY_SSL} target_saved={TARGET_SAVED}"
    )
    return 0


if __name__ == "__main__":
    import sys

    def _to_int(s: str, default: int) -> int:
        try:
            return int(s)
        except Exception:
            return default

    lb = _to_int(sys.argv[1], LOOKBACK_HOURS_DEFAULT) if len(sys.argv) > 1 else LOOKBACK_HOURS_DEFAULT
    mf = _to_int(sys.argv[2], MAX_FRAMES_DEFAULT)     if len(sys.argv) > 2 else MAX_FRAMES_DEFAULT
    sz = _to_int(sys.argv[3], DEFAULT_IMAGE_SIZE)      if len(sys.argv) > 3 else DEFAULT_IMAGE_SIZE

    raise SystemExit(main(lookback_hours=lb, max_frames=mf, image_size=sz))
