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
from PIL import Image, ImageStat

log = logging.getLogger("fetch_latest")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")

IN_GHA = os.getenv("GITHUB_ACTIONS", "false").lower() == "true"

# Silence urllib3 "Unverified HTTPS request" warnings when verify=False
from urllib3.exceptions import InsecureRequestWarning  # type: ignore
warnings.filterwarnings("ignore", category=InsecureRequestWarning)

# Mirrors (try in order)
BASES = [
    "https://himawari8-dl.nict.go.jp/himawari8/img",
    "https://himawari8.nict.go.jp/himawari8/img",
]

HEADERS = {"User-Agent": "himawari-ml/1.0 (research; GitHub Actions)"}

# ---- Actions-friendly limits (override via env) ----
# Save at most this many frames per run (prevents long runs)
TARGET_SAVED = int(os.getenv("TARGET_SAVED", "24" if IN_GHA else "48"))
# Frames to consider at most (checked timestamps)
MAX_FRAMES_DEFAULT = int(os.getenv("MAX_FRAMES", "96" if IN_GHA else "192"))
# Lookback window
LOOKBACK_HOURS_DEFAULT = int(os.getenv("LOOKBACK_HOURS", "48" if IN_GHA else "72"))

# Image size expected by your ML (keep 256)
DEFAULT_IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "256"))

# If set, overrides TLS verification.
# Recommended:
#   - GitHub Actions: 0
#   - Local: try 1; if your Windows cert chain is broken, 0 works.
# Values accepted: "0", "1", "true", "false"
_HV = os.getenv("HIMAWARI_VERIFY", "").strip().lower()
if _HV in ("0", "false", "no"):
    VERIFY_SSL = False
elif _HV in ("1", "true", "yes"):
    VERIFY_SSL = True
else:
    # auto
    VERIFY_SSL = (not IN_GHA)

# Darkness threshold (inside Earth disk only). Higher = allow more nighttime.
# Good starting point: 0.70–0.85
MAX_DARK_FRAC = float(os.getenv("MAX_DARK_FRAC", "0.85"))

# Tiling config for full disk (4d / 550 -> 4x4 tiles)
LEVEL = 4
TILE_SIZE = 550


@dataclass
class FetchCfg:
    retries: int = 3
    timeout_s: int = 15
    backoff_base_s: float = 1.0


PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _is_valid_png_bytes(b: bytes, min_bytes: int = 8000) -> bool:
    # Many failure modes return tiny payloads or HTML despite 200.
    if b is None:
        return False
    if len(b) < min_bytes:
        return False
    if not b.startswith(PNG_MAGIC):
        return False
    return True


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def _get(s: requests.Session, url: str, cfg: FetchCfg) -> bytes | None:
    """
    Return content bytes if successful + sane. Otherwise None.
    Avoids log spam: logs only final failure for that URL.
    """
    last_err: Exception | None = None
    for i in range(cfg.retries):
        try:
            r = s.get(url, timeout=cfg.timeout_s, verify=VERIFY_SSL)
            # 404 is common when a timestamp/tile doesn't exist—don't waste retries.
            if r.status_code == 404:
                return None
            r.raise_for_status()

            b = r.content
            if not _is_valid_png_bytes(b):
                # Sometimes we get a "PNG" content-type but tiny/corrupt payload.
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


def _earth_disk_mask(img_rgb: Image.Image, thresh: int = 12) -> Image.Image:
    """
    Create a mask of "non-space" pixels (Earth disk + clouds) by thresholding luminance.
    Returns a mode 'L' image with 0/255 values.
    """
    g = img_rgb.convert("L")
    # Simple threshold: space is near-black; disk has non-trivial luminance.
    return g.point(lambda p: 255 if p > thresh else 0)


def _dark_fraction_on_disk(img_rgb: Image.Image) -> float:
    """
    Fraction of near-black pixels INSIDE the Earth disk mask.
    This avoids counting the huge black background.
    """
    mask = _earth_disk_mask(img_rgb, thresh=12)
    g = img_rgb.convert("L")

    # Compute stats only where mask is 255
    # PIL doesn't natively compute masked histograms nicely, so we do it via cropping trick:
    # Convert masked region to an image where outside-mask is mid-gray (won't count as black),
    # then compute "black-ish" fraction by histogram and correct for outside area.
    outside_fill = 128
    g2 = Image.composite(g, Image.new("L", g.size, outside_fill), mask)

    hist = g2.histogram()
    total = sum(hist)
    if total == 0:
        return 1.0

    # "Near-black" inside disk => 0..9. Outside is ~128, so not counted.
    near_black = sum(hist[:10])

    # But total includes outside pixels too (at 128). We want denom = inside pixels only:
    inside = ImageStat.Stat(mask).sum[0] / 255.0  # count of 255 pixels
    if inside <= 0:
        return 1.0

    return float(near_black / inside)


def fetch_frame(ts: datetime, out_dir: Path, image_size: int, cfg: FetchCfg) -> bool:
    """
    Fetch full-disk 4x4 tiles for timestamp ts, stitch, crop, resize, and save.
    Returns True if saved.
    """
    y, m, d = ts.strftime("%Y"), ts.strftime("%m"), ts.strftime("%d")
    hms = ts.strftime("%H%M%S")
    rel_prefix = f"D531106/{LEVEL}d/{TILE_SIZE}/{y}/{m}/{d}/{hms}"

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"himawari_{hms}.png"

    s = _session()

    for base in BASES:
        try:
            # Download all tiles; if any missing/corrupt, abandon this base.
            tiles: list[list[Image.Image]] = []
            for ty in range(LEVEL):
                row: list[Image.Image] = []
                for tx in range(LEVEL):
                    url = f"{base}/{rel_prefix}_{tx}_{ty}.png"
                    b = _get(s, url, cfg)
                    if b is None:
                        raise RuntimeError(f"tile missing/corrupt ({tx},{ty})")
                    im = Image.open(BytesIO(b)).convert("RGB")
                    row.append(im)
                tiles.append(row)

            # Stitch
            full = Image.new("RGB", (LEVEL * TILE_SIZE, LEVEL * TILE_SIZE))
            for ty in range(LEVEL):
                for tx in range(LEVEL):
                    full.paste(tiles[ty][tx], (tx * TILE_SIZE, ty * TILE_SIZE))

            # Crop to square then resize to model input size
            cropped = _center_crop_square(full)
            resized = cropped.resize((image_size, image_size), Image.BILINEAR)

            dark_frac = _dark_fraction_on_disk(resized)
            if dark_frac > MAX_DARK_FRAC:
                log.info(f"Skip {out_path.name} (dark_frac_on_disk={dark_frac:.3f} > {MAX_DARK_FRAC})")
                return False

            resized.save(out_path)
            log.info(f"Saved {out_path} (dark_frac_on_disk={dark_frac:.3f})")
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
    out = Path("data/raw")
    ts = latest_timestamp_10min()
    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)

    cfg = FetchCfg()

    saved = 0
    checked = 0

    while checked < max_frames and ts >= cutoff and saved < TARGET_SAVED:
        checked += 1
        if fetch_frame(ts, out, image_size=image_size, cfg=cfg):
            saved += 1
        ts -= timedelta(minutes=10)

    log.info(f"Done. Saved {saved} frames (checked {checked}). "
             f"verify_ssl={VERIFY_SSL} target_saved={TARGET_SAVED} max_dark_frac={MAX_DARK_FRAC}")
    return 0


if __name__ == "__main__":
    # Backward compatible:
    #   python -m himawari_ml.ingest.fetch_latest [lookback_hours] [max_frames] [image_size]
    import sys

    def _to_int(s: str, default: int) -> int:
        try:
            return int(s)
        except Exception:
            return default

    lb = _to_int(sys.argv[1], LOOKBACK_HOURS_DEFAULT) if len(sys.argv) > 1 else LOOKBACK_HOURS_DEFAULT
    mf = _to_int(sys.argv[2], MAX_FRAMES_DEFAULT) if len(sys.argv) > 2 else MAX_FRAMES_DEFAULT
    sz = _to_int(sys.argv[3], DEFAULT_IMAGE_SIZE) if len(sys.argv) > 3 else DEFAULT_IMAGE_SIZE

    raise SystemExit(main(lookback_hours=lb, max_frames=mf, image_size=sz))
