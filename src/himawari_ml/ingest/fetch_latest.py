# src/himawari_ml/ingest/fetch_latest.py
from __future__ import annotations
# MUST be first — before requests / urllib3 are imported
import os
import warnings

if os.getenv("GITHUB_ACTIONS") == "true":
    from urllib3.exceptions import InsecureRequestWarning
    warnings.filterwarnings("ignore", category=InsecureRequestWarning)
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone
from io import BytesIO

import requests
from PIL import Image

log = logging.getLogger("fetch_latest")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")

IN_GHA = os.getenv("GITHUB_ACTIONS", "false").lower() == "true"

# ✅ Official NICT CDN endpoints (mirrors)
BASES = [
    "https://himawari8-dl.nict.go.jp/himawari8/img",
    "https://himawari8.nict.go.jp/himawari8/img",
]

HEADERS = {
    "User-Agent": "himawari-ml/1.0 (research; GitHub Actions)"
}

# In Actions, NICT sometimes fails TLS chain validation; locally it usually works.
VERIFY_SSL = not IN_GHA

# If you want to keep everything even if it’s mostly black, set to 1.0
# Default 0.98 = skip frames that are ~98% near-black after resize.
SKIP_DARK_FRAC = float(os.getenv("SKIP_DARK_FRAC", "0.98"))


def _get(url: str, retries: int = 4, timeout: int = 30) -> requests.Response:
    last_err: Exception | None = None
    for i in range(retries):
        try:
            r = requests.get(
                url,
                headers=HEADERS,
                timeout=timeout,
                verify=VERIFY_SSL,
            )
            r.raise_for_status()
            return r
        except Exception as e:
            last_err = e
            log.warning(f"Fetch failed ({i+1}/{retries}) for {url} -> {e}")
            time.sleep(2 ** i)
    raise RuntimeError(f"Failed after {retries} retries: {url} ({last_err})")


def latest_timestamp() -> datetime:
    # Himawari updates every 10 minutes
    now = datetime.now(timezone.utc)
    minute = (now.minute // 10) * 10
    return now.replace(minute=minute, second=0, microsecond=0)


def _center_crop_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def _dark_fraction(img256: Image.Image) -> float:
    """
    Fraction of pixels that are near-black in a 256x256 RGB image.
    Avoids numpy dependency by using PIL convert to L and histogram.
    """
    g = img256.convert("L")
    hist = g.histogram()  # 256 bins
    total = sum(hist)
    if total == 0:
        return 1.0
    near_black = sum(hist[:10])  # 0..9
    return near_black / total


def fetch_frame(ts: datetime, out_dir: Path, image_size: int = 256) -> bool:
    """
    Fetch Himawari full disk for timestamp `ts` using 4d/550 tiled product,
    stitch 4x4 tiles, center-crop, resize to `image_size`, save to out_dir.
    """
    y, m, d = ts.strftime("%Y"), ts.strftime("%m"), ts.strftime("%d")
    hms = ts.strftime("%H%M%S")

    # 4d/550 => 4x4 tiles, each 550px => stitched 2200x2200
    level = 4
    tile_size = 550
    n = level  # tiles are 0..3

    rel_prefix = f"D531106/{level}d/{tile_size}/{y}/{m}/{d}/{hms}"

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"himawari_{hms}.png"

    for base in BASES:
        try:
            # Download tiles row-major (y then x)
            tiles: list[list[Image.Image]] = []
            for yy in range(n):
                row: list[Image.Image] = []
                for xx in range(n):
                    url = f"{base}/{rel_prefix}_{xx}_{yy}.png"
                    r = _get(url)
                    im = Image.open(BytesIO(r.content)).convert("RGB")
                    row.append(im)
                tiles.append(row)

            # Stitch
            full = Image.new("RGB", (n * tile_size, n * tile_size))
            for yy in range(n):
                for xx in range(n):
                    full.paste(tiles[yy][xx], (xx * tile_size, yy * tile_size))

            # Crop + resize for ML
            cropped = _center_crop_square(full)
            img256 = cropped.resize((image_size, image_size), Image.BILINEAR)

            # Optional: skip mostly-black frames
            dark_frac = _dark_fraction(img256)
            if dark_frac >= SKIP_DARK_FRAC:
                log.warning(
                    f"Skipping mostly-dark frame {hms} (dark_frac={dark_frac:.3f}) from base={base}"
                )
                return False

            img256.save(path)
            log.info(f"Saved {path} (dark_frac={dark_frac:.3f})")
            return True

        except Exception as e:
            log.warning(f"Base failed: {base} -> {e}")

    log.warning(f"All sources failed for {rel_prefix}_x_y.png (stitched)")
    return False


def main(lookback_hours: int = 72, max_frames: int = 192, image_size: int = 256):
    out = Path("data/raw")
    ts = latest_timestamp()

    saved = 0
    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)

    for _ in range(max_frames):
        if ts < cutoff:
            break

        if fetch_frame(ts, out, image_size=image_size):
            saved += 1

        ts -= timedelta(minutes=10)

    log.info(f"Done. Saved {saved} frames.")


if __name__ == "__main__":
    # Optional CLI overrides without introducing argparse dependency in the module API
    # Usage example:
    #   python -m himawari_ml.ingest.fetch_latest 72 192 256
    import sys

    try:
        lb = int(sys.argv[1]) if len(sys.argv) > 1 else 72
        mf = int(sys.argv[2]) if len(sys.argv) > 2 else 192
        sz = int(sys.argv[3]) if len(sys.argv) > 3 else 256
    except Exception:
        lb, mf, sz = 72, 192, 256

    main(lookback_hours=lb, max_frames=mf, image_size=sz)
