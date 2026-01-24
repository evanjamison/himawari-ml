# src/himawari_ml/ingest/fetch_latest.py
from __future__ import annotations

import os
import time
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone

import requests

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

VERIFY_SSL = not IN_GHA  # 👈 disable SSL verification only in Actions


def _get(url: str, retries=4, timeout=30):
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
            log.warning(f"Fetch failed ({i+1}/{retries}) for {url} -> {e}")
            time.sleep(2 ** i)
    raise RuntimeError(f"Failed after {retries} retries: {url}")


def latest_timestamp() -> datetime:
    # Himawari updates every 10 minutes
    now = datetime.now(timezone.utc)
    minute = (now.minute // 10) * 10
    return now.replace(minute=minute, second=0, microsecond=0)


def fetch_frame(ts: datetime, out_dir: Path):
    y, m, d = ts.strftime("%Y"), ts.strftime("%m"), ts.strftime("%d")
    hms = ts.strftime("%H%M%S")

    rel = f"D531106/4d/550/{y}/{m}/{d}/{hms}_0_0.png"

    for base in BASES:
        url = f"{base}/{rel}"
        try:
            r = _get(url)
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"himawari_{hms}.png"
            path.write_bytes(r.content)
            log.info(f"Saved {path}")
            return True
        except Exception as e:
            log.warning(f"Base failed: {base} -> {e}")

    log.warning(f"All sources failed for {rel}")
    return False


def main(lookback_hours=72, max_frames=192):
    out = Path("data/raw")
    ts = latest_timestamp()

    saved = 0
    for _ in range(max_frames):
        if ts < datetime.now(timezone.utc) - timedelta(hours=lookback_hours):
            break

        if fetch_frame(ts, out):
            saved += 1

        ts -= timedelta(minutes=10)

    log.info(f"Done. Saved {saved} frames.")


if __name__ == "__main__":
    main()
