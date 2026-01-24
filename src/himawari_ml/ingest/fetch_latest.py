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

import requests
from PIL import Image

from himawari_ml.utils.io import data_dir, ensure_dir
from himawari_ml.utils.logging import get_logger

logger = get_logger()

_ON_GHA = os.getenv("GITHUB_ACTIONS") == "true"

# Cache a "best" base once we find one that works (but still fallback if it flakes)
_RESOLVED_BASE: Optional[str] = None


@dataclass(frozen=True)
class HimawariConfig:
    product: str = os.getenv("HIMAWARI_PRODUCT", "D531106")
    level: str = os.getenv("HIMAWARI_LEVEL", "4d")  # 4d => 4x4 tiles
    tile_px: int = int(os.getenv("HIMAWARI_TILE_PX", "550"))

    timeout_s: int = int(os.getenv("HIMAWARI_TIMEOUT_S", "60"))
    retries: int = int(os.getenv("HIMAWARI_RETRIES", "4"))
    backoff_s: float = float(os.getenv("HIMAWARI_BACKOFF_S", "1.5"))


CFG = HimawariConfig()


def _default_bases() -> list[str]:
    """
    Mirror candidates. These can be overridden by env:
      HIMAWARI_BASES="http://a/.../img,http://b/.../img"
    """
    # Keep these as HTTP to avoid TLS/cert weirdness on runners.
    # If a candidate is wrong/unavailable, we'll just skip it.
    return [
        "http://himawari8-dl.nict.go.jp/himawari8/img",
        # Some environments/regions have better routing to alternate hostnames.
        # If these don't exist, they will simply fail and be skipped.
        "http://himawari8.nict.go.jp/himawari8/img",
        "http://himawari8-data.nict.go.jp/himawari8/img",
    ]


def _bases() -> list[str]:
    env = (os.getenv("HIMAWARI_BASES") or "").strip()
    if env:
        parts = [p.strip() for p in env.split(",") if p.strip()]
        if parts:
            return parts
    # Back-compat: allow single base via HIMAWARI_BASE
    single = (os.getenv("HIMAWARI_BASE") or "").strip()
    if single:
        return [single]
    return _default_bases()


def _grid_n(level: str) -> int:
    digits = "".join(c for c in level if c.isdigit())
    if not digits:
        raise ValueError(f"Bad level: {level}")
    return int(digits)


def _sleep_backoff(i: int) -> None:
    time.sleep(CFG.backoff_s * (2**i))


def _request_bytes_absolute(url: str) -> bytes:
    """
    Request a fully-formed URL with retries.
    """
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
            logger.warning(f"Fetch failed ({i+1}/{CFG.retries}) for {url} -> {e}. Sleeping {CFG.backoff_s*(2**i):.1f}s")
            _sleep_backoff(i)

    raise RuntimeError(f"Failed after {CFG.retries} retries: {url}") from last_err


def _request_bytes_path(path: str) -> tuple[bytes, str]:
    """
    Request a path using:
      1) cached resolved base if set
      2) otherwise try bases in order
    Returns (bytes, base_used).
    """
    global _RESOLVED_BASE

    candidates: list[str] = []
    if _RESOLVED_BASE:
        candidates.append(_RESOLVED_BASE)
    for b in _bases():
        if b not in candidates:
            candidates.append(b)

    last_err: Optional[Exception] = None
    for base in candidates:
        url = f"{base}{path}"
        try:
            raw = _request_bytes_absolute(url)
            # Update cache on success
            _RESOLVED_BASE = base
            return raw, base
        except Exception as e:
            last_err = e
            logger.warning(f"Base failed: {base} -> {e}")

    raise RuntimeError(f"All bases failed for path: {path}") from last_err


def _latest_timestamp_from_json() -> Optional[datetime]:
    bust = int(time.time())
    path = f"/{CFG.product}/latest.json?cb={bust}"

    try:
        raw, used = _request_bytes_path(path)
        if _ON_GHA:
            logger.warning(f"GITHUB_ACTIONS=true -> using base={used}")

        obj = json.loads(raw.decode("utf-8", errors="replace"))
        date_str = obj.get("date") or obj.get("datetime") or obj.get("time")
        if not date_str:
            logger.warning(f"latest.json missing date field. keys={list(obj.keys())}")
            return None

        date_str = date_str.replace("T", " ").replace("Z", "").strip()
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.replace(tzinfo=timezone.utc)
            except ValueError:
                continue

        logger.warning(f"Could not parse latest.json date: {date_str}")
        return None

    except Exception as e:
        logger.warning(f"Could not fetch/parse latest.json: {e}")
        return None


def _round_down(dt: datetime, minutes: int) -> datetime:
    dt = dt.replace(second=0, microsecond=0)
    minute = (dt.minute // minutes) * minutes
    return dt.replace(minute=minute)


def _candidate_times(anchor: Optional[datetime], lookback_hours: int, step_minutes: int) -> list[datetime]:
    if anchor is None:
        anchor = datetime.now(timezone.utc)
    anchor = _round_down(anchor, step_minutes)

    n_steps = max(1, int((lookback_hours * 60) // step_minutes))
    return [anchor - timedelta(minutes=step_minutes * k) for k in range(n_steps + 1)]


def _tile_path(ts: datetime, x: int, y: int) -> str:
    yyyy = ts.strftime("%Y")
    mm = ts.strftime("%m")
    dd = ts.strftime("%d")
    hhmmss = ts.strftime("%H%M%S")
    return f"/{CFG.product}/{CFG.level}/{CFG.tile_px}/{yyyy}/{mm}/{dd}/{hhmmss}_{x}_{y}.png"


def _download_and_stitch(ts: datetime) -> Image.Image:
    n = _grid_n(CFG.level)
    canvas = Image.new("RGB", (CFG.tile_px * n, CFG.tile_px * n), color=(0, 0, 0))

    for y in range(n):
        for x in range(n):
            path = _tile_path(ts, x, y)
            try:
                b, used = _request_bytes_path(path)
                tile = Image.open(BytesIO(b)).convert("RGB")
                canvas.paste(tile, (x * CFG.tile_px, y * CFG.tile_px))
            except Exception as e:
                logger.warning(f"Tile failed ({x},{y}) {path} -> {e}")

    return canvas


def _looks_like_real_image(img: Image.Image) -> bool:
    # Reject mosaics that are almost entirely black (common when timestamp has missing tiles).
    small = img.resize((128, 128))
    near_black = 0
    total = 128 * 128
    for r, g, b in small.getdata():
        if (r + g + b) < 15:
            near_black += 1
    return (near_black / total) < 0.98


def _out_name(ts: datetime) -> str:
    return f"himawari_{CFG.product}_{CFG.level}_{ts.strftime('%Y%m%d_%H%M%S')}Z.png"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(data_dir() / "raw"), help="Directory to save mosaics")
    ap.add_argument("--lookback-hours", type=int, default=48)
    ap.add_argument("--step-minutes", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=96)
    ap.add_argument("--no-latest-json", action="store_true")
    args = ap.parse_args()

    out_dir = ensure_dir(Path(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    if _ON_GHA:
        logger.warning(f"GITHUB_ACTIONS=true -> mirror fallback enabled. bases={_bases()}")

    anchor = None if args.no_latest_json else _latest_timestamp_from_json()
    candidates = _candidate_times(anchor, args.lookback_hours, args.step_minutes)

    saved = 0
    tried = 0
    for ts in candidates:
        if saved >= args.max_frames:
            break

        out_path = out_dir / _out_name(ts)
        if out_path.exists():
            continue

        tried += 1
        logger.info(f"[{tried}] Trying {ts.isoformat()} -> {out_path.name}")
        img = _download_and_stitch(ts)

        if not _looks_like_real_image(img):
            logger.info("Looks blank-ish; skipping.")
            continue

        img.save(out_path)
        saved += 1
        logger.info(f"Saved: {out_path}")

    logger.info(f"Done. saved={saved} tried={tried} out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

