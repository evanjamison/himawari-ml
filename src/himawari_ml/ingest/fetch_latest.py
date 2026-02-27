# src/himawari_ml/ingest/fetch_latest.py
from __future__ import annotations

import bz2
import ftplib
import io
import logging
import os
import struct
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from himawari_ml.utils.paths import raw_latest_dir, frame_filename

log = logging.getLogger("fetch_latest")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")

IN_GHA = os.getenv("GITHUB_ACTIONS", "false").lower() == "true"

# ---------------------------------------------------------------------------
# JAXA P-Tree FTP credentials
# In GitHub Actions: store as repository secrets PTREE_USER and PTREE_PASS
# and add to the ingest workflow env block.
# ---------------------------------------------------------------------------
FTP_HOST = os.getenv("PTREE_HOST", "ftp.ptree.jaxa.jp")
FTP_USER = os.getenv("PTREE_USER", "")
FTP_PASS = os.getenv("PTREE_PASS", "")

# ---------------------------------------------------------------------------
# Limits (override via env)
# ---------------------------------------------------------------------------
TARGET_SAVED          = int(os.getenv("TARGET_SAVED",    "24" if IN_GHA else "48"))
MAX_FRAMES_DEFAULT    = int(os.getenv("MAX_FRAMES",      "96" if IN_GHA else "192"))
LOOKBACK_HOURS_DEFAULT = int(os.getenv("LOOKBACK_HOURS", "48" if IN_GHA else "72"))
DEFAULT_IMAGE_SIZE    = int(os.getenv("IMAGE_SIZE",      "256"))

# ---------------------------------------------------------------------------
# HSD constants
# Full-disk (FLDK) split into 10 vertical segments: S0110 ... S1010
# B01=blue (1km/R10), B02=green (1km/R10), B03=red (500m/R05)
# ---------------------------------------------------------------------------
SEGMENTS = 10
BANDS = [
    ("B01", "R10"),   # blue
    ("B02", "R10"),   # green
    ("B03", "R05"),   # red — higher res, downsampled to match B01/B02
]


@dataclass
class FetchCfg:
    retries: int = 3
    backoff_base_s: float = 2.0
    ftp_timeout_s: int = 30


# ---------------------------------------------------------------------------
# FTP helpers
# ---------------------------------------------------------------------------

def _connect(cfg: FetchCfg) -> ftplib.FTP:
    """Open and return an authenticated FTP connection to P-Tree."""
    if not FTP_USER or not FTP_PASS:
        raise RuntimeError(
            "PTREE_USER and PTREE_PASS environment variables are not set. "
            "Add them as GitHub Actions secrets and reference them in your workflow env block."
        )
    for attempt in range(cfg.retries):
        try:
            ftp = ftplib.FTP(FTP_HOST, timeout=cfg.ftp_timeout_s)
            ftp.login(FTP_USER, FTP_PASS)
            ftp.set_pasv(True)
            log.info(f"FTP connected: {FTP_HOST} as {FTP_USER}")
            return ftp
        except Exception as e:
            log.warning(f"FTP connect attempt {attempt+1}/{cfg.retries} failed: {e}")
            if attempt < cfg.retries - 1:
                time.sleep(cfg.backoff_base_s * (2 ** attempt))
    raise RuntimeError(f"Could not connect to {FTP_HOST} after {cfg.retries} attempts")


def _ftp_read(ftp: ftplib.FTP, remote_path: str, cfg: FetchCfg) -> bytes | None:
    """Download a single file from FTP into memory. Returns None if not found."""
    for attempt in range(cfg.retries):
        try:
            buf = io.BytesIO()
            ftp.retrbinary(f"RETR {remote_path}", buf.write)
            return buf.getvalue()
        except ftplib.error_perm as e:
            if "550" in str(e):
                return None   # file not found — not an error
            log.warning(f"FTP perm error {remote_path}: {e}")
            return None
        except Exception as e:
            log.warning(f"FTP read attempt {attempt+1} failed for {remote_path}: {e}")
            if attempt < cfg.retries - 1:
                time.sleep(cfg.backoff_base_s * (2 ** attempt))
    return None


# ---------------------------------------------------------------------------
# HSD parser
# Himawari Standard Data format reference:
# "Himawari Standard Data User's Manual" (JMA/JAXA)
# ---------------------------------------------------------------------------

def _parse_hsd_segment(data: bytes) -> np.ndarray | None:
    """
    Parse a decompressed HSD segment and return float32 reflectance [0,1].

    HSD block layout — each block starts with:
      1 byte  : block number
      2 bytes : block length (uint16, includes these 3 header bytes)

    Known blocks and their total sizes (bytes):
      Block 1  : 282  — basic info (lines, columns)
      Block 2  : 50   — data info
      Block 3  : 50   — projection
      Block 4  : 127  — navigation
      Block 5  : 115  — calibration (slope at +3, intercept at +11)
      Block 6  : 17   — inter-calibration
      Block 7  : 59   — segment info
      Block 8  : 33   — navigation correction
      Block 9  : 13   — observation time
      Block 10 : 13   — error info
      Block 11 : 13   — spare
      Data block follows immediately after block 11
    """
    try:
        # Walk blocks dynamically using the block-length field
        # to find the true data start — more robust than hardcoded offsets
        offset = 0
        block_data: dict[int, bytes] = {}

        for _ in range(11):
            if offset + 3 > len(data):
                break
            block_num = struct.unpack_from("B", data, offset)[0]
            block_len = struct.unpack_from(">H", data, offset + 1)[0]
            if block_len == 0:
                break
            block_data[block_num] = data[offset: offset + block_len]
            offset += block_len

        # Data starts immediately after block 11
        data_offset = offset

        # Parse block 1 for dimensions
        b1 = block_data.get(1, b"")
        if len(b1) < 50:
            log.warning("HSD: block 1 too short")
            return None

        endian = ">" if struct.unpack_from("B", b1, 5)[0] == 0 else "<"
        nlines = struct.unpack_from(f"{endian}H", b1, 6)[0]
        ncols  = struct.unpack_from(f"{endian}H", b1, 8)[0]

        # Parse block 5 for calibration
        b5 = block_data.get(5, b"")
        if len(b5) >= 27:
            slope     = struct.unpack_from(f"{endian}d", b5, 3)[0]
            intercept = struct.unpack_from(f"{endian}d", b5, 11)[0]
        else:
            # Fallback: use linear scaling without calibration
            slope, intercept = 1.0 / 65535.0, 0.0

        # Align data offset to 2-byte boundary
        remaining = len(data) - data_offset
        if remaining % 2 != 0:
            data_offset += 1

        raw = np.frombuffer(data[data_offset:], dtype=np.dtype(f"{endian}u2"))

        if raw.size < nlines * ncols:
            log.warning(f"HSD: expected {nlines*ncols} pixels, got {raw.size} (offset={data_offset})")
            return None

        raw = raw[:nlines * ncols].reshape(nlines, ncols).astype(np.float32)

        invalid     = (raw == 0) | (raw == 65535)
        reflectance = np.clip(raw * slope + intercept, 0.0, 1.0)
        reflectance[invalid] = 0.0

        return reflectance

    except Exception as e:
        log.warning(f"HSD parse error: {e}")
        return None


def _remote_path(ts: datetime, band: str, res: str, seg: int) -> str:
    """
    Build JAXA P-Tree FTP path.
    /jma/hsd/YYYYMM/DD/HH/HS_H09_YYYYMMDD_HHmm_Bxx_FLDK_Rxx_Sxxx0.DAT.bz2

    Notes:
    - Directory uses HH (hour only), filename uses HHmm (hour + minute)
    - Himawari-9 (H09) replaced Himawari-8 (H08) in December 2022
    """
    ym   = ts.strftime("%Y%m")
    d    = ts.strftime("%d")
    ymd  = ts.strftime("%Y%m%d")
    hhmm = ts.strftime("%H%M")
    hh   = ts.strftime("%H")        # directory uses hour only
    seg_s = f"S{seg:02d}{SEGMENTS:02d}"
    fname = f"HS_H09_{ymd}_{hhmm}_{band}_FLDK_{res}_{seg_s}.DAT.bz2"
    return f"/jma/hsd/{ym}/{d}/{hh}/{fname}"


# ---------------------------------------------------------------------------
# Frame fetch
# ---------------------------------------------------------------------------

def fetch_frame(
    ts: datetime,
    out_dir: Path,
    image_size: int,
    cfg: FetchCfg,
    ftp: ftplib.FTP,
) -> bool:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / frame_filename(ts)

    if out_path.exists():
        log.info(f"Already exists, skipping: {out_path.name}")
        return True

    band_arrays: dict[str, np.ndarray] = {}

    for band, res in BANDS:
        segments: list[np.ndarray] = []

        for seg in range(1, SEGMENTS + 1):
            remote = _remote_path(ts, band, res, seg)
            compressed = _ftp_read(ftp, remote, cfg)

            if compressed is None:
                log.warning(f"Missing: {remote}")
                break

            try:
                raw_bytes = bz2.decompress(compressed)
            except Exception as e:
                log.warning(f"bz2 decompress failed {remote}: {e}")
                break

            arr = _parse_hsd_segment(raw_bytes)
            if arr is None:
                break

            segments.append(arr)

        if len(segments) != SEGMENTS:
            log.warning(f"Band {band} incomplete ({len(segments)}/{SEGMENTS}) for {ts.isoformat()}")
            return False

        band_arrays[band] = np.concatenate(segments, axis=0)

    if len(band_arrays) != len(BANDS):
        return False

    # Align resolutions: B03 (500m) → downsample to B01/B02 (1km) size
    h_ref = band_arrays["B01"].shape[0]
    w_ref = band_arrays["B01"].shape[1]

    def _resize_band(arr: np.ndarray) -> np.ndarray:
        if arr.shape == (h_ref, w_ref):
            return arr
        pil = Image.fromarray((arr * 255).astype(np.uint8))
        pil = pil.resize((w_ref, h_ref), Image.BILINEAR)
        return np.array(pil).astype(np.float32) / 255.0

    r = _resize_band(band_arrays["B03"])
    g = _resize_band(band_arrays["B02"])
    b = _resize_band(band_arrays["B01"])

    rgb = np.clip(np.stack([r, g, b], axis=-1) * 255, 0, 255).astype(np.uint8)

    img  = Image.fromarray(rgb)
    w, h = img.size
    side = min(w, h)
    img  = img.crop(((w-side)//2, (h-side)//2, (w+side)//2, (h+side)//2))
    img  = img.resize((image_size, image_size), Image.BILINEAR)
    img.save(out_path)

    log.info(f"Saved {out_path.name}")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def latest_timestamp_10min() -> datetime:
    now    = datetime.now(timezone.utc)
    minute = (now.minute // 10) * 10
    return now.replace(minute=minute, second=0, microsecond=0)


def main(
    lookback_hours: int = LOOKBACK_HOURS_DEFAULT,
    max_frames: int     = MAX_FRAMES_DEFAULT,
    image_size: int     = DEFAULT_IMAGE_SIZE,
) -> int:
    out = (
        Path(os.getenv("HIMAWARI_OUT_DIR", "")).expanduser()
        if os.getenv("HIMAWARI_OUT_DIR")
        else raw_latest_dir()
    )

    ts     = latest_timestamp_10min()
    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    cfg    = FetchCfg()

    log.info(f"JAXA P-Tree ingest: lookback={lookback_hours}h target={TARGET_SAVED} out={out}")

    try:
        ftp = _connect(cfg)
    except RuntimeError as e:
        log.error(str(e))
        log.info(f"Done. Saved 0 frames. out={out} target_saved={TARGET_SAVED}")
        return 0

    saved = checked = 0

    try:
        while checked < max_frames and ts >= cutoff and saved < TARGET_SAVED:
            checked += 1
            try:
                if fetch_frame(ts, out, image_size=image_size, cfg=cfg, ftp=ftp):
                    saved += 1
            except ftplib.all_errors as e:
                log.warning(f"FTP error on {ts.isoformat()}, reconnecting: {e}")
                try:
                    ftp.quit()
                except Exception:
                    pass
                try:
                    ftp = _connect(cfg)
                except RuntimeError:
                    log.error("Could not reconnect, stopping.")
                    break
            ts -= timedelta(minutes=10)
    finally:
        try:
            ftp.quit()
        except Exception:
            pass

    log.info(
        f"Done. Saved {saved} frames (checked {checked}). "
        f"out={out} target_saved={TARGET_SAVED}"
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
