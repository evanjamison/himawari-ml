from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path


DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
SAMPLE_DIR = DATA_DIR / "sample"
OUT_DIR = Path("out")


def utc_today_date() -> date:
    return datetime.now(timezone.utc).date()


def ymd(d: date) -> str:
    return d.strftime("%Y-%m-%d")


def raw_day_dir(d: date | None = None) -> Path:
    """data/raw/YYYY-MM-DD/"""
    d = d or utc_today_date()
    return RAW_DIR / ymd(d)


def raw_latest_dir(d: date | None = None) -> Path:
    """data/raw/YYYY-MM-DD/latest/"""
    return raw_day_dir(d) / "latest"


def raw_burst_dir(label: str, d: date | None = None) -> Path:
    """data/raw/YYYY-MM-DD/burst_<label>/"""
    d = d or utc_today_date()
    label = label.strip().replace(" ", "_") or "01"
    return raw_day_dir(d) / f"burst_{label}"


def sample_sequence_dir(name: str = "sequenceA") -> Path:
    """data/sample/raw/<name>/"""
    name = name.strip().replace(" ", "_") or "sequenceA"
    return SAMPLE_DIR / "raw" / name


def frame_filename(ts_utc: datetime) -> str:
    """himawari_YYYYMMDDTHHMMSSZ.png (UTC)"""
    if ts_utc.tzinfo is None:
        ts_utc = ts_utc.replace(tzinfo=timezone.utc)
    ts_utc = ts_utc.astimezone(timezone.utc)
    return f"himawari_{ts_utc.strftime('%Y%m%dT%H%M%SZ')}.png"
