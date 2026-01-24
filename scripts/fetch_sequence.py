from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from PIL import Image

from himawari_ml.ingest.fetch_latest import (
    CFG,
    _latest_timestamp_from_json,
    _candidate_times_fallback,
    _download_and_stitch,
    _looks_like_real_image,
)
from himawari_ml.utils.io import data_dir, ensure_dir
from himawari_ml.utils.logging import get_logger

logger = get_logger()


def main(
    n_frames: int = 12,
    step_minutes: int = 10,
    subdir: str = "sequence",
) -> int:
    """
    Fetch a short temporal sequence of Himawari frames.

    Saves images to: data/raw/<subdir>/
    """

    out_dir = ensure_dir(data_dir() / "raw" / subdir)

    ts0 = _latest_timestamp_from_json()
    candidates = (
        [ts0 - timedelta(minutes=step_minutes * i) for i in range(200)]
        if ts0
        else _candidate_times_fallback(n_back=200)
    )

    saved = 0

    for ts in candidates:
        if saved >= n_frames:
            break

        logger.info(f"Fetching frame at {ts.isoformat()}")

        try:
            img = _download_and_stitch(ts)
        except Exception as e:
            logger.warning(f"Failed to stitch at {ts}: {e}")
            continue

        if not _looks_like_real_image(img):
            logger.warning("Image looks blank; skipping")
            continue

        fname = (
            f"himawari_{CFG.product}_{CFG.level}_"
            f"{ts.strftime('%Y%m%d_%H%M%S')}Z.png"
        )
        out_path = out_dir / fname
        img.save(out_path)

        logger.info(f"Saved {out_path}")
        saved += 1

    if saved == 0:
        raise SystemExit("No valid frames fetched.")

    logger.info(f"Fetched {saved} frames → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
