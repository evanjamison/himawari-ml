from __future__ import annotations

from pathlib import Path
import json
import numpy as np
from PIL import Image

from himawari_ml.utils.config import SETTINGS
from himawari_ml.utils.io import data_dir, ensure_dir
from himawari_ml.utils.logging import get_logger
from himawari_ml.preprocess.image_cleaning import resize_and_normalize
from himawari_ml.preprocess.cloud_heuristics import pseudo_cloud_mask

logger = get_logger()

def main() -> int:
    raw_dir = data_dir() / "raw"
    if not raw_dir.exists():
        raise SystemExit("No raw data found. Run: python -m himawari_ml.ingest.fetch_latest")

    interim_dir = ensure_dir(data_dir() / "interim")
    processed_dir = ensure_dir(data_dir() / "processed")

    # Take the most recent file by modified time
    raw_files = sorted(raw_dir.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not raw_files:
        raise SystemExit("No PNG files in data/raw. Provide a sample URL that returns an image.")

    latest = raw_files[0]
    img = Image.open(latest).convert("RGB")
    rgb01, img_r = resize_and_normalize(img, SETTINGS.image_size)

    mask = pseudo_cloud_mask(rgb01, thresh=0.7)

    # Save interim artifacts
    img_out = interim_dir / "latest_resized.png"
    Image.fromarray((rgb01 * 255).astype("uint8")).save(img_out)

    mask_out = interim_dir / "latest_mask.png"
    Image.fromarray((mask * 255).astype("uint8")).save(mask_out)

    # Save processed arrays (numpy)
    x_path = processed_dir / "latest_rgb01.npy"
    y_path = processed_dir / "latest_mask.npy"
    np.save(x_path, rgb01)
    np.save(y_path, mask)

    meta = {
        "source_file": str(latest.name),
        "image_size": SETTINGS.image_size,
        "mask_method": "brightness_threshold",
        "mask_thresh": 0.7,
    }
    (processed_dir / "latest_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    cloud_pct = float(mask.mean()) * 100.0
    logger.info(f"Built dataset from {latest.name} | cloud_coverage={cloud_pct:.2f}%")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
