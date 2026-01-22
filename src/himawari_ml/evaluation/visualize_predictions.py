from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from himawari_ml.utils.io import data_dir, ensure_dir, outputs_dir
from himawari_ml.utils.logging import get_logger

logger = get_logger()

def main() -> int:
    interim = data_dir() / "interim"
    img_p = interim / "latest_resized.png"
    mask_p = interim / "latest_mask.png"
    if not img_p.exists() or not mask_p.exists():
        raise SystemExit("Missing interim artifacts. Run: python -m himawari_ml.preprocess.build_dataset")

    img = np.asarray(Image.open(img_p).convert("RGB"))
    mask = np.asarray(Image.open(mask_p).convert("L"))

    out_fig = ensure_dir(outputs_dir() / "figures") / "latest_overlay.png"

    plt.figure()
    plt.imshow(img)
    plt.imshow(mask, alpha=0.35)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_fig, dpi=200, bbox_inches="tight")
    plt.close()

    # simple report
    cloud_pct = (mask > 0).mean() * 100.0
    out_rep = ensure_dir(outputs_dir() / "reports") / "latest_report.txt"
    out_rep.write_text(f"Cloud coverage (% pixels): {cloud_pct:.2f}\n", encoding="utf-8")

    logger.info(f"Wrote figure: {out_fig}")
    logger.info(f"Wrote report: {out_rep}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
