from __future__ import annotations

import numpy as np
import torch

from himawari_ml.models.autoencoder import TinyAutoencoder
from himawari_ml.utils.io import data_dir, ensure_dir, outputs_dir
from himawari_ml.utils.logging import get_logger

logger = get_logger()

def main() -> int:
    x_path = data_dir() / "processed" / "latest_rgb01.npy"
    if not x_path.exists():
        raise SystemExit("No processed data. Run: python -m himawari_ml.preprocess.build_dataset")

    x = np.load(x_path)  # (H,W,3) in [0,1]
    x_t = torch.from_numpy(x).permute(2,0,1).unsqueeze(0)  # (1,3,H,W)

    model = TinyAutoencoder()
    model.eval()
    with torch.no_grad():
        emb = model(x_t).cpu().numpy().squeeze()

    out = ensure_dir(outputs_dir() / "metrics")
    np.save(out / "latest_embedding.npy", emb)
    logger.info(f"Saved embedding vector: outputs/metrics/latest_embedding.npy | dim={emb.shape[0]}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
