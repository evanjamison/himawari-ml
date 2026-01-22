from __future__ import annotations

from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA

from himawari_ml.utils.io import outputs_dir, ensure_dir
from himawari_ml.utils.logging import get_logger

logger = get_logger()

def main() -> int:
    emb_path = outputs_dir() / "metrics" / "latest_embedding.npy"
    if not emb_path.exists():
        raise SystemExit("No embedding found. Run: python -m himawari_ml.train.train_autoencoder")

    emb = np.load(emb_path).reshape(1, -1)  # single sample
    # With one sample PCA isn't meaningful; this script is a scaffold.
    # Once you accumulate embeddings over time, stack them and PCA will work.
    pca = PCA(n_components=min(emb.shape[1], 2))
    pca.fit(emb)

    out = ensure_dir(outputs_dir() / "metrics")
    np.save(out / "pca_components.npy", pca.components_)
    np.save(out / "pca_explained_variance_ratio.npy", pca.explained_variance_ratio_)

    logger.info("Saved PCA scaffold outputs (needs multiple samples to be meaningful).")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
