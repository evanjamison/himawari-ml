from __future__ import annotations
import numpy as np

def pseudo_cloud_mask(rgb01: np.ndarray, thresh: float = 0.7) -> np.ndarray:
    """Simple pseudo-label baseline:
    marks pixels as cloud when brightness is high.
    rgb01: float array (H,W,3) in [0,1]
    returns: uint8 mask (H,W) in {0,1}
    """
    brightness = rgb01.mean(axis=2)
    mask = (brightness >= thresh).astype("uint8")
    return mask
