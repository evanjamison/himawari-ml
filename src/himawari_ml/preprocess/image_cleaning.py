from __future__ import annotations

from PIL import Image
import numpy as np

def resize_and_normalize(img: Image.Image, size: int) -> tuple[np.ndarray, Image.Image]:
    img_r = img.resize((size, size), resample=Image.BICUBIC).convert("RGB")
    arr = np.asarray(img_r).astype("float32") / 255.0  # (H,W,3) in [0,1]
    return arr, img_r
