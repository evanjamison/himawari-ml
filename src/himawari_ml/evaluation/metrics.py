from __future__ import annotations
import numpy as np

def iou(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true = (y_true > 0).astype("uint8")
    y_pred = (y_pred > 0).astype("uint8")
    inter = (y_true & y_pred).sum()
    union = (y_true | y_pred).sum()
    return float(inter) / float(union + eps)
