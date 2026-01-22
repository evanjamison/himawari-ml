from __future__ import annotations
import torch
import torch.nn as nn

class TinyUNet(nn.Module):
    """Small starter U-Net-like model for segmentation (placeholder).
    Replace with a proper U-Net once you're ready.
    """
    def __init__(self, in_ch: int = 3, out_ch: int = 1):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
        )
        self.head = nn.Conv2d(16, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.enc(x)
        return self.head(z)
