from __future__ import annotations
import torch
import torch.nn as nn

class TinyAutoencoder(nn.Module):
    def __init__(self, in_ch: int = 3, latent: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.to_latent = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(32, latent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        z = self.to_latent(z).flatten(1)
        return self.fc(z)
