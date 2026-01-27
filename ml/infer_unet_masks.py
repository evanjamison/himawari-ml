from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Robust repo root detection
# -------------------------
REPO_ROOT = Path(__file__).resolve()
while not (REPO_ROOT / "pyproject.toml").exists() and not (REPO_ROOT / ".git").exists():
    if REPO_ROOT == REPO_ROOT.parent:
        break
    REPO_ROOT = REPO_ROOT.parent


# -------------------------
# Model (match train_unet.py)
# -------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # pad if odd dims
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNetSmall(nn.Module):
    def __init__(self, in_ch: int = 3, base: int = 32, out_ch: int = 1):
        super().__init__()
        self.inc = DoubleConv(in_ch, base)
        self.down1 = Down(base, base * 2)
        self.down2 = Down(base * 2, base * 4)
        self.down3 = Down(base * 4, base * 8)
        self.up1 = Up(base * 8, base * 4)
        self.up2 = Up(base * 4, base * 2)
        self.up3 = Up(base * 2, base)
        self.outc = nn.Conv2d(base, out_ch, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.outc(x)


# -------------------------
# Inference
# -------------------------
def load_rgb(p: Path, image_size: int) -> np.ndarray:
    im = Image.open(p).convert("RGB")
    if im.size != (image_size, image_size):
        im = im.resize((image_size, image_size), resample=Image.Resampling.BILINEAR)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    # CHW
    arr = np.transpose(arr, (2, 0, 1))
    return arr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to trained .pt checkpoint from train_unet.py")
    ap.add_argument("--metrics-csv", default=str(REPO_ROOT / "out" / "cloud_metrics.csv"))
    ap.add_argument("--outdir", default=str(REPO_ROOT / "out"))
    ap.add_argument("--image-size", type=int, default=256, help="Must match training image size")
    ap.add_argument("--base-ch", type=int, default=32, help="Must match training base channels")
    ap.add_argument("--threshold", type=float, default=0.5, help="Sigmoid threshold for binary mask")
    ap.add_argument("--device", default="", help="cpu|cuda (default auto)")
    args = ap.parse_args()

    out_root = Path(args.outdir).resolve()
    out_masks = out_root / "masks_unet"
    out_masks.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.metrics_csv)
    if "img_ok" in df.columns:
        df = df[df["img_ok"] == 1].copy()

    # Resolve relpaths to absolute files
    df["relpath"] = df["relpath"].astype(str).str.replace("\\", "/", regex=False)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device)
    model = UNetSmall(in_ch=3, base=int(args.base_ch), out_ch=1).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    new_mask_paths = []
    cloud_fracs = []

    with torch.no_grad():
        for rp in df["relpath"].tolist():
            img_path = (REPO_ROOT / rp).resolve()
            x = load_rgb(img_path, args.image_size)
            xt = torch.from_numpy(x).unsqueeze(0).to(device)

            logits = model(xt)
            prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

            mask = (prob >= float(args.threshold)).astype(np.uint8) * 255

            stem = Path(rp).stem
            mask_path = out_masks / f"{stem}_mask.png"
            Image.fromarray(mask).save(mask_path)

            new_mask_paths.append(mask_path.relative_to(REPO_ROOT).as_posix())
            cloud_fracs.append(float((mask > 0).mean()))

    df["mask_path"] = new_mask_paths
    df["cloud_fraction_unet"] = cloud_fracs

    out_csv = out_root / "cloud_metrics_unet.csv"
    df.to_csv(out_csv, index=False)

    print(f"Wrote: {out_csv}")
    print(f"Wrote masks: {out_masks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
