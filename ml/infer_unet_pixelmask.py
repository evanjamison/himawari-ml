from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

import torch
import torch.nn as nn


# ---------------------------
# Model (must match train_unet_pixelmask.py)
# ---------------------------

def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, in_ch: int = 3, base_ch: int = 32):
        super().__init__()
        self.enc1 = conv_block(in_ch, base_ch)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = conv_block(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = conv_block(base_ch * 2, base_ch * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = conv_block(base_ch * 4, base_ch * 8)

        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec3 = conv_block(base_ch * 8, base_ch * 4)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = conv_block(base_ch * 4, base_ch * 2)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = conv_block(base_ch * 2, base_ch)

        # NOTE: name is "out" in trainer
        self.out = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        d3 = self.up3(e4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out(d1)


# ---------------------------
# IO helpers
# ---------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def list_images(frames: Path, recursive: bool) -> List[Path]:
    if frames.is_file():
        return [frames]
    if recursive:
        out: List[Path] = []
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            out.extend(frames.rglob(ext))
        return sorted(out)
    return sorted(list(frames.glob("*.png")) + list(frames.glob("*.jpg")) + list(frames.glob("*.jpeg")))


def load_rgb(path: Path, image_size: int) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    im = im.resize((image_size, image_size), resample=Image.BILINEAR)
    arr = np.array(im).astype(np.float32) / 255.0  # H W 3 in [0,1]
    return arr


def save_mask_png(mask01: np.ndarray, out_path: Path) -> None:
    # mask01 is HxW float in {0,1}
    im = Image.fromarray((mask01 * 255.0).astype(np.uint8), mode="L")
    im.save(out_path)


# ---------------------------
# Main
# ---------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Infer pixel-mask U-Net (seg mode).")
    ap.add_argument("--frames", type=str, required=True, help="Frames directory or single image.")
    ap.add_argument("--outdir", type=str, required=True, help="Output directory.")
    ap.add_argument("--ckpt", type=str, required=True, help="Checkpoint path (unet_pixelmask_latest.pt).")

    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--recursive", action="store_true")
    args = ap.parse_args()

    frames = Path(args.frames)
    outdir = Path(args.outdir)
    ckpt_path = Path(args.ckpt)

    masks_dir = outdir / "masks_unet"
    ensure_dir(masks_dir)

    print(f"[INFER-SEG] frames={frames}")
    print(f"[INFER-SEG] outdir={outdir}")
    print(f"[INFER-SEG] ckpt={ckpt_path}")
    print(f"[INFER-SEG] image_size={args.image_size} threshold={args.threshold} device={args.device}")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise RuntimeError("Checkpoint does not look like train_unet_pixelmask output (missing state_dict).")

    cfg = ckpt.get("config", {})
    base_ch = int(cfg.get("base_ch", 32))
    print(f"[INFER-SEG] base_ch(from ckpt config)={base_ch}")

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    model = UNet(in_ch=3, base_ch=base_ch).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    files = list_images(frames, recursive=args.recursive)
    if not files:
        raise RuntimeError(f"No images found under: {frames}")

    rows = []
    for p in files:
        x = load_rgb(p, args.image_size)  # H W 3 float
        xt = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(device)  # 1 3 H W

        with torch.no_grad():
            logits = model(xt)
            prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()  # H W

        mask = (prob >= float(args.threshold)).astype(np.float32)  # H W

        out_mask_path = masks_dir / f"{p.stem}_mask.png"
        save_mask_png(mask, out_mask_path)

        white_frac = float(mask.mean())
        rows.append(
            {
                "filename": p.name,
                "mask_path": str(out_mask_path),
                "threshold": float(args.threshold),
                "cloud_fraction": white_frac,
                "white_pixels": int(mask.sum()),
                "total_pixels": int(mask.size),
            }
        )

    csv_path = outdir / "cloud_metrics_unet.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["filename", "mask_path", "threshold", "cloud_fraction", "white_pixels", "total_pixels"],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[INFER-SEG] wrote masks -> {masks_dir}")
    print(f"[INFER-SEG] wrote metrics -> {csv_path}")
    print(f"[INFER-SEG] count={len(rows)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
