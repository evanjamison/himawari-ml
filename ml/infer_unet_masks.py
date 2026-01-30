from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Repo root detection
# -------------------------
REPO_ROOT = Path(__file__).resolve()
while not (REPO_ROOT / "pyproject.toml").exists() and not (REPO_ROOT / ".git").exists():
    if REPO_ROOT == REPO_ROOT.parent:
        break
    REPO_ROOT = REPO_ROOT.parent


# -------------------------
# Blocks (match checkpoint BN key patterns like enc2.4.running_mean)
# -------------------------
def double_conv(in_ch: int, out_ch: int) -> nn.Sequential:
    # 0 conv, 1 bn, 2 relu, 3 conv, 4 bn, 5 relu
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def concat_with_pad(x_up: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
    diffY = x_skip.size(2) - x_up.size(2)
    diffX = x_skip.size(3) - x_up.size(3)
    x_up = F.pad(x_up, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
    return torch.cat([x_skip, x_up], dim=1)


# -------------------------
# 3-level U-Net (enc1-enc3, bottleneck, up1-up3, dec1-dec3, head)
# This matches your ckpt: no enc4/dec4/up4, and uses head.*
# -------------------------
class UNet3(nn.Module):
    def __init__(self, in_ch: int, base: int, out_ch: int = 1):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        self.enc1 = double_conv(in_ch, base)         # 32
        self.enc2 = double_conv(base, base * 2)      # 64
        self.enc3 = double_conv(base * 2, base * 4)  # 128

        self.bottleneck = double_conv(base * 4, base * 8)  # 256

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
        self.dec3 = double_conv(base * 8, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.dec2 = double_conv(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.dec1 = double_conv(base * 2, base)

        # checkpoint uses head.weight/head.bias
        self.head = nn.Conv2d(base, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        d3 = concat_with_pad(d3, e3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = concat_with_pad(d2, e2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = concat_with_pad(d1, e1)
        d1 = self.dec1(d1)

        return self.head(d1)


# -------------------------
# 4-level U-Net (in case you later switch checkpoints)
# -------------------------
class UNet4(nn.Module):
    def __init__(self, in_ch: int, base: int, out_ch: int = 1):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        self.enc1 = double_conv(in_ch, base)
        self.enc2 = double_conv(base, base * 2)
        self.enc3 = double_conv(base * 2, base * 4)
        self.enc4 = double_conv(base * 4, base * 8)

        self.bottleneck = double_conv(base * 8, base * 16)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, 2)
        self.dec4 = double_conv(base * 16, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
        self.dec3 = double_conv(base * 8, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.dec2 = double_conv(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.dec1 = double_conv(base * 2, base)

        self.head = nn.Conv2d(base, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = concat_with_pad(d4, e4)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = concat_with_pad(d3, e3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = concat_with_pad(d2, e2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = concat_with_pad(d1, e1)
        d1 = self.dec1(d1)

        return self.head(d1)


# -------------------------
# IO helpers
# -------------------------
def load_rgb(p: Path, image_size: int) -> np.ndarray:
    im = Image.open(p).convert("RGB")

    # Pillow compatibility (older versions lack Image.Resampling)
    Resampling = getattr(Image, "Resampling", None)
    bilinear = Resampling.BILINEAR if Resampling is not None else Image.BILINEAR

    if im.size != (image_size, image_size):
        im = im.resize((image_size, image_size), resample=bilinear)

    arr = np.asarray(im, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    return arr


def iter_images(frames_dir: Path, recursive: bool) -> List[Path]:
    exts = (".png", ".jpg", ".jpeg", ".webp")
    if recursive:
        paths = [p for p in frames_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    else:
        paths = [p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(paths)


def to_repo_rel(p: Path) -> str:
    try:
        return p.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except Exception:
        return p.resolve().as_posix()


# -------------------------
# Checkpoint loading
# -------------------------
def extract_state_dict(ckpt_obj) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt_obj, dict):
        for k in ("model", "state_dict", "model_state_dict"):
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                return ckpt_obj[k]
        return ckpt_obj
    raise ValueError("Unrecognized checkpoint format")


def normalize_keys(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]

        # common variants
        nk = nk.replace("final", "head").replace("final_conv", "head")
        nk = nk.replace("outc", "head")

        nk = nk.replace("upconv4", "up4").replace("upconv3", "up3").replace("upconv2", "up2").replace("upconv1", "up1")
        nk = nk.replace("up_conv4", "up4").replace("up_conv3", "up3").replace("up_conv2", "up2").replace("up_conv1", "up1")

        out[nk] = v
    return out


def infer_base_and_depth(sd: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    """
    base = out_channels of enc1.0.weight
    depth = 4 if enc4.0.weight exists else 3
    """
    w = sd.get("enc1.0.weight")
    if w is None:
        raise RuntimeError("Cannot infer base channels: missing enc1.0.weight in checkpoint")
    base = int(w.shape[0])

    depth = 4 if "enc4.0.weight" in sd else 3
    return base, depth


# -------------------------
# Main
# -------------------------
def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument("--frames", default="", help="Directory containing input frames/images")
    ap.add_argument("--recursive", action="store_true", help="Recursively search --frames directory")
    ap.add_argument("--metrics-csv", default="", help="CSV with 'relpath' (optionally 'img_ok'). If set, uses this mode.")

    ap.add_argument("--ckpt", required=True, help="Path to trained .pt checkpoint")
    ap.add_argument("--outdir", required=True, help="Output directory for Phase 3 artifacts")
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--device", default="", help="cpu|cuda (default auto)")

    args = ap.parse_args()

    metrics_csv = str(args.metrics_csv).strip()
    frames_arg = str(args.frames).strip()
    if not metrics_csv and not frames_arg:
        raise SystemExit("ERROR: Provide either --frames DIR (Phase 3) or --metrics-csv CSV")

    out_root = Path(args.outdir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    out_masks = out_root / "masks_unet"
    out_masks.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint first, infer architecture
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = (REPO_ROOT / ckpt_path).resolve()
    if not ckpt_path.exists():
        raise SystemExit(f"ERROR: --ckpt not found: {ckpt_path}")

    ckpt_obj = torch.load(ckpt_path, map_location=device)
    sd = normalize_keys(extract_state_dict(ckpt_obj))
    base, depth = infer_base_and_depth(sd)

    # Collect images + build df
    if metrics_csv:
        mp = Path(metrics_csv)
        if not mp.is_absolute():
            mp = (REPO_ROOT / mp).resolve()
        if not mp.exists():
            raise SystemExit(f"ERROR: metrics CSV not found: {mp}")

        df = pd.read_csv(mp)
        if "relpath" not in df.columns:
            raise SystemExit("ERROR: metrics CSV must include 'relpath' column")
        if "img_ok" in df.columns:
            df = df[df["img_ok"] == 1].copy()

        df["relpath"] = df["relpath"].astype(str).str.replace("\\", "/", regex=False)

        img_paths: List[Path] = []
        for rp in df["relpath"].tolist():
            p = Path(rp)
            if not p.is_absolute():
                p = (REPO_ROOT / p).resolve()
            img_paths.append(p)
    else:
        frames_dir = Path(frames_arg).expanduser()
        if not frames_dir.is_absolute():
            frames_dir = (REPO_ROOT / frames_dir).resolve()
        if not frames_dir.exists() or not frames_dir.is_dir():
            raise SystemExit(f"ERROR: --frames directory not found: {frames_dir}")

        img_paths = iter_images(frames_dir, bool(args.recursive))
        if not img_paths:
            raise SystemExit(f"ERROR: No images found in {frames_dir} (recursive={bool(args.recursive)})")
        df = pd.DataFrame({"relpath": [to_repo_rel(p) for p in img_paths]})

    # Build model that matches ckpt
    if depth == 3:
        model = UNet3(in_ch=3, base=base, out_ch=1).to(device)
    else:
        model = UNet4(in_ch=3, base=base, out_ch=1).to(device)

    # Strict load SHOULD succeed now
    model.load_state_dict(sd, strict=True)
    model.eval()

    # Inference
    mask_paths: List[str] = []
    cloud_fracs: List[float] = []

    with torch.no_grad():
        for img_path in img_paths:
            if not img_path.exists():
                raise SystemExit(f"ERROR: Image not found: {img_path}")

            x = load_rgb(img_path, int(args.image_size))
            xt = torch.from_numpy(x).unsqueeze(0).to(device)

            logits = model(xt)  # (1,1,H,W)
            prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
            # --- disk mask (inference-time) ---
            # x is CHW in [0,1]
            disk = (x.mean(axis=0) > 0.02).astype(np.float32)   # 1 inside Earth, 0 in space
            prob = prob * disk                                  # zero out space predictions

            mask = (prob >= float(args.threshold)).astype(np.uint8) * 255
            mask_out = out_masks / f"{img_path.stem}_mask.png"
            Image.fromarray(mask).save(mask_out)

            mask_paths.append(to_repo_rel(mask_out))
            cloud_fracs.append(float((mask > 0).mean()))

    df["mask_path"] = mask_paths
    df["cloud_fraction_unet"] = cloud_fracs

    out_csv = out_root / "cloud_metrics_unet.csv"
    df.to_csv(out_csv, index=False)

    print(f"Loaded checkpoint with base={base}, depth={depth}")
    print(f"Wrote: {out_csv}")
    print(f"Wrote masks: {out_masks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

