from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional

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
# Helpers
# -------------------------
def load_rgb(p: Path, image_size: int) -> np.ndarray:
    im = Image.open(p).convert("RGB")
    if im.size != (image_size, image_size):
        im = im.resize((image_size, image_size), resample=Image.Resampling.BILINEAR)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    # CHW
    arr = np.transpose(arr, (2, 0, 1))
    return arr


def _iter_images(frames_dir: Path, recursive: bool) -> List[Path]:
    exts = (".png", ".jpg", ".jpeg", ".webp")
    if recursive:
        paths = [p for p in frames_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    else:
        paths = [p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(paths)


def _rel_to_repo(p: Path) -> str:
    """Return a repo-relative POSIX path if possible; otherwise just a POSIX absolute path."""
    try:
        return p.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except Exception:
        return p.resolve().as_posix()


# -------------------------
# Main
# -------------------------
def main() -> int:
    ap = argparse.ArgumentParser()

    # Phase 3 runner calls these (fix for your error)
    ap.add_argument("--frames", default="", help="Directory containing input frames/images")
    ap.add_argument("--recursive", action="store_true", help="Recursively search --frames directory")

    # Existing mode (metrics-driven)
    ap.add_argument(
        "--metrics-csv",
        default="",
        help="CSV containing at least a 'relpath' column (optionally 'img_ok'). If provided, this mode is used.",
    )

    ap.add_argument("--ckpt", required=True, help="Path to trained .pt checkpoint from train_unet.py")
    ap.add_argument("--outdir", default=str(REPO_ROOT / "out"))
    ap.add_argument("--image-size", type=int, default=256, help="Must match training image size")
    ap.add_argument("--base-ch", type=int, default=32, help="Must match training base channels")
    ap.add_argument("--threshold", type=float, default=0.5, help="Sigmoid threshold for binary mask")
    ap.add_argument("--device", default="", help="cpu|cuda (default auto)")
    args = ap.parse_args()

    # Validate input mode
    metrics_csv = str(args.metrics_csv).strip()
    frames_arg = str(args.frames).strip()

    if not metrics_csv and not frames_arg:
        raise SystemExit("ERROR: Provide either --metrics-csv CSV or --frames DIR")

    out_root = Path(args.outdir).resolve()
    out_masks = out_root / "masks_unet"
    out_masks.mkdir(parents=True, exist_ok=True)

    # Build dataframe of inputs
    if metrics_csv:
        metrics_path = Path(metrics_csv)
        if not metrics_path.is_absolute():
            metrics_path = (REPO_ROOT / metrics_path).resolve()
        if not metrics_path.exists():
            raise SystemExit(f"ERROR: --metrics-csv not found: {metrics_path}")

        df = pd.read_csv(metrics_path)
        if "relpath" not in df.columns:
            raise SystemExit("ERROR: metrics CSV must contain a 'relpath' column")

        if "img_ok" in df.columns:
            df = df[df["img_ok"] == 1].copy()

        # Normalize relpaths
        df["relpath"] = df["relpath"].astype(str).str.replace("\\", "/", regex=False)

        # Resolve to actual files
        img_paths: List[Path] = []
        for rp in df["relpath"].tolist():
            p = Path(rp)
            if not p.is_absolute():
                p = (REPO_ROOT / p).resolve()
            else:
                p = p.resolve()
            img_paths.append(p)

    else:
        frames_dir = Path(frames_arg).expanduser()
        if not frames_dir.is_absolute():
            frames_dir = (REPO_ROOT / frames_dir).resolve()
        if not frames_dir.exists():
            raise SystemExit(f"ERROR: --frames directory not found: {frames_dir}")
        if not frames_dir.is_dir():
            raise SystemExit(f"ERROR: --frames is not a directory: {frames_dir}")

        img_paths = _iter_images(frames_dir, bool(args.recursive))
        if not img_paths:
            raise SystemExit(f"ERROR: No images found in {frames_dir} (recursive={bool(args.recursive)})")

        # Create a minimal df compatible with downstream expectations
        df = pd.DataFrame({"relpath": [_rel_to_repo(p) for p in img_paths]})

    # Device
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = (REPO_ROOT / ckpt_path).resolve()
    if not ckpt_path.exists():
        raise SystemExit(f"ERROR: --ckpt not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)

    model = UNetSmall(in_ch=3, base=int(args.base_ch), out_ch=1).to(device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        # Allow raw state_dict checkpoints too
        model.load_state_dict(ckpt)
    model.eval()

    new_mask_paths: List[str] = []
    cloud_fracs: List[float] = []

    with torch.no_grad():
        for rp, img_path in zip(df["relpath"].tolist(), img_paths):
            if not img_path.exists():
                raise SystemExit(f"ERROR: Image not found: {img_path}")

            x = load_rgb(img_path, args.image_size)
            xt = torch.from_numpy(x).unsqueeze(0).to(device)

            logits = model(xt)
            prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

            mask = (prob >= float(args.threshold)).astype(np.uint8) * 255

            stem = img_path.stem
            mask_path = out_masks / f"{stem}_mask.png"
            Image.fromarray(mask).save(mask_path)

            # Store paths relative to repo if possible (keeps CSV stable)
            try:
                new_mask_paths.append(mask_path.resolve().relative_to(REPO_ROOT.resolve()).as_posix())
            except Exception:
                new_mask_paths.append(mask_path.resolve().as_posix())

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

if __name__ == "__main__":
    raise SystemExit(main())
