from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple, Optional
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


REPO_ROOT = Path(__file__).resolve()
while not (REPO_ROOT / "pyproject.toml").exists() and not (REPO_ROOT / ".git").exists():
    if REPO_ROOT == REPO_ROOT.parent:
        break
    REPO_ROOT = REPO_ROOT.parent



# -------------------------
# Utils
# -------------------------

def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def as_posix_rel(p: Path) -> str:
    return p.relative_to(REPO_ROOT).as_posix()


# -------------------------
# Dataset
# -------------------------

class CloudMaskDataset(Dataset):
    """
    Uses out/cloud_metrics.csv which contains:
      - relpath (raw image path relative to repo root)
      - mask_path (mask path relative to repo root)
      - img_ok
    """
    def __init__(
        self,
        metrics_csv: Path,
        image_size: int = 256,
        augment: bool = False,
    ):
        self.metrics_csv = metrics_csv
        self.image_size = int(image_size)
        self.augment = bool(augment)

        df = pd.read_csv(metrics_csv)

        # Keep only readable frames
        if "img_ok" in df.columns:
            df = df[df["img_ok"] == 1].copy()

        needed = ["relpath", "mask_path"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"cloud_metrics.csv missing columns: {missing}")

        # Normalize slashes
        df["relpath"] = df["relpath"].astype(str).str.replace("\\", "/", regex=False)
        df["mask_path"] = df["mask_path"].astype(str).str.replace("\\", "/", regex=False)

        # Only keep rows that actually exist on disk
        def exists(rel: str) -> bool:
            return (REPO_ROOT / rel).exists()

        df = df[df["relpath"].map(exists) & df["mask_path"].map(exists)].copy()
        df = df.reset_index(drop=True)

        if len(df) == 0:
            raise ValueError("No usable rows found. Make sure you ran cloud_mask_baseline.py and files exist.")

        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def _resize(self, im: Image.Image, is_mask: bool) -> Image.Image:
        # Use nearest for masks to preserve binary edges
        resample = Image.Resampling.NEAREST if is_mask else Image.Resampling.BILINEAR
        return im.resize((self.image_size, self.image_size), resample=resample)

    def _augment_pair(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (3,H,W), y: (1,H,W)
        # Random flips + 90-degree rotations (safe augmentations for global Earth disk)
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[2])  # horizontal
            y = torch.flip(y, dims=[2])
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[1])  # vertical
            y = torch.flip(y, dims=[1])
        # rotate k * 90
        k = int(torch.randint(0, 4, (1,)).item())
        if k:
            x = torch.rot90(x, k, dims=[1, 2])
            y = torch.rot90(y, k, dims=[1, 2])
        return x, y

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = REPO_ROOT / row["relpath"]
        mask_path = REPO_ROOT / row["mask_path"]

        im = Image.open(img_path).convert("RGB")
        m = Image.open(mask_path).convert("L")

        im = self._resize(im, is_mask=False)
        m = self._resize(m, is_mask=True)

        x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)  # (3,H,W)
        y = torch.from_numpy((np.asarray(m, dtype=np.float32) > 127).astype(np.float32))[None, ...]  # (1,H,W)

        if self.augment:
            x, y = self._augment_pair(x, y)

        return {
            "x": x,
            "y": y,
            "relpath": row["relpath"],
            "mask_path": row["mask_path"],
        }


# -------------------------
# Model (small U-Net)
# -------------------------

def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNetSmall(nn.Module):
    def __init__(self, in_ch: int = 3, base: int = 32, out_ch: int = 1):
        super().__init__()
        self.enc1 = conv_block(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = conv_block(base * 4, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = conv_block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = conv_block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = conv_block(base * 2, base)

        self.head = nn.Conv2d(base, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.head(d1)  # logits


# -------------------------
# Loss + metrics
# -------------------------

def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    inter = (probs * targets).sum(dim=1)
    denom = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


@torch.no_grad()
def dice_iou_from_logits(logits: torch.Tensor, targets: torch.Tensor, thr: float = 0.5, eps: float = 1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs >= thr).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    inter = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - inter
    iou = (inter + eps) / (union + eps)

    denom = preds.sum(dim=1) + targets.sum(dim=1)
    dice = (2 * inter + eps) / (denom + eps)
    return float(dice.mean().cpu().item()), float(iou.mean().cpu().item())


# -------------------------
# Visualization
# -------------------------

@torch.no_grad()
def save_val_previews(model: nn.Module, dl: DataLoader, out_png: Path, device: torch.device, n: int = 8):
    model.eval()
    batch = next(iter(dl))
    x = batch["x"][:n].to(device)
    y = batch["y"][:n].to(device)

    logits = model(x)
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()

    # Make a simple grid: RGB | GT mask | Pred mask
    n = x.size(0)
    fig_h = n * 2.5
    plt.figure(figsize=(10, fig_h))

    for i in range(n):
        rgb = x[i].detach().cpu().permute(1, 2, 0).numpy()
        gt = y[i].detach().cpu()[0].numpy()
        pr = preds[i].detach().cpu()[0].numpy()

        # Row i has 3 columns
        ax1 = plt.subplot(n, 3, i * 3 + 1)
        ax1.imshow(rgb)
        ax1.axis("off")
        if i == 0:
            ax1.set_title("RGB")

        ax2 = plt.subplot(n, 3, i * 3 + 2)
        ax2.imshow(gt, cmap="gray")
        ax2.axis("off")
        if i == 0:
            ax2.set_title("Baseline mask (GT)")

        ax3 = plt.subplot(n, 3, i * 3 + 3)
        ax3.imshow(pr, cmap="gray")
        ax3.axis("off")
        if i == 0:
            ax3.set_title("U-Net pred")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def save_training_curves(hist: pd.DataFrame, out_png: Path):
    plt.figure()
    plt.plot(hist["epoch"], hist["train_loss"], label="train_loss")
    plt.plot(hist["epoch"], hist["val_loss"], label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("U-Net training curves")
    plt.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


# -------------------------
# Train loop
# -------------------------

def run_epoch(model, dl, opt, device, train: bool, bce_weight: float):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    n_batches = 0

    bce = nn.BCEWithLogitsLoss()

    for batch in dl:
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        if train:
            opt.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = bce_weight * bce(logits, y) + (1.0 - bce_weight) * dice_loss_from_logits(logits, y)

            if train:
                loss.backward()
                opt.step()

        d, i = dice_iou_from_logits(logits.detach(), y.detach())

        total_loss += float(loss.detach().cpu().item())
        total_dice += d
        total_iou += i
        n_batches += 1

    return {
        "loss": total_loss / max(1, n_batches),
        "dice": total_dice / max(1, n_batches),
        "iou": total_iou / max(1, n_batches),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-csv", default=str(REPO_ROOT / "out" / "cloud_metrics.csv"),
                    help="CSV produced by cloud_mask_baseline.py (out/cloud_metrics.csv)")
    ap.add_argument("--outdir", default=str(REPO_ROOT / "out"), help="Base output directory (default: out/)")
    ap.add_argument("--image-size", type=int, default=256, help="Resize images/masks to this square size")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--base-ch", type=int, default=32, help="U-Net base channels (32 good CPU/GPU balance)")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--augment", action="store_true", help="Enable simple flips/rotations")
    ap.add_argument("--bce-weight", type=float, default=0.5, help="Blend BCE and Dice (0.5 = equal)")
    ap.add_argument("--device", default="", help="Force device: cpu | cuda. Default auto.")
    args = ap.parse_args()

    set_seed(args.seed)

    out_root = Path(args.outdir).resolve()
    out_models = out_root / "models"
    out_viz = out_root / "viz"
    out_models.mkdir(parents=True, exist_ok=True)
    out_viz.mkdir(parents=True, exist_ok=True)

    metrics_csv = Path(args.metrics_csv).resolve()
    ds = CloudMaskDataset(metrics_csv=metrics_csv, image_size=args.image_size, augment=args.augment)

    # Split train/val (random; later you can split by time)
    n = len(ds)
    idx = np.arange(n)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(idx)

    n_val = int(math.ceil(n * float(args.val_frac)))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    train_ds = torch.utils.data.Subset(ds, train_idx.tolist())
    val_ds = torch.utils.data.Subset(CloudMaskDataset(metrics_csv=metrics_csv, image_size=args.image_size, augment=False), val_idx.tolist())

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)

    # Device
    if args.device.strip():
        device = torch.device(args.device.strip())
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNetSmall(in_ch=3, base=args.base_ch, out_ch=1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    run_id = f"unet_cloudmask_{utc_stamp()}"
    best_path = out_models / f"{run_id}.pt"
    metrics_path = out_models / f"{run_id}.metrics.csv"
    config_path = out_models / f"{run_id}.config.json"
    curves_png = out_viz / "unet_training_curves.png"
    previews_png = out_viz / "unet_val_previews.png"

    # Save config
    config = vars(args) | {
        "run_id": run_id,
        "device": str(device),
        "n_total": n,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
    }
    config_path.write_text(json.dumps(config, indent=2))

    best_val_loss = float("inf")
    history = []

    print(f"Run: {run_id}")
    print(f"Device: {device}")
    print(f"Train/Val: {len(train_ds)}/{len(val_ds)}")
    print(f"Saving best model to: {best_path}")

    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_dl, opt, device, train=True, bce_weight=float(args.bce_weight))
        va = run_epoch(model, val_dl, opt, device, train=False, bce_weight=float(args.bce_weight))

        row = {
            "epoch": epoch,
            "train_loss": tr["loss"],
            "train_dice": tr["dice"],
            "train_iou": tr["iou"],
            "val_loss": va["loss"],
            "val_dice": va["dice"],
            "val_iou": va["iou"],
        }
        history.append(row)

        print(
            f"epoch {epoch:02d} | "
            f"train loss {tr['loss']:.4f} dice {tr['dice']:.3f} iou {tr['iou']:.3f} | "
            f"val loss {va['loss']:.4f} dice {va['dice']:.3f} iou {va['iou']:.3f}"
        )

        # Save best
        if va["loss"] < best_val_loss:
            best_val_loss = va["loss"]
            torch.save({"model": model.state_dict(), "config": config}, best_path)

    hist = pd.DataFrame(history)
    hist.to_csv(metrics_path, index=False)
    save_training_curves(hist, curves_png)

    # Save previews using best checkpoint
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    save_val_previews(model, val_dl, previews_png, device=device, n=8)

    print(f"Wrote: {metrics_path}")
    print(f"Wrote: {curves_png}")
    print(f"Wrote: {previews_png}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
