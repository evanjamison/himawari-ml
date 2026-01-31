from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ---------------------------
# Utilities
# ---------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def list_images_recursive(root: Path) -> List[Path]:
    files: List[Path] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
        files.extend(root.rglob(ext))
    return sorted(files)


def read_grayscale(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("L"))


def read_rgb(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("RGB"))


# ---------------------------
# Simple U-Net (key-compatible with infer_unet_masks.py)
# IMPORTANT:
#   enc1/enc2/... are nn.Sequential so state_dict keys are enc1.0.weight etc.
#   This matches your existing ml/infer_unet_masks.py expectations.
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
# Losses + metrics
# ---------------------------

def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    num = 2 * (probs * targets).sum(dim=(2, 3))
    den = (probs + targets).sum(dim=(2, 3)) + eps
    dice = num / den
    return 1.0 - dice.mean()


@torch.no_grad()
def dice_iou_from_logits(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6
) -> Tuple[float, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    inter = (preds * targets).sum(dim=(2, 3))
    union = (preds + targets - preds * targets).sum(dim=(2, 3)) + eps
    iou = (inter + eps) / union
    dice = (2 * inter + eps) / ((preds + targets).sum(dim=(2, 3)) + eps)
    return float(dice.mean().item()), float(iou.mean().item())


# ---------------------------
# Dataset
# ---------------------------

@dataclass
class Pair:
    img: Path
    mask: Path
    key: str


class FrameMaskDataset(Dataset):
    """
    Expects:
      frames_dir: contains RGB images (*.png)
      masks_dir : contains corresponding mask PNGs

    Matching rule:
      mask filename matches frame stem, or stem + "_mask"
      Recommended: frame "himawari_...Z.png" <-> mask "himawari_...Z_mask.png"
    """

    def __init__(
        self,
        frames_dir: Path,
        masks_dir: Path,
        image_size: int = 256,
        threshold: int = 128,
        augment: bool = True,
        recursive: bool = True,
    ):
        self.frames_dir = frames_dir
        self.masks_dir = masks_dir
        self.image_size = image_size
        self.threshold = threshold
        self.augment = augment

        frame_files = list_images_recursive(frames_dir) if recursive else sorted(frames_dir.glob("*.png"))
        mask_files = list_images_recursive(masks_dir) if recursive else sorted(masks_dir.glob("*.png"))

        mask_map = {}
        for m in mask_files:
            s = m.stem
            mask_map[s] = m
            if s.endswith("_mask"):
                mask_map[s[:-5]] = m  # strip "_mask"

        pairs: List[Pair] = []
        for f in frame_files:
            stem = f.stem
            m = mask_map.get(stem) or mask_map.get(stem + "_mask")
            if m is None:
                continue
            pairs.append(Pair(img=f, mask=m, key=stem))

        if not pairs:
            raise RuntimeError(
                f"No frame/mask pairs found.\nframes_dir={frames_dir}\nmasks_dir={masks_dir}\n"
                f"Expected matching stems like frame.png <-> frame_mask.png"
            )

        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def _resize(self, arr: np.ndarray, is_mask: bool) -> np.ndarray:
        pil = Image.fromarray(arr)
        resample = Image.NEAREST if is_mask else Image.BILINEAR
        pil = pil.resize((self.image_size, self.image_size), resample=resample)
        return np.array(pil)

    def _augment(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.augment:
            return x, y

        if random.random() < 0.5:
            x = np.flip(x, axis=1).copy()
            y = np.flip(y, axis=1).copy()
        if random.random() < 0.5:
            x = np.flip(x, axis=0).copy()
            y = np.flip(y, axis=0).copy()

        k = random.randint(0, 3)
        if k:
            x = np.rot90(x, k, axes=(0, 1)).copy()
            y = np.rot90(y, k, axes=(0, 1)).copy()
        return x, y

    def __getitem__(self, idx: int):
        pair = self.pairs[idx]
        x = read_rgb(pair.img)         # uint8 HxWx3
        y = read_grayscale(pair.mask)  # uint8 HxW

        x = self._resize(x, is_mask=False)
        y = self._resize(y, is_mask=True)

        x, y = self._augment(x, y)

        x = (x.astype(np.float32) / 255.0).transpose(2, 0, 1)  # 3xHxW
        y = (y.astype(np.uint8) >= self.threshold).astype(np.float32)[None, :, :]  # 1xHxW

        return torch.from_numpy(x), torch.from_numpy(y), pair.key


# ---------------------------
# Train loop
# ---------------------------

def split_by_time(pairs: List[Pair], val_frac: float = 0.2) -> Tuple[List[int], List[int]]:
    idxs = list(range(len(pairs)))
    idxs.sort(key=lambda i: pairs[i].key)
    n = len(idxs)
    n_val = max(1, int(math.floor(n * val_frac)))
    val = idxs[-n_val:]
    train = idxs[:-n_val]
    return train, val


def split_random(n: int, val_frac: float = 0.2, seed: int = 0) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    idxs = list(range(n))
    rng.shuffle(idxs)
    n_val = max(1, int(math.floor(n * val_frac)))
    val = idxs[:n_val]
    train = idxs[n_val:]
    return train, val


def save_preview_grid(
    out_path: Path,
    keys: List[str],
    xs: torch.Tensor,
    ys: torch.Tensor,
    logits: torch.Tensor,
    threshold: float = 0.5,
    max_items: int = 8,
):
    # Simple contact sheet: (input, GT, pred)
    from PIL import ImageDraw

    xs = xs[:max_items].cpu().numpy()
    ys = ys[:max_items].cpu().numpy()
    probs = torch.sigmoid(logits[:max_items]).cpu().numpy()
    preds = (probs >= threshold).astype(np.uint8) * 255

    rows = []
    for i in range(xs.shape[0]):
        x = (xs[i].transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
        gt = (ys[i, 0] * 255.0).astype(np.uint8)
        pr = preds[i, 0].astype(np.uint8)

        x_img = Image.fromarray(x)
        gt_img = Image.fromarray(gt).convert("RGB")
        pr_img = Image.fromarray(pr).convert("RGB")

        strip_h = 18
        strip = Image.new("RGB", (x_img.width * 3, strip_h), (20, 20, 20))
        draw = ImageDraw.Draw(strip)
        draw.text((6, 2), f"{keys[i]}  |  input / gt / pred", fill=(240, 240, 240))

        row = Image.new("RGB", (x_img.width * 3, x_img.height + strip_h))
        row.paste(strip, (0, 0))
        row.paste(x_img, (0, strip_h))
        row.paste(gt_img, (x_img.width, strip_h))
        row.paste(pr_img, (x_img.width * 2, strip_h))
        rows.append(row)

    if not rows:
        return

    W = rows[0].width
    H = sum(r.height for r in rows)
    sheet = Image.new("RGB", (W, H), (0, 0, 0))
    yoff = 0
    for r in rows:
        sheet.paste(r, (0, yoff))
        yoff += r.height

    ensure_dir(out_path.parent)
    sheet.save(out_path)


def main():
    ap = argparse.ArgumentParser(description="Train U-Net on pixel masks (true segmentation).")
    ap.add_argument("--frames-dir", type=str, required=True, help="Directory containing input frames.")
    ap.add_argument("--masks-dir", type=str, required=True, help="Directory containing teacher masks (png).")
    ap.add_argument("--outdir", type=str, default="out", help="Output directory (writes out/models, out/viz).")

    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--mask-threshold", type=int, default=128, help=">= this value considered positive in mask.")

    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--base-ch", type=int, default=32)

    ap.add_argument("--bce-weight", type=float, default=0.5, help="Weight for BCE in (dice + bce).")
    ap.add_argument("--pos-weight", type=float, default=3.0, help="Positive class weight for BCE.")

    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--no-augment", dest="augment", action="store_false")
    ap.set_defaults(augment=True)

    ap.add_argument("--split-by-time", action="store_true", help="Sort by filename and use last fraction as val.")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--threshold-eval", type=float, default=0.5, help="Threshold for reporting Dice/IoU.")
    args = ap.parse_args()

    set_seed(args.seed)

    frames_dir = Path(args.frames_dir)
    masks_dir = Path(args.masks_dir)
    outdir = Path(args.outdir)
    models_dir = outdir / "models"
    viz_dir = outdir / "viz"
    ensure_dir(models_dir)
    ensure_dir(viz_dir)

    if args.device == "cuda":
        device = "cuda"
    elif args.device == "cpu":
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = FrameMaskDataset(
        frames_dir=frames_dir,
        masks_dir=masks_dir,
        image_size=args.image_size,
        threshold=args.mask_threshold,
        augment=args.augment,
        recursive=True,
    )

    if args.split_by_time:
        train_idx, val_idx = split_by_time(ds.pairs, val_frac=args.val_frac)
    else:
        train_idx, val_idx = split_random(len(ds), val_frac=args.val_frac, seed=args.seed)

    train_ds = torch.utils.data.Subset(ds, train_idx)
    val_ds = torch.utils.data.Subset(ds, val_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = UNet(in_ch=3, base_ch=args.base_ch).to(device)

    pos_weight = torch.tensor([args.pos_weight], device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_val = float("inf")

    run_id = f"unet_pixelmask_{frames_dir.name}_{args.image_size}px_seed{args.seed}"
    cfg = vars(args) | {
        "run_id": run_id,
        "device_resolved": device,
        "n_pairs": len(ds),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
    }

    (models_dir / f"{run_id}.config.json").write_text(json.dumps(cfg, indent=2))

    print(f"[INFO] device={device} pairs={len(ds)} train={len(train_ds)} val={len(val_ds)}")
    print(f"[INFO] run_id={run_id}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_losses, tr_dice, tr_iou = [], [], []

        for x, y, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)

            ld = dice_loss_from_logits(logits, y)
            lb = bce(logits, y)
            loss = ld + args.bce_weight * lb

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            d, i = dice_iou_from_logits(logits.detach(), y, threshold=args.threshold_eval)
            tr_losses.append(loss.item())
            tr_dice.append(d)
            tr_iou.append(i)

        model.eval()
        va_losses, va_dice, va_iou = [], [], []
        preview_batch = None

        with torch.no_grad():
            for x, y, keys in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                logits = model(x)

                ld = dice_loss_from_logits(logits, y)
                lb = bce(logits, y)
                loss = ld + args.bce_weight * lb

                d, i = dice_iou_from_logits(logits, y, threshold=args.threshold_eval)
                va_losses.append(loss.item())
                va_dice.append(d)
                va_iou.append(i)

                if preview_batch is None:
                    preview_batch = (list(keys), x.detach().cpu(), y.detach().cpu(), logits.detach().cpu())

        tr_loss = float(np.mean(tr_losses)) if tr_losses else float("nan")
        va_loss = float(np.mean(va_losses)) if va_losses else float("nan")
        tr_d = float(np.mean(tr_dice)) if tr_dice else float("nan")
        va_d = float(np.mean(va_dice)) if va_dice else float("nan")
        tr_i = float(np.mean(tr_iou)) if tr_iou else float("nan")
        va_i = float(np.mean(va_iou)) if va_iou else float("nan")

        print(
            f"epoch {epoch:02d} | "
            f"train loss {tr_loss:.4f} dice {tr_d:.3f} iou {tr_i:.3f} | "
            f"val loss {va_loss:.4f} dice {va_d:.3f} iou {va_i:.3f}"
        )

        if preview_batch is not None:
            keys, xcpu, ycpu, lcpu = preview_batch
            save_preview_grid(
                viz_dir / f"{run_id}_val_preview_epoch{epoch:02d}.png",
                keys,
                xcpu,
                ycpu,
                lcpu,
                threshold=args.threshold_eval,
            )

        if va_loss < best_val:
            best_val = va_loss
            ckpt_path = models_dir / f"{run_id}.pt"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": cfg,
                    "best_val_loss": best_val,
                    "epoch": epoch,
                },
                ckpt_path,
            )
            (models_dir / "unet_pixelmask_latest.pt").write_bytes(ckpt_path.read_bytes())
            print(f"[SAVE] best checkpoint -> {ckpt_path}")
            print(f"[SAVE] wrote canonical -> {models_dir / 'unet_pixelmask_latest.pt'}")

    print("[DONE]")


if __name__ == "__main__":
    main()


