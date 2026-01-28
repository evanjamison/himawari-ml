from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

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
# Model: "enc*/dec*/bottleneck" naming (matches your ckpt)
# -------------------------
def _double_conv(in_ch: int, out_ch: int) -> nn.Sequential:
    # indices: 0 conv, 1 bn, 2 relu, 3 conv, 4 bn, 5 relu
    # (matches keys like enc2.4.running_mean)
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNetEncDec(nn.Module):
    def __init__(self, in_ch: int = 3, base: int = 16, out_ch: int = 1):
        super().__init__()

        self.enc1 = _double_conv(in_ch, base)
        self.enc2 = _double_conv(base, base * 2)
        self.enc3 = _double_conv(base * 2, base * 4)
        self.enc4 = _double_conv(base * 4, base * 8)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = _double_conv(base * 8, base * 16)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, kernel_size=2, stride=2)
        self.dec4 = _double_conv(base * 16, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec3 = _double_conv(base * 8, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = _double_conv(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = _double_conv(base * 2, base)

        self.final = nn.Conv2d(base, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = _concat_with_pad(d4, e4)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = _concat_with_pad(d3, e3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = _concat_with_pad(d2, e2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = _concat_with_pad(d1, e1)
        d1 = self.dec1(d1)

        return self.final(d1)


def _concat_with_pad(x_up: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
    diffY = x_skip.size(2) - x_up.size(2)
    diffX = x_skip.size(3) - x_up.size(3)
    x_up = F.pad(x_up, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
    return torch.cat([x_skip, x_up], dim=1)


# -------------------------
# IO helpers
# -------------------------
def load_rgb(p: Path, image_size: int) -> np.ndarray:
    im = Image.open(p).convert("RGB")
    if im.size != (image_size, image_size):
        im = im.resize((image_size, image_size), resample=Image.Resampling.BILINEAR)
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
# Checkpoint utilities
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

        nk = nk.replace("upconv4", "up4").replace("upconv3", "up3").replace("upconv2", "up2").replace("upconv1", "up1")
        nk = nk.replace("up_conv4", "up4").replace("up_conv3", "up3").replace("up_conv2", "up2").replace("up_conv1", "up1")
        nk = nk.replace("final_conv", "final")

        out[nk] = v
    return out


def infer_base_ch_from_ckpt(sd: Dict[str, torch.Tensor]) -> Optional[int]:
    """
    Infer base channels from bottleneck first conv weight:
      bottleneck.0.weight has shape [base*16, base*8, 3, 3]
    So base = out_channels / 16.
    """
    w = sd.get("bottleneck.0.weight")
    if w is None:
        return None
    out_ch = int(w.shape[0])
    if out_ch % 16 != 0:
        return None
    base = out_ch // 16
    # sanity: common values 16/32/64
    if base <= 0:
        return None
    return base


def load_model_weights(model: nn.Module, sd: Dict[str, torch.Tensor]) -> None:
    # strict load should succeed now that base matches
    model.load_state_dict(sd, strict=True)


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

    # If you pass --base-ch we will honor it; otherwise we auto-detect from ckpt.
    ap.add_argument("--base-ch", type=int, default=0, help="UNet base channels (0 = auto from checkpoint)")

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

    # Device
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint first (so we can infer base channels)
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = (REPO_ROOT / ckpt_path).resolve()
    if not ckpt_path.exists():
        raise SystemExit(f"ERROR: --ckpt not found: {ckpt_path}")

    ckpt_obj = torch.load(ckpt_path, map_location=device)
    sd = normalize_keys(extract_state_dict(ckpt_obj))

    base_ch = int(args.base_ch)
    if base_ch <= 0:
        inferred = infer_base_ch_from_ckpt(sd)
        if inferred is None:
            raise SystemExit("ERROR: Could not auto-infer --base-ch from checkpoint. Please pass --base-ch explicitly.")
        base_ch = inferred

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
            else:
                p = p.resolve()
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

    # Build model with correct base, then load weights
    model = UNetEncDec(in_ch=3, base=base_ch, out_ch=1).to(device)
    load_model_weights(model, sd)
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

            mask = (prob >= float(args.threshold)).astype(np.uint8) * 255
            mask_out = out_masks / f"{img_path.stem}_mask.png"
            Image.fromarray(mask).save(mask_out)

            mask_paths.append(to_repo_rel(mask_out))
            cloud_fracs.append(float((mask > 0).mean()))

    df["mask_path"] = mask_paths
    df["cloud_fraction_unet"] = cloud_fracs

    out_csv = out_root / "cloud_metrics_unet.csv"
    df.to_csv(out_csv, index=False)

    print(f"Auto base-ch: {base_ch}")
    print(f"Wrote: {out_csv}")
    print(f"Wrote masks: {out_masks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


if __name__ == "__main__":
    raise SystemExit(main())
