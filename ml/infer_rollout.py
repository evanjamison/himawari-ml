# ml/infer_rollout.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]


# -----------------------------
# I/O helpers
# -----------------------------
def _norm_path(s: str) -> str:
    return str(s).replace("\\", "/")


def _exists(rel: str) -> bool:
    return (REPO_ROOT / rel).exists()


def _load_mask_tensor(relpath: str, size: int) -> torch.Tensor:
    """
    Loads binary mask -> (1,H,W) float {0,1}
    """
    p = (REPO_ROOT / relpath).resolve()
    im = Image.open(p).convert("L").resize((size, size), Image.Resampling.NEAREST)
    arr = np.asarray(im, dtype=np.uint8)
    m = (arr > 127).astype(np.float32)
    return torch.from_numpy(m)[None, ...]


def _load_image_tensor(relpath: str, size: int, use_rgb: bool) -> torch.Tensor:
    """
    Loads satellite image -> (1,H,W) gray or (3,H,W) rgb, float [0,1]
    """
    p = (REPO_ROOT / relpath).resolve()
    im = Image.open(p)
    if use_rgb:
        im = im.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        arr = np.asarray(im, dtype=np.uint8)  # (H,W,3)
        return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    else:
        im = im.convert("L").resize((size, size), Image.Resampling.BILINEAR)
        arr = np.asarray(im, dtype=np.uint8)  # (H,W)
        return torch.from_numpy(arr)[None, ...].float() / 255.0


def _load_rgb_numpy(relpath: str, size: int) -> Optional[np.ndarray]:
    p = (REPO_ROOT / relpath).resolve()
    if not p.exists():
        return None
    im = Image.open(p).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


# -----------------------------
# Model (must match training)
# -----------------------------
class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch: int, hid_ch: int, k: int = 3):
        super().__init__()
        pad = k // 2
        self.hid_ch = hid_ch
        self.conv = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, kernel_size=k, padding=pad)

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor], c: Optional[torch.Tensor]):
        if h is None or c is None:
            B, _, H, W = x.shape
            h = torch.zeros(B, self.hid_ch, H, W, device=x.device)
            c = torch.zeros(B, self.hid_ch, H, W, device=x.device)

        cat = torch.cat([x, h], dim=1)
        gates = self.conv(cat)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


class ConvLSTMNowcaster(nn.Module):
    """
    Input:  (B,T,C,H,W)
    Output: (B,1,H,W) logits
    """
    def __init__(self, in_ch: int, hid_ch: int = 48):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, hid_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.cell = ConvLSTMCell(hid_ch, hid_ch, k=3)
        self.head = nn.Sequential(
            nn.Conv2d(hid_ch, hid_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid_ch, 1, 1),
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x_seq.shape
        h = c = None
        for t in range(T):
            feat = self.enc(x_seq[:, t])
            h, c = self.cell(feat, h, c)
        return self.head(h)


# -----------------------------
# Data loading
# -----------------------------
def load_df(metrics_csv: Path, img_col: Optional[str] = None) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(metrics_csv)

    if "mask_path" not in df.columns:
        raise ValueError("CSV must contain 'mask_path'.")

    if img_col is not None:
        if img_col not in df.columns:
            raise ValueError(f"--img-col '{img_col}' not found. Columns={list(df.columns)}")
        ic = img_col
    else:
        ic = None
        for cand in ["rgb_path", "relpath", "image_path", "frame_path"]:
            if cand in df.columns:
                ic = cand
                break
        if ic is None:
            raise ValueError(
                "Could not auto-detect image column. Add one of: rgb_path, relpath, image_path, frame_path "
                "or pass --img-col."
            )

    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
        df = df.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc")
    else:
        df = df.copy()

    df["mask_path"] = df["mask_path"].astype(str).map(_norm_path)
    df[ic] = df[ic].astype(str).map(_norm_path)

    keep = df["mask_path"].map(_exists) & df[ic].map(_exists)
    df = df[keep].reset_index(drop=True)

    if len(df) < 2:
        raise ValueError(f"Not enough valid rows after filtering existing files. rows={len(df)}")

    return df, ic


def make_input_window(
    masks: List[torch.Tensor],  # list of (1,H,W)
    img_paths: List[str],       # list of relpaths length T
    image_size: int,
    use_rgb: bool,
) -> torch.Tensor:
    """
    Build (1,T,C,H,W) where C = 1(mask) + (1 or 3 image)
    """
    frames = []
    for m, ip in zip(masks, img_paths):
        im = _load_image_tensor(ip, image_size, use_rgb=use_rgb)
        frames.append(torch.cat([m, im], dim=0))  # (C,H,W)
    x = torch.stack(frames, dim=0)[None, ...]  # (1,T,C,H,W)
    return x


# -----------------------------
# Rollout
# -----------------------------
@torch.no_grad()
def rollout(
    model: nn.Module,
    df: pd.DataFrame,
    img_col: str,
    seq_len: int,
    image_size: int,
    start_idx: int,
    horizon: int,
    use_rgb: bool,
    thr: float,
    feedback: str,  # "prob" or "bin"
    device: torch.device,
) -> List[Dict[str, Any]]:
    """
    Returns list of dicts for each forecast step with rgb/gt/pred.
    """
    n = len(df)
    if start_idx < 0 or start_idx + seq_len + horizon >= n:
        raise ValueError(
            f"start_idx out of range. Need start_idx + seq_len + horizon < n. "
            f"Got start_idx={start_idx}, seq_len={seq_len}, horizon={horizon}, n={n}"
        )

    # initial mask history from GT
    mask_hist: List[torch.Tensor] = []
    for i in range(start_idx, start_idx + seq_len):
        mask_hist.append(_load_mask_tensor(df.loc[i, "mask_path"], image_size))

    results = []

    for k in range(1, horizon + 1):
        target_i = start_idx + seq_len + (k - 1)
        # input images correspond to the SAME timesteps as the masks in window
        img_window_paths = [df.loc[i, img_col] for i in range(target_i - seq_len, target_i)]

        x = make_input_window(
            masks=mask_hist[-seq_len:],
            img_paths=img_window_paths,
            image_size=image_size,
            use_rgb=use_rgb,
        ).to(device)

        logits = model(x)
        prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()  # (H,W)
        pred_bin = (prob > thr).astype(np.float32)

        # choose feedback mask for next step
        if feedback == "prob":
            next_mask = torch.from_numpy(prob.astype(np.float32))[None, ...]  # (1,H,W)
        else:
            next_mask = torch.from_numpy(pred_bin.astype(np.float32))[None, ...]

        # append prediction to history
        mask_hist.append(next_mask)

        # GT + RGB for target time (within dataset)
        gt = _load_mask_tensor(df.loc[target_i, "mask_path"], image_size)[0].numpy()
        rgb = _load_rgb_numpy(df.loc[target_i, img_col], image_size)

        results.append(
            {
                "step": k,
                "target_index": int(target_i),
                "timestamp": str(df.loc[target_i, "timestamp_utc"]) if "timestamp_utc" in df.columns else str(target_i),
                "rgb": rgb,
                "gt": gt,
                "pred_prob": prob,
                "pred_bin": pred_bin,
            }
        )

    return results


def save_rollout_grid(rows: List[Dict[str, Any]], out_png: Path, thr: float) -> None:
    """
    Rows: each has rgb, gt, pred_prob, pred_bin
    """
    if not rows:
        return

    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(14, 4 * n))
    if n == 1:
        axes = np.array([axes])

    for i, r in enumerate(rows):
        ax0, ax1, ax2, ax3 = axes[i]

        rgb = r["rgb"]
        if rgb is not None:
            ax0.imshow(rgb)
            ax0.set_title(f"RGB (t+{r['step']})")
        else:
            ax0.imshow(r["gt"], cmap="gray")
            ax0.set_title(f"GT (no RGB) t+{r['step']}")
        ax0.axis("off")

        ax1.imshow(r["gt"], cmap="gray")
        ax1.set_title(f"GT mask (t+{r['step']})")
        ax1.axis("off")

        ax2.imshow(r["pred_prob"], cmap="gray")
        ax2.set_title(f"Pred prob (t+{r['step']})")
        ax2.axis("off")

        ax3.imshow(r["pred_bin"], cmap="gray")
        ax3.set_title(f"Pred bin (thr={thr}, t+{r['step']})")
        ax3.axis("off")

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-csv", required=True, help="CSV with mask_path + image path column")
    ap.add_argument("--ckpt", required=True, help="Path to checkpoint .pt (e.g., out/models/convlstm_nowcaster_best_gray.pt)")
    ap.add_argument("--outdir", default=str(REPO_ROOT / "out" / "rollout"))
    ap.add_argument("--start-idx", type=int, default=0, help="Start row index for the history window")
    ap.add_argument("--horizon", type=int, default=5, help="Forecast steps to roll out (t+1..t+K)")
    ap.add_argument("--thr", type=float, default=0.30, help="Threshold for binarized preview")
    ap.add_argument("--feedback", choices=["prob", "bin"], default="prob", help="What to feed back as next mask")
    ap.add_argument("--use-rgb", action="store_true", help="Use RGB imagery input (must match training)")
    ap.add_argument("--img-col", default=None, help="Override image column name in CSV")
    ap.add_argument("--device", default="cuda", help="cuda or cpu")
    args = ap.parse_args()

    metrics_csv = Path(args.metrics_csv).resolve()
    ckpt_path = Path(args.ckpt).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    # load checkpoint + cfg
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("cfg", {}) or {}
    seq_len = int(cfg.get("seq_len", 6))
    image_size = int(cfg.get("image_size", 256))
    hid_ch = int(cfg.get("hid_ch", 48))

    # load dataframe
    df, img_col = load_df(metrics_csv, img_col=args.img_col)

    # determine in_ch
    # C = 1(mask) + (1 gray or 3 rgb)
    in_ch = 1 + (3 if args.use_rgb else 1)

    model = ConvLSTMNowcaster(in_ch=in_ch, hid_ch=hid_ch).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    rows = rollout(
        model=model,
        df=df,
        img_col=img_col,
        seq_len=seq_len,
        image_size=image_size,
        start_idx=int(args.start_idx),
        horizon=int(args.horizon),
        use_rgb=bool(args.use_rgb),
        thr=float(args.thr),
        feedback=str(args.feedback),
        device=device,
    )

    tag = "rgb" if args.use_rgb else "gray"
    out_png = outdir / f"rollout_{tag}_start{args.start_idx}_K{args.horizon}_thr{args.thr}_{args.feedback}.png"
    save_rollout_grid(rows, out_png, thr=float(args.thr))

    print(f"[done] saved rollout grid -> {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
