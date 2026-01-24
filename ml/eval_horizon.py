# ml/eval_horizon.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

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
# I/O
# -----------------------------
def _norm_path(s: str) -> str:
    return str(s).replace("\\", "/")


def _exists(rel: str) -> bool:
    return (REPO_ROOT / rel).exists()


def _load_mask(relpath: str, size: int) -> torch.Tensor:
    """(1,H,W) float {0,1}"""
    p = (REPO_ROOT / relpath).resolve()
    im = Image.open(p).convert("L").resize((size, size), Image.Resampling.NEAREST)
    arr = np.asarray(im, dtype=np.uint8)
    m = (arr > 127).astype(np.float32)
    return torch.from_numpy(m.copy())[None, ...]  # copy() avoids 'not writable' warning


def _load_img(relpath: str, size: int, use_rgb: bool) -> torch.Tensor:
    """(1,H,W) gray or (3,H,W) rgb float [0,1]"""
    p = (REPO_ROOT / relpath).resolve()
    im = Image.open(p)
    if use_rgb:
        im = im.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        arr = np.asarray(im, dtype=np.uint8)
        x = torch.from_numpy(arr.copy()).permute(2, 0, 1).float() / 255.0
        return x
    else:
        im = im.convert("L").resize((size, size), Image.Resampling.BILINEAR)
        arr = np.asarray(im, dtype=np.uint8)
        x = torch.from_numpy(arr.copy())[None, ...].float() / 255.0
        return x


def load_df(metrics_csv: Path, img_col: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
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

    # sort by time if present
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
        df = df.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc")

    df = df.copy()
    df["mask_path"] = df["mask_path"].astype(str).map(_norm_path)
    df[ic] = df[ic].astype(str).map(_norm_path)

    keep = df["mask_path"].map(_exists) & df[ic].map(_exists)
    df = df[keep].reset_index(drop=True)

    if len(df) < 10:
        raise ValueError(f"Too few valid rows after filtering existing files: {len(df)}")

    return df, ic


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
# Metrics
# -----------------------------
def soft_dice(prob: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    # prob in [0,1], gt in {0,1}
    p = prob.reshape(-1)
    g = gt.reshape(-1)
    inter = (p * g).sum()
    denom = p.sum() + g.sum()
    return float((2 * inter + eps) / (denom + eps))


def bin_iou(pred_bin: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    p = pred_bin.reshape(-1).astype(np.float32)
    g = gt.reshape(-1).astype(np.float32)
    inter = (p * g).sum()
    union = p.sum() + g.sum() - inter
    return float((inter + eps) / (union + eps))


def bin_dice(pred_bin: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    p = pred_bin.reshape(-1).astype(np.float32)
    g = gt.reshape(-1).astype(np.float32)
    inter = (p * g).sum()
    denom = p.sum() + g.sum()
    return float((2 * inter + eps) / (denom + eps))


# -----------------------------
# Rollout eval
# -----------------------------
@torch.no_grad()
def eval_one_ckpt(
    df: pd.DataFrame,
    img_col: str,
    ckpt_path: Path,
    use_rgb: bool,
    img_col_override: Optional[str],
    device: torch.device,
    # eval knobs
    horizon: int,
    n_samples: int,
    sample_stride: int,
    start_offset: int,
    thr: float,
    feedback: str,  # "prob" or "bin"
    # model cfg fallback
    seq_len_arg: Optional[int],
    image_size_arg: Optional[int],
    hid_ch_arg: Optional[int],
) -> pd.DataFrame:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("cfg", {}) or {}

    seq_len = int(seq_len_arg or cfg.get("seq_len", 6))
    image_size = int(image_size_arg or cfg.get("image_size", 256))
    hid_ch = int(hid_ch_arg or cfg.get("hid_ch", 48))

    # in_ch must match how you trained
    in_ch = 1 + (3 if use_rgb else 1)

    model = ConvLSTMNowcaster(in_ch=in_ch, hid_ch=hid_ch).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    N = len(df)
    # valid starts must allow: start + seq_len + horizon < N
    max_start = N - (seq_len + horizon) - 1
    if max_start <= 0:
        raise ValueError(f"Not enough frames for eval: N={N}, seq_len={seq_len}, horizon={horizon}")

    starts = list(range(start_offset, max_start, sample_stride))
    if n_samples > 0:
        starts = starts[:n_samples]

    rows: List[Dict[str, Any]] = []

    for s in starts:
        # initial GT mask history
        mask_hist: List[torch.Tensor] = [
            _load_mask(df.loc[i, "mask_path"], image_size) for i in range(s, s + seq_len)
        ]

        for k in range(1, horizon + 1):
            target_i = s + seq_len + (k - 1)

            # images aligned with current mask window
            img_paths = [df.loc[i, img_col] for i in range(target_i - seq_len, target_i)]
            frames = []
            for m, ip in zip(mask_hist[-seq_len:], img_paths):
                im = _load_img(ip, image_size, use_rgb)
                frames.append(torch.cat([m, im], dim=0))  # (C,H,W)

            x = torch.stack(frames, dim=0)[None, ...].to(device)  # (1,T,C,H,W)
            logits = model(x)
            prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
            pred_bin = (prob > thr).astype(np.float32)

            gt = _load_mask(df.loc[target_i, "mask_path"], image_size)[0].numpy()

            # metrics
            sd = soft_dice(prob, gt)
            bd = bin_dice(pred_bin, gt)
            bi = bin_iou(pred_bin, gt)

            rows.append(
                {
                    "ckpt": ckpt_path.name,
                    "start": int(s),
                    "target_index": int(target_i),
                    "horizon": int(k),
                    "soft_dice": sd,
                    "bin_dice": bd,
                    "bin_iou": bi,
                }
            )

            # feedback
            if feedback == "prob":
                next_mask = torch.from_numpy(prob.astype(np.float32))[None, ...]
            else:
                next_mask = torch.from_numpy(pred_bin.astype(np.float32))[None, ...]

            mask_hist.append(next_mask)

    return pd.DataFrame(rows)


def summarize_by_horizon(df_long: pd.DataFrame, label: str) -> pd.DataFrame:
    g = df_long.groupby("horizon", as_index=False).agg(
        soft_dice_mean=("soft_dice", "mean"),
        soft_dice_std=("soft_dice", "std"),
        bin_dice_mean=("bin_dice", "mean"),
        bin_dice_std=("bin_dice", "std"),
        bin_iou_mean=("bin_iou", "mean"),
        bin_iou_std=("bin_iou", "std"),
        n=("soft_dice", "count"),
    )
    g.insert(0, "label", label)
    return g


def plot_curves(summary: pd.DataFrame, out_png: Path, title: str) -> None:
    # summary has rows for possibly multiple labels
    labels = summary["label"].unique().tolist()

    plt.figure(figsize=(10, 6))
    for lab in labels:
        sub = summary[summary["label"] == lab].sort_values("horizon")
        plt.plot(sub["horizon"], sub["soft_dice_mean"], label=f"{lab} soft_dice")
    plt.title(title)
    plt.xlabel("forecast horizon (t+k)")
    plt.ylabel("soft dice (mean)")
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    for lab in labels:
        sub = summary[summary["label"] == lab].sort_values("horizon")
        plt.plot(sub["horizon"], sub["bin_iou_mean"], label=f"{lab} bin_iou")
    plt.title(title)
    plt.xlabel("forecast horizon (t+k)")
    plt.ylabel("binary IoU @ thr (mean)")
    plt.legend()
    plt.tight_layout()
    out_png2 = out_png.with_name(out_png.stem.replace("softdice", "biniou") + out_png.suffix)
    plt.savefig(out_png2, dpi=150)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument("--metrics-csv", required=True)
    ap.add_argument("--outdir", default=str(REPO_ROOT / "out" / "horizon"))

    # one or two checkpoints
    ap.add_argument("--ckpt", default=None, help="Evaluate a single checkpoint")
    ap.add_argument("--ckpt-a", default=None, help="Checkpoint A (e.g., baseline)")
    ap.add_argument("--ckpt-b", default=None, help="Checkpoint B (e.g., scheduled sampling)")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")

    # data/model options
    ap.add_argument("--use-rgb", action="store_true")
    ap.add_argument("--img-col", default=None)

    # eval knobs
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--n-samples", type=int, default=50, help="How many start windows to evaluate (0 = all)")
    ap.add_argument("--sample-stride", type=int, default=5, help="Step between start windows")
    ap.add_argument("--start-offset", type=int, default=0)
    ap.add_argument("--thr", type=float, default=0.30)
    ap.add_argument("--feedback", choices=["prob", "bin"], default="prob")

    # override model cfg if needed
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--image-size", type=int, default=None)
    ap.add_argument("--hid-ch", type=int, default=None)

    ap.add_argument("--device", default="cuda")

    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    df, ic = load_df(Path(args.metrics_csv).resolve(), img_col=args.img_col)

    # determine which mode
    items: List[Tuple[Path, str]] = []
    if args.ckpt:
        items.append((Path(args.ckpt).resolve(), Path(args.ckpt).stem))
    else:
        if not (args.ckpt_a and args.ckpt_b):
            raise ValueError("Provide either --ckpt OR both --ckpt-a and --ckpt-b.")
        items.append((Path(args.ckpt_a).resolve(), args.label_a))
        items.append((Path(args.ckpt_b).resolve(), args.label_b))

    all_summaries = []
    for ckpt_path, label in items:
        long_df = eval_one_ckpt(
            df=df,
            img_col=ic,
            ckpt_path=ckpt_path,
            use_rgb=bool(args.use_rgb),
            img_col_override=args.img_col,
            device=device,
            horizon=int(args.horizon),
            n_samples=int(args.n_samples),
            sample_stride=int(args.sample_stride),
            start_offset=int(args.start_offset),
            thr=float(args.thr),
            feedback=str(args.feedback),
            seq_len_arg=args.seq_len,
            image_size_arg=args.image_size,
            hid_ch_arg=args.hid_ch,
        )

        # save long form too (optional but useful)
        long_out = outdir / f"horizon_long_{label}.csv"
        long_df.to_csv(long_out, index=False)
        print(f"[done] wrote {long_out}")

        summary = summarize_by_horizon(long_df, label=label)
        all_summaries.append(summary)

    summary_all = pd.concat(all_summaries, ignore_index=True)
    tag = "rgb" if args.use_rgb else "gray"
    combo = "single" if args.ckpt else "compare"
    out_csv = outdir / f"horizon_metrics_{combo}_{tag}_K{args.horizon}_thr{args.thr}_{args.feedback}.csv"
    summary_all.to_csv(out_csv, index=False)
    print(f"[done] wrote {out_csv}")

    out_png = outdir / f"horizon_curves_softdice_{combo}_{tag}_K{args.horizon}_thr{args.thr}_{args.feedback}.png"
    plot_curves(summary_all, out_png, title=f"Horizon metrics ({tag}) K={args.horizon} thr={args.thr} fb={args.feedback}")
    print(f"[done] wrote {out_png} (and binIoU companion png)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
