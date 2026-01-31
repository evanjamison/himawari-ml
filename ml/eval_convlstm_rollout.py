from __future__ import annotations

import argparse
from dataclasses import dataclass
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
# IO helpers (same conventions as train_convlstm.py)
# -----------------------------
def _norm_path(s: str) -> str:
    return str(s).replace("\\", "/")


def _abs_from_rel(relpath: str) -> Path:
    return (REPO_ROOT / relpath).resolve()


def _load_img(relpath: str) -> Image.Image:
    return Image.open(_abs_from_rel(relpath))


def _load_mask_np(relpath: str, size: int) -> np.ndarray:
    """
    Load mask PNG -> binary {0,1} float32, shape (H,W).
    """
    im = _load_img(relpath).convert("L").resize((size, size), Image.Resampling.NEAREST)
    arr = np.asarray(im, dtype=np.uint8)
    m = (arr > 127).astype(np.float32)
    return m


def _load_mask_tensor(relpath: str, size: int) -> torch.Tensor:
    m = _load_mask_np(relpath, size)
    # ensure writable (prevents torch "not writable" warning)
    m = m.copy()
    return torch.from_numpy(m)[None, ...]  # (1,H,W)


def _load_image_tensor(relpath: str, size: int, use_rgb: bool) -> torch.Tensor:
    """
    Load image -> float32 tensor in [0,1]
      gray: (1,H,W)
      rgb:  (3,H,W)
    """
    im = _load_img(relpath)
    if use_rgb:
        im = im.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        arr = np.asarray(im, dtype=np.uint8).copy()  # (H,W,3)
        x = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        return x
    else:
        im = im.convert("L").resize((size, size), Image.Resampling.BILINEAR)
        arr = np.asarray(im, dtype=np.uint8).copy()  # (H,W)
        x = torch.from_numpy(arr)[None, ...].float() / 255.0
        return x


def _try_load_rgb_np(relpath: str, size: int) -> Optional[np.ndarray]:
    p = _abs_from_rel(relpath)
    if not p.exists():
        return None
    im = Image.open(p).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


# -----------------------------
# Metrics
# -----------------------------
def _iou(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    pred = (pred > 0.5).astype(np.float32)
    gt = (gt > 0.5).astype(np.float32)
    inter = float((pred * gt).sum())
    union = float((pred + gt - pred * gt).sum())
    return (inter + eps) / (union + eps)


def _dice(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    pred = (pred > 0.5).astype(np.float32)
    gt = (gt > 0.5).astype(np.float32)
    inter = float((pred * gt).sum())
    denom = float(pred.sum() + gt.sum())
    return (2.0 * inter + eps) / (denom + eps)


def _cloud_frac(m: np.ndarray) -> float:
    return float((m > 0.5).mean())


# -----------------------------
# ConvLSTM model (copy of train_convlstm.py)
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
    Input:  (B,T,C,H,W) where C = 1(mask) + img channels
    Output: (B,1,H,W) logits for next mask
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
# Sequence building from metrics CSV
# -----------------------------
@dataclass
class SeqWindow:
    start: int
    mask_init_paths: List[str]         # len T, indices s..s+T-1
    img_all_paths: List[str]           # len (T+K-1), indices s..s+T+K-2
    y_future_paths: List[str]          # len K, indices s+T..s+T+K-1
    t_in: str
    t_targets: List[str]              # len K


def _pick_image_col(df: pd.DataFrame, img_col: Optional[str]) -> str:
    if img_col is not None:
        if img_col not in df.columns:
            raise ValueError(f"--img-col '{img_col}' not in CSV columns: {list(df.columns)}")
        return img_col
    for cand in ["rgb_path", "relpath", "image_path", "frame_path"]:
        if cand in df.columns:
            return cand
    raise ValueError("CSV must include one of rgb_path / relpath / image_path / frame_path (or pass --img-col).")


def _load_and_clean_df(csv_path: Path, img_col: Optional[str]) -> Tuple[pd.DataFrame, str]:
    df = pd.read_csv(csv_path)
    if "mask_path" not in df.columns:
        raise ValueError("CSV must contain 'mask_path' column.")

    ic = _pick_image_col(df, img_col)

    # sort by timestamp if exists
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
        df = df.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc")
        tcol = "timestamp_utc"
    else:
        df = df.copy()
        tcol = None

    df["mask_path"] = df["mask_path"].astype(str).map(_norm_path)
    df[ic] = df[ic].astype(str).map(_norm_path)

    def _exists(rel: str) -> bool:
        return _abs_from_rel(rel).exists()

    keep = df["mask_path"].map(_exists) & df[ic].map(_exists)
    df = df[keep].reset_index(drop=True)

    if len(df) < 4:
        raise ValueError(f"Not enough rows after filtering existing files: {len(df)}")

    # add a stable time label for reporting
    if tcol is None:
        df["_tlabel"] = df.index.astype(str)
    else:
        df["_tlabel"] = df[tcol].astype(str)

    return df, ic


def _build_windows(df: pd.DataFrame, ic: str, T: int, K: int, stride: int) -> List[SeqWindow]:
    T = int(T)
    K = int(max(1, K))
    stride = int(max(1, stride))

    # Need indices through s+T+K-1 for y_future
    max_start = len(df) - (T + K)  # inclusive start max
    if max_start < 0:
        raise ValueError(f"Not enough rows for T={T}, K={K}. rows={len(df)} need>={T+K}")

    starts = list(range(0, max_start + 1, stride))
    windows: List[SeqWindow] = []
    for s in starts:
        mask_init = df.loc[s : s + T - 1, "mask_path"].tolist()
        img_all = df.loc[s : s + T + K - 2, ic].tolist()          # T+K-1 images
        y_future = df.loc[s + T : s + T + K - 1, "mask_path"].tolist()
        t_targets = df.loc[s + T : s + T + K - 1, "_tlabel"].tolist()

        windows.append(
            SeqWindow(
                start=int(s),
                mask_init_paths=mask_init,
                img_all_paths=img_all,
                y_future_paths=y_future,
                t_in=str(df.loc[s, "_tlabel"]),
                t_targets=[str(x) for x in t_targets],
            )
        )
    return windows


# -----------------------------
# Rollout eval
# -----------------------------
@torch.no_grad()
def _predict_rollout(
    model: nn.Module,
    win: SeqWindow,
    image_size: int,
    use_rgb: bool,
    thr: float,
    feedback: str,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Closed-loop rollout for one window.
    Returns probs and bins for each horizon j=1..K
    """
    model.eval()

    T = len(win.mask_init_paths)
    K = len(win.y_future_paths)

    # Load initial mask history (T,1,H,W)
    masks_init = torch.stack([_load_mask_tensor(p, image_size) for p in win.mask_init_paths], dim=0).to(device)

    # Load images needed for rollout (T+K-1, Ic, H, W)
    imgs_all = torch.stack([_load_image_tensor(p, image_size, use_rgb) for p in win.img_all_paths], dim=0).to(device)

    # GT future masks (K,1,H,W) as numpy
    gt_future_np = [ _load_mask_np(p, image_size) for p in win.y_future_paths ]

    # mask_hist: (T,1,H,W)
    mask_hist = masks_init.clone()

    probs: List[np.ndarray] = []
    bins: List[np.ndarray] = []

    for j in range(K):
        # image window for step j: imgs_all[j : j+T]  (T, Ic, H, W)
        img_window = imgs_all[j : j + T]
        x = torch.cat([mask_hist, img_window], dim=1)  # (T, 1+Ic, H, W)
        x = x.unsqueeze(0)  # (1,T,1+Ic,H,W)

        logits = model(x)              # (1,1,H,W)
        prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
        pb = (prob >= thr).astype(np.float32)

        probs.append(prob)
        bins.append(pb)

        # feedback for next step (except after last)
        if j < K - 1:
            if feedback == "bin":
                fb = torch.from_numpy(pb.copy())[None, ...].to(device)  # (1,H,W)
            else:
                fb = torch.from_numpy(prob.copy())[None, ...].to(device)

            # shift: drop oldest, append fb
            fb = fb.unsqueeze(0)  # (1,1,H,W) but we need (1,1,H,W) per time
            # mask_hist is (T,1,H,W)
            mask_hist = torch.cat([mask_hist[1:], fb], dim=0)

    return {
        "probs": probs,
        "bins": bins,
        "gt": gt_future_np,
    }


def _persistence_baseline(win: SeqWindow, image_size: int, thr: float) -> List[np.ndarray]:
    """
    Persistence: predict Y_t (last mask in context) for all horizons.
    """
    last = _load_mask_np(win.mask_init_paths[-1], image_size)
    # already binary 0/1, but keep as float32
    return [last.copy() for _ in win.y_future_paths]


def _make_contact_sheet(
    out_png: Path,
    rows: List[Dict[str, Any]],
    thr: float,
) -> None:
    """
    rows: list of dicts with keys:
      - rgb: np.uint8(H,W,3) or None
      - j: horizon int
      - gt: np(H,W)
      - pers: np(H,W)
      - prob: np(H,W)
      - bin: np(H,W)
      - label: str
    """
    if not rows:
        return

    # group by sample label, then horizons
    # We'll render one row per horizon per sample (so sample_count*K rows).
    n = len(rows)
    fig, axes = plt.subplots(n, 5, figsize=(16, 3.2 * n))
    if n == 1:
        axes = np.array([axes])

    for i, r in enumerate(rows):
        ax0, ax1, ax2, ax3, ax4 = axes[i]

        rgb = r.get("rgb", None)
        if isinstance(rgb, np.ndarray):
            ax0.imshow(rgb)
        else:
            ax0.imshow(r["gt"], cmap="gray")
        ax0.set_title(f'{r["label"]}\nRGB @ t+{r["j"]}')
        ax0.axis("off")

        ax1.imshow(r["gt"], cmap="gray")
        ax1.set_title(f"GT mask (t+{r['j']})")
        ax1.axis("off")

        ax2.imshow(r["pers"], cmap="gray")
        ax2.set_title("Persistence")
        ax2.axis("off")

        ax3.imshow(r["prob"], cmap="gray")
        ax3.set_title("ConvLSTM prob")
        ax3.axis("off")

        ax4.imshow(r["bin"], cmap="gray")
        ax4.set_title(f"ConvLSTM bin (thr={thr})")
        ax4.axis("off")

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument("--metrics-csv", required=True, help="Phase4 metrics CSV with mask_path + image-path column")
    ap.add_argument("--ckpt", required=True, help="ConvLSTM checkpoint (.pt) from train_convlstm.py")
    ap.add_argument("--outdir", required=True, help="Output directory for rollout_eval artifacts")

    ap.add_argument("--img-col", default=None, help="Override image column name in CSV (e.g., rgb_path)")
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--seq-len", type=int, default=6, help="Context length T (must match training)")
    ap.add_argument("--rollout-k", type=int, default=2, help="Eval horizon K")
    ap.add_argument("--stride", type=int, default=1)

    ap.add_argument("--val-frac", type=float, default=0.2, help="Use last val-frac of windows for evaluation")
    ap.add_argument("--max-windows", type=int, default=0, help="Optional cap on eval windows (0=all)")
    ap.add_argument("--thr", type=float, default=0.30)
    ap.add_argument("--feedback", choices=["prob", "bin"], default="prob", help="Closed-loop feedback type")

    ap.add_argument("--use-rgb", action="store_true", help="Force RGB input (overrides ckpt cfg if set)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])

    ap.add_argument("--viz-samples", type=int, default=3, help="How many windows (anchors) to visualize")
    ap.add_argument("--viz-evenly-spaced", action="store_true", help="Pick viz anchors evenly across eval set")

    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # device
    if args.device == "cuda":
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load ckpt + cfg
    ckpt = torch.load(Path(args.ckpt).resolve(), map_location="cpu")
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError("Checkpoint must be a dict with key 'model_state_dict' (from train_convlstm.py).")

    cfg = ckpt.get("cfg", {}) or {}
    hid_ch = int(cfg.get("hid_ch", 48))
    ckpt_use_rgb = bool(cfg.get("use_rgb", False))

    # decide rgb
    use_rgb = bool(args.use_rgb) or ckpt_use_rgb

    # model in_ch
    img_ch = 3 if use_rgb else 1
    in_ch = 1 + img_ch

    model = ConvLSTMNowcaster(in_ch=in_ch, hid_ch=hid_ch)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.to(device)

    # load df and build windows
    df, ic = _load_and_clean_df(Path(args.metrics_csv).resolve(), args.img_col)

    T = int(args.seq_len)
    K = int(max(1, args.rollout_k))
    windows = _build_windows(df, ic=ic, T=T, K=K, stride=int(args.stride))

    if len(windows) < 2:
        raise ValueError(f"Not enough windows to evaluate. windows={len(windows)}")

    # eval set = last val_frac of windows (time-safe)
    n = len(windows)
    n_val = max(1, int(round(float(args.val_frac) * n)))
    eval_windows = windows[-n_val:]

    if args.max_windows and int(args.max_windows) > 0:
        eval_windows = eval_windows[: int(args.max_windows)]

    print(f"[eval] rows={len(df)} windows_total={n} windows_eval={len(eval_windows)}")
    print(f"[eval] img_col={ic} use_rgb={use_rgb} in_ch={in_ch} hid_ch={hid_ch} device={device}")
    print(f"[eval] T={T} K={K} thr={args.thr} feedback={args.feedback}")

    # select viz anchors
    viz_n = max(0, int(args.viz_samples))
    viz_rows: List[Dict[str, Any]] = []
    if viz_n > 0:
        if args.viz_evenly_spaced and len(eval_windows) > 1:
            idxs = np.linspace(0, len(eval_windows) - 1, num=min(viz_n, len(eval_windows))).round().astype(int).tolist()
        else:
            idxs = list(range(min(viz_n, len(eval_windows))))

        viz_set = {int(i) for i in idxs}
    else:
        viz_set = set()

    # run eval
    metrics_rows: List[Dict[str, Any]] = []

    for wi, win in enumerate(eval_windows):
        # baselines + model
        pers_preds = _persistence_baseline(win, image_size=int(args.image_size), thr=float(args.thr))
        out = _predict_rollout(
            model=model,
            win=win,
            image_size=int(args.image_size),
            use_rgb=use_rgb,
            thr=float(args.thr),
            feedback=str(args.feedback),
            device=device,
        )

        gt = out["gt"]            # list len K (H,W)
        probs = out["probs"]      # list len K (H,W) float in [0,1]
        bins = out["bins"]        # list len K (H,W) 0/1

        # For visualization, load RGB of the target frame at each horizon if possible
        # We'll use the frame path from df at the corresponding t+1, t+2, ...
        # For simplicity: use win.img_all_paths[T-1 + j] which corresponds to frame at t+j (target)
        # img_all covers indices s..s+T+K-2, so target t+1 is at index s+T, which is img_all index T
        # But img_all_paths list maps s..s+T+K-2 => target t+1 is position T (0-based).
        # For each horizon j=1..K, target frame index in img_all_paths is T-1 + j.
        rgbs = []
        for j in range(1, K + 1):
            idx = (T - 1) + j
            if 0 <= idx < len(win.img_all_paths):
                rgbs.append(_try_load_rgb_np(win.img_all_paths[idx], int(args.image_size)))
            else:
                rgbs.append(None)

        # record per-horizon metrics
        for j in range(1, K + 1):
            gtj = gt[j - 1]
            persj = pers_preds[j - 1]
            binj = bins[j - 1]

            metrics_rows.append({
                "window_start": win.start,
                "t_in": win.t_in,
                "t_target": win.t_targets[j - 1],
                "horizon": j,
                "method": "persistence",
                "iou": _iou(persj, gtj),
                "dice": _dice(persj, gtj),
                "cloud_frac_pred": _cloud_frac(persj),
                "cloud_frac_gt": _cloud_frac(gtj),
                "cloud_frac_abs_err": abs(_cloud_frac(persj) - _cloud_frac(gtj)),
            })
            metrics_rows.append({
                "window_start": win.start,
                "t_in": win.t_in,
                "t_target": win.t_targets[j - 1],
                "horizon": j,
                "method": "convlstm",
                "iou": _iou(binj, gtj),
                "dice": _dice(binj, gtj),
                "cloud_frac_pred": _cloud_frac(binj),
                "cloud_frac_gt": _cloud_frac(gtj),
                "cloud_frac_abs_err": abs(_cloud_frac(binj) - _cloud_frac(gtj)),
            })

            # add to contact sheet if chosen
            if wi in viz_set:
                label = f"win@{win.t_in}"
                viz_rows.append({
                    "label": label,
                    "j": j,
                    "rgb": rgbs[j - 1],
                    "gt": gtj,
                    "pers": persj,
                    "prob": probs[j - 1],
                    "bin": binj,
                })

    # write per-window metrics
    mdf = pd.DataFrame(metrics_rows)
    out_metrics = outdir / "rollout_metrics.csv"
    mdf.to_csv(out_metrics, index=False)

    # summary by horizon+method
    g = mdf.groupby(["method", "horizon"], as_index=False).agg(
        iou_mean=("iou", "mean"),
        iou_std=("iou", "std"),
        dice_mean=("dice", "mean"),
        dice_std=("dice", "std"),
        cloud_frac_abs_err_mean=("cloud_frac_abs_err", "mean"),
        cloud_frac_abs_err_std=("cloud_frac_abs_err", "std"),
        n=("iou", "count"),
    )
    out_summary = outdir / "rollout_summary.csv"
    g.to_csv(out_summary, index=False)

    # contact sheet
    out_png = outdir / "rollout_contact_sheet.png"
    _make_contact_sheet(out_png, viz_rows, thr=float(args.thr))

    # quick console summary
    print("[done] wrote:", out_metrics)
    print("[done] wrote:", out_summary)
    if viz_rows:
        print("[done] wrote:", out_png)
    else:
        print("[done] no viz rows (set --viz-samples > 0)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
