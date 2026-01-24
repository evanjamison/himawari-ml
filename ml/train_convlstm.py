from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")  # safe on headless runners
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]


# -----------------------------
# Helpers
# -----------------------------
def _norm_path(s: str) -> str:
    return str(s).replace("\\", "/")


def _load_img(relpath: str) -> Image.Image:
    p = (REPO_ROOT / relpath).resolve()
    return Image.open(p)


def _load_mask_tensor(relpath: str, size: int) -> torch.Tensor:
    """
    Loads a mask PNG, converts to binary {0,1}, returns (1,H,W) float32.
    """
    im = _load_img(relpath).convert("L").resize((size, size), Image.Resampling.NEAREST)
    arr = np.asarray(im, dtype=np.uint8)
    m = (arr > 127).astype(np.float32)
    return torch.from_numpy(m)[None, ...]  # (1,H,W)


def _load_image_tensor(relpath: str, size: int, use_rgb: bool) -> torch.Tensor:
    """
    Loads a satellite image, returns:
      - grayscale: (1,H,W) float32 in [0,1]
      - rgb:       (3,H,W) float32 in [0,1]
    """
    im = _load_img(relpath)
    if use_rgb:
        im = im.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        arr = np.asarray(im, dtype=np.uint8)  # (H,W,3)
        x = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0  # (3,H,W)
        return x
    else:
        im = im.convert("L").resize((size, size), Image.Resampling.BILINEAR)
        arr = np.asarray(im, dtype=np.uint8)  # (H,W)
        x = torch.from_numpy(arr)[None, ...].float() / 255.0  # (1,H,W)
        return x


def _try_load_rgb_numpy(relpath: str, size: int) -> Optional[np.ndarray]:
    """
    For preview plotting only.
    """
    if not relpath:
        return None
    p = (REPO_ROOT / relpath).resolve()
    if not p.exists():
        return None
    im = Image.open(p).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def dice_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs.view(probs.size(0), -1)
    target = target.view(target.size(0), -1)
    inter = (probs * target).sum(dim=1)
    denom = probs.sum(dim=1) + target.sum(dim=1)
    return ((2 * inter + eps) / (denom + eps)).mean()


def iou_from_logits(logits: torch.Tensor, target: torch.Tensor, thr: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    pred = (probs > thr).float()
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    inter = (pred * target).sum(dim=1)
    union = (pred + target - pred * target).sum(dim=1)
    return ((inter + eps) / (union + eps)).mean()


# -----------------------------
# Dataset (multi-step ready)
# -----------------------------
class MaskImageSeqDatasetSS(Dataset):
    """
    Scheduled-sampling-ready dataset.

    For each start s, returns:
      masks_init: (T, 1, H, W)  for indices s..s+T-1
      imgs_all:   (T+K-1, Ic, H, W) for indices s..s+T+K-2
      y_future:   (K, 1, H, W)  for indices s+T..s+T+K-1

    CSV must contain:
      - mask_path
      - one image path col (auto-detected or --img-col)

    Optional:
      - timestamp_utc
    """
    def __init__(
        self,
        csv_path: Path,
        seq_len: int,
        image_size: int,
        stride: int = 1,
        rollout_k: int = 2,
        auto_shrink: bool = True,
        use_rgb: bool = False,
        img_col: Optional[str] = None,
    ):
        df = pd.read_csv(csv_path)

        if "mask_path" not in df.columns:
            raise ValueError("CSV must contain a 'mask_path' column (relative path to mask PNG).")

        # choose image column
        if img_col is not None:
            if img_col not in df.columns:
                raise ValueError(f"--img-col '{img_col}' not found in CSV columns: {list(df.columns)}")
            ic = img_col
        else:
            ic = None
            for cand in ["rgb_path", "relpath", "image_path", "frame_path"]:
                if cand in df.columns:
                    ic = cand
                    break
            if ic is None:
                raise ValueError(
                    "Your CSV must include one of: rgb_path, relpath, image_path, frame_path (or pass --img-col)."
                )

        # sort by time if available
        if "timestamp_utc" in df.columns:
            df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
            df = df.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc")
        else:
            df = df.copy()

        df["mask_path"] = df["mask_path"].astype(str).map(_norm_path)
        df[ic] = df[ic].astype(str).map(_norm_path)

        def _exists(rel: str) -> bool:
            return (REPO_ROOT / rel).exists()

        keep = df["mask_path"].map(_exists) & df[ic].map(_exists)
        df = df[keep].reset_index(drop=True)

        if len(df) < 2:
            raise ValueError(f"Not enough rows after filtering existing mask+image files. Need >=2, got {len(df)}.")

        seq_len = int(seq_len)
        rollout_k = int(max(1, rollout_k))
        need = seq_len + rollout_k  # need s..s+seq_len+K-1 exists (because y_future needs +K)

        if len(df) < need:
            if not auto_shrink:
                raise ValueError(f"Not enough rows. Need >= {need}, got {len(df)}.")
            # shrink seq_len first (keep K)
            new_seq_len = max(1, len(df) - rollout_k)
            print(f"[warn] shrinking seq_len from {seq_len} -> {new_seq_len} (frames={len(df)}, K={rollout_k})")
            seq_len = new_seq_len
            need = seq_len + rollout_k
            if len(df) < need:
                raise ValueError(f"Still not enough rows after shrink. frames={len(df)} need={need}")

        self.df = df
        self.seq_len = seq_len
        self.rollout_k = rollout_k
        self.image_size = int(image_size)
        self.stride = int(stride)
        self.use_rgb = bool(use_rgb)
        self.img_col = ic

        # Start positions must allow targets up to s+T+K-1
        # We will use images up to s+T+K-2 (T+K-1 images), and y up to s+T+K-1.
        self.starts = list(range(0, len(df) - (self.seq_len + self.rollout_k) + 1, self.stride))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int):
        s = self.starts[idx]
        T = self.seq_len
        K = self.rollout_k

        # initial masks for s..s+T-1
        mask_paths_init = self.df.loc[s : s + T - 1, "mask_path"].tolist()
        masks_init = torch.stack([_load_mask_tensor(p, self.image_size) for p in mask_paths_init], dim=0)  # (T,1,H,W)

        # images for s..s+T+K-2  (T+K-1 images)
        img_paths_all = self.df.loc[s : s + T + K - 2, self.img_col].tolist()
        imgs_all = torch.stack([_load_image_tensor(p, self.image_size, self.use_rgb) for p in img_paths_all], dim=0)  # (T+K-1,Ic,H,W)

        # future GT masks for steps 1..K: indices s+T .. s+T+K-1
        y_paths = self.df.loc[s + T : s + T + K - 1, "mask_path"].tolist()
        y_future = torch.stack([_load_mask_tensor(p, self.image_size) for p in y_paths], dim=0)  # (K,1,H,W)

        # preview RGB of t+1 target frame
        rgb_target = _try_load_rgb_numpy(self.df.loc[s + T, self.img_col], self.image_size)

        meta = {
            "start_index": int(s),
            "t_in_start": str(self.df.loc[s, "timestamp_utc"]) if "timestamp_utc" in self.df.columns else str(s),
            "t_target1": str(self.df.loc[s + T, "timestamp_utc"]) if "timestamp_utc" in self.df.columns else str(s + T),
            "img_col": self.img_col,
            "rgb": rgb_target,
        }

        return masks_init, imgs_all, y_future, meta


# -----------------------------
# ConvLSTM
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
    Input:  (B,T,C,H,W)  where C = 1(mask) + image channels
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
            nn.Conv2d(hid_ch, 1, 1),  # logits
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x_seq.shape
        h = c = None
        for t in range(T):
            feat = self.enc(x_seq[:, t])  # (B,hid,H,W)
            h, c = self.cell(feat, h, c)
        return self.head(h)


# -----------------------------
# Loss
# -----------------------------
def dice_loss_with_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs.view(probs.size(0), -1)
    target = target.view(target.size(0), -1)
    inter = (probs * target).sum(dim=1)
    denom = probs.sum(dim=1) + target.sum(dim=1)
    dice = (2 * inter + eps) / (denom + eps)
    return 1 - dice.mean()


def estimate_pos_weight(train_dl: DataLoader, device: torch.device, max_batches: int = 10) -> float:
    """
    Estimate pos_weight = neg/pos using GT target (step1) from a few batches.
    """
    pos = 0.0
    neg = 0.0
    seen = 0
    for _, _, y_future, _ in train_dl:
        y = y_future[:, 0].to(device)  # step1 GT
        pos += float(y.sum().item())
        neg += float((1.0 - y).sum().item())
        seen += 1
        if seen >= max_batches:
            break
    if pos <= 1.0:
        return 1.0
    return float(np.clip(neg / pos, 1.0, 50.0))


# -----------------------------
# Scheduled sampling schedule
# -----------------------------
def teacher_forcing_prob(epoch: int, warmup_epochs: int, decay_epochs: int, end_prob: float) -> float:
    """
    Returns probability of using GT as feedback (teacher forcing).
    - warmup: prob=1
    - then linearly decays to end_prob over decay_epochs
    """
    end_prob = float(np.clip(end_prob, 0.0, 1.0))
    warmup_epochs = int(max(0, warmup_epochs))
    decay_epochs = int(max(1, decay_epochs))

    if epoch <= warmup_epochs:
        return 1.0

    t = (epoch - warmup_epochs) / decay_epochs
    t = float(np.clip(t, 0.0, 1.0))
    return (1.0 - t) * 1.0 + t * end_prob


# -----------------------------
# Viz
# -----------------------------
def save_training_curves(hist: pd.DataFrame, out_png: Path) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(hist["epoch"], hist["train_loss"], label="train_loss")
    plt.plot(hist["epoch"], hist["val_loss"], label="val_loss")
    plt.title("ConvLSTM training curves")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()


@torch.no_grad()
def save_val_previews_step1(
    model: nn.Module,
    dl: DataLoader,
    device: torch.device,
    out_png: Path,
    thr: float = 0.30,
    max_rows: int = 3,
) -> None:
    """
    Preview only step1 (t+1) to keep plots comparable.
    """
    model.eval()
    rows = []

    for masks_init, imgs_all, y_future, meta in dl:
        masks_init = masks_init.to(device)      # (B,T,1,H,W)
        imgs_all = imgs_all.to(device)          # (B,T+K-1,Ic,H,W)
        y1 = y_future[:, 0].to(device)          # (B,1,H,W)

        T = masks_init.shape[1]
        img_window = imgs_all[:, 0:T]           # images aligned with initial window
        x = torch.cat([masks_init, img_window], dim=2)  # (B,T,1+Ic,H,W)

        logits = model(x)
        prob = torch.sigmoid(logits)
        pred_bin = (prob > thr).float()

        gt0 = y1[0, 0].detach().cpu().numpy()
        pr0 = prob[0, 0].detach().cpu().numpy()
        pb0 = pred_bin[0, 0].detach().cpu().numpy()
        rgb0 = None
        if isinstance(meta, dict) and "rgb" in meta:
            rgb0 = meta["rgb"][0]

        rows.append((rgb0, gt0, pr0, pb0))
        if len(rows) >= max_rows:
            break

    if not rows:
        return

    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(14, 4 * n))
    if n == 1:
        axes = np.array([axes])

    for i, (rgb, gt, prob, binm) in enumerate(rows):
        ax0, ax1, ax2, ax3 = axes[i]

        if rgb is not None and isinstance(rgb, np.ndarray):
            ax0.imshow(rgb)
            ax0.set_title("RGB (target frame)")
        else:
            ax0.imshow(gt, cmap="gray")
            ax0.set_title("GT mask (no RGB)")
        ax0.axis("off")

        ax1.imshow(gt, cmap="gray")
        ax1.set_title("GT mask")
        ax1.axis("off")

        ax2.imshow(prob, cmap="gray")
        ax2.set_title("Pred prob")
        ax2.axis("off")

        ax3.imshow(binm, cmap="gray")
        ax3.set_title(f"Pred bin (thr={thr})")
        ax3.axis("off")

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()


# -----------------------------
# Train
# -----------------------------
@dataclass
class TrainCfg:
    seq_len: int = 6
    rollout_k: int = 2
    image_size: int = 256
    stride: int = 1
    batch_size: int = 4
    epochs: int = 30
    lr: float = 3e-4
    hid_ch: int = 48
    bce_weight: float = 0.7
    dice_weight: float = 0.3
    seed: int = 1337
    val_frac: float = 0.2
    auto_pos_weight: bool = True
    pos_weight_cap: float = 50.0

    # scheduled sampling
    ss_warmup: int = 5
    ss_decay_epochs: int = 20
    ss_end: float = 0.3
    ss_feedback: str = "prob"  # "prob" or "bin"


def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument("--metrics-csv", required=True, help="CSV with mask_path + image-path column")
    ap.add_argument("--outdir", default=str(REPO_ROOT / "out"))

    ap.add_argument("--seq-len", type=int, default=6)
    ap.add_argument("--rollout-k", type=int, default=2, help="How many autoregressive steps to train (>=1).")
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--stride", type=int, default=1)

    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hid-ch", type=int, default=48)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.2)

    ap.add_argument("--no-auto-pos-weight", action="store_true")
    ap.add_argument("--preview-thr", type=float, default=0.30)
    ap.add_argument("--preview-rows", type=int, default=3)

    ap.add_argument("--use-rgb", action="store_true", help="Use RGB (3ch) imagery input instead of grayscale (1ch).")
    ap.add_argument("--img-col", default=None, help="Override image column name in CSV (e.g., rgb_path).")

    # scheduled sampling knobs
    ap.add_argument("--ss-warmup", type=int, default=5, help="Epochs of pure teacher forcing (GT feedback).")
    ap.add_argument("--ss-decay-epochs", type=int, default=20, help="Epochs to decay teacher forcing to --ss-end.")
    ap.add_argument("--ss-end", type=float, default=0.3, help="Final teacher-forcing probability.")
    ap.add_argument("--ss-feedback", choices=["prob", "bin"], default="prob", help="What to feed back when not using GT.")

    args = ap.parse_args()

    cfg = TrainCfg(
        seq_len=args.seq_len,
        rollout_k=max(1, args.rollout_k),
        image_size=args.image_size,
        stride=args.stride,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        hid_ch=args.hid_ch,
        val_frac=float(args.val_frac),
        auto_pos_weight=not bool(args.no_auto_pos_weight),
        ss_warmup=int(args.ss_warmup),
        ss_decay_epochs=int(args.ss_decay_epochs),
        ss_end=float(args.ss_end),
        ss_feedback=str(args.ss_feedback),
    )

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_root = Path(args.outdir).resolve()
    out_models = out_root / "models"
    out_viz = out_root / "viz"
    out_models.mkdir(parents=True, exist_ok=True)
    out_viz.mkdir(parents=True, exist_ok=True)

    ds = MaskImageSeqDatasetSS(
        Path(args.metrics_csv).resolve(),
        seq_len=cfg.seq_len,
        rollout_k=cfg.rollout_k,
        image_size=cfg.image_size,
        stride=cfg.stride,
        auto_shrink=True,
        use_rgb=bool(args.use_rgb),
        img_col=args.img_col,
    )

    if ds.seq_len != cfg.seq_len:
        cfg.seq_len = ds.seq_len
        print(f"[info] using seq_len={cfg.seq_len}")
    if ds.rollout_k != cfg.rollout_k:
        cfg.rollout_k = ds.rollout_k
        print(f"[info] using rollout_k={cfg.rollout_k}")

    n = len(ds)
    if n < 2:
        raise ValueError(f"Dataset too small (windows={n}). Need at least 2 windows.")

    n_val = max(1, int(round(cfg.val_frac * n)))
    n_train = n - n_val
    if n_train < 1:
        n_train = 1
        n_val = n - 1

    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val])
    print(f"Dataset windows: total={n} train={n_train} val={n_val} | device={device}")

    def collate(batch):
        masks_init, imgs_all, y_future, metas = zip(*batch)
        masks_init = torch.stack(masks_init, dim=0)  # (B,T,1,H,W)
        imgs_all = torch.stack(imgs_all, dim=0)      # (B,T+K-1,Ic,H,W)
        y_future = torch.stack(y_future, dim=0)      # (B,K,1,H,W)
        meta = {k: [m[k] for m in metas] for k in metas[0].keys()}
        return masks_init, imgs_all, y_future, meta

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate)

    # determine in_ch from dataset settings
    img_ch = 3 if args.use_rgb else 1
    in_ch = 1 + img_ch
    print(f"[info] model in_ch={in_ch} (mask + imagery), img_ch={img_ch}, rollout_k={cfg.rollout_k}")

    model = ConvLSTMNowcaster(in_ch=in_ch, hid_ch=cfg.hid_ch).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    if cfg.auto_pos_weight:
        pw = estimate_pos_weight(train_dl, device=device, max_batches=10)
        pw = float(np.clip(pw, 1.0, cfg.pos_weight_cap))
        print(f"[info] BCE pos_weight={pw:.2f} (auto)")
        bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=device))
    else:
        bce = nn.BCEWithLogitsLoss()

    best_val = float("inf")
    hist_rows = []

    tag = "rgb" if args.use_rgb else "gray"
    ckpt_path = out_models / f"convlstm_ss_best_{tag}.pt"

    for epoch in range(1, cfg.epochs + 1):
        tf_prob = teacher_forcing_prob(epoch, cfg.ss_warmup, cfg.ss_decay_epochs, cfg.ss_end)

        # ---- TRAIN ----
        model.train()
        tr_loss = 0.0
        tr_dice = 0.0
        tr_iou = 0.0
        tr_n = 0

        for masks_init, imgs_all, y_future, _ in train_dl:
            masks_init = masks_init.to(device)  # (B,T,1,H,W)
            imgs_all = imgs_all.to(device)      # (B,T+K-1,Ic,H,W)
            y_future = y_future.to(device)      # (B,K,1,H,W)

            B, T, _, H, W = masks_init.shape
            K = y_future.shape[1]

            # rolling mask history (B,T,1,H,W)
            mask_hist = masks_init

            loss_total = 0.0
            logits_step1 = None
            y_step1 = None

            for k in range(K):
                # image window aligned with current mask window
                img_window = imgs_all[:, k : k + T]  # (B,T,Ic,H,W)
                x = torch.cat([mask_hist, img_window], dim=2)  # (B,T,1+Ic,H,W)
                logits = model(x)  # (B,1,H,W)

                yk = y_future[:, k]  # (B,1,H,W)
                loss_k = cfg.bce_weight * bce(logits, yk) + cfg.dice_weight * dice_loss_with_logits(logits, yk)
                loss_total = loss_total + loss_k

                # Save step1 for reporting metrics
                if k == 0:
                    logits_step1 = logits
                    y_step1 = yk

                # choose feedback for next step (except after last step)
                if k < K - 1:
                    use_gt = (torch.rand(B, device=device) < tf_prob).float().view(B, 1, 1, 1)  # (B,1,1,1)

                    prob = torch.sigmoid(logits).detach()
                    if cfg.ss_feedback == "bin":
                        pred_fb = (prob > 0.30).float()  # fixed binarization for feedback stability
                    else:
                        pred_fb = prob

                    fb = use_gt * yk + (1.0 - use_gt) * pred_fb  # (B,1,H,W)

                    # shift window: drop oldest, append feedback
                    fb = fb.unsqueeze(1)  # (B,1,1,H,W)
                    mask_hist = torch.cat([mask_hist[:, 1:], fb], dim=1)

            loss_total = loss_total / K

            opt.zero_grad(set_to_none=True)
            loss_total.backward()
            opt.step()

            bs = masks_init.size(0)
            tr_loss += float(loss_total.item()) * bs

            # metrics only on step1 (t+1) to compare across runs
            tr_dice += float(dice_from_logits(logits_step1, y_step1).item()) * bs
            tr_iou += float(iou_from_logits(logits_step1, y_step1).item()) * bs
            tr_n += bs

        tr_loss /= max(1, tr_n)
        tr_dice /= max(1, tr_n)
        tr_iou /= max(1, tr_n)

        # ---- VAL (teacher forcing only, step1 metrics) ----
        model.eval()
        va_loss = 0.0
        va_dice = 0.0
        va_iou = 0.0
        va_n = 0

        with torch.no_grad():
            for masks_init, imgs_all, y_future, _ in val_dl:
                masks_init = masks_init.to(device)
                imgs_all = imgs_all.to(device)
                y_future = y_future.to(device)

                B, T, _, H, W = masks_init.shape
                K = y_future.shape[1]

                mask_hist = masks_init
                loss_total = 0.0
                logits_step1 = None
                y_step1 = None

                for k in range(K):
                    img_window = imgs_all[:, k : k + T]
                    x = torch.cat([mask_hist, img_window], dim=2)
                    logits = model(x)

                    yk = y_future[:, k]
                    loss_k = cfg.bce_weight * bce(logits, yk) + cfg.dice_weight * dice_loss_with_logits(logits, yk)
                    loss_total = loss_total + loss_k

                    if k == 0:
                        logits_step1 = logits
                        y_step1 = yk

                    if k < K - 1:
                        # pure teacher forcing in validation
                        fb = yk.unsqueeze(1)
                        mask_hist = torch.cat([mask_hist[:, 1:], fb], dim=1)

                loss_total = loss_total / K

                bs = masks_init.size(0)
                va_loss += float(loss_total.item()) * bs
                va_dice += float(dice_from_logits(logits_step1, y_step1).item()) * bs
                va_iou += float(iou_from_logits(logits_step1, y_step1).item()) * bs
                va_n += bs

        va_loss /= max(1, va_n)
        va_dice /= max(1, va_n)
        va_iou /= max(1, va_n)

        print(
            f"epoch {epoch:02d} | tf_prob={tf_prob:.2f} | "
            f"train loss {tr_loss:.4f} dice {tr_dice:.3f} iou {tr_iou:.3f} | "
            f"val loss {va_loss:.4f} dice {va_dice:.3f} iou {va_iou:.3f}"
        )

        hist_rows.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_loss": va_loss,
            "train_dice": tr_dice,
            "val_dice": va_dice,
            "train_iou": tr_iou,
            "val_iou": va_iou,
            "tf_prob": tf_prob,
            "seq_len": cfg.seq_len,
            "rollout_k": cfg.rollout_k,
            "image_size": cfg.image_size,
            "hid_ch": cfg.hid_ch,
            "use_rgb": bool(args.use_rgb),
            "ss_end": cfg.ss_end,
            "ss_feedback": cfg.ss_feedback,
        })

        if va_loss < best_val:
            best_val = va_loss
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch, "cfg": cfg.__dict__, "val_loss": best_val},
                ckpt_path,
            )
            print(f"[info] saved best checkpoint -> {ckpt_path}")

    hist = pd.DataFrame(hist_rows)
    hist_csv = out_root / f"convlstm_history_ss_{tag}.csv"
    hist.to_csv(hist_csv, index=False)

    curves_png = out_viz / f"convlstm_training_curves_ss_{tag}.png"
    save_training_curves(hist, curves_png)

    preview_png = out_viz / f"convlstm_val_previews_ss_{tag}.png"
    save_val_previews_step1(
        model, val_dl, device, preview_png,
        thr=float(args.preview_thr),
        max_rows=int(args.preview_rows),
    )

    print(f"[done] history: {hist_csv}")
    print(f"[done] curves:  {curves_png}")
    print(f"[done] preview: {preview_png}")
    print(f"[done] best ckpt: {ckpt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

