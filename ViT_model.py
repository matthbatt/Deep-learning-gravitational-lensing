"""
ViT for Gravitational Lensing Parameter Estimation
====================================================
Predicts 5 lensing parameters from 140×140 GRIZ images:
  - Einstein radius (θ_E)
  - External shear (γ₁, γ₂)
  - Complex ellipticity (e₁, e₂)

Dataset: ~40k simulated strongly-lensed galaxy images, 4-band (GRIZ)
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from pathlib import Path


# ─────────────────────────────────────────────
# 1.  Model components
# ─────────────────────────────────────────────

class MLP(nn.Module):
    """Feed-forward block with GELU activation and optional dropout."""
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Multi-head self-attention with optional attention dropout."""
    def __init__(self, dim: int, heads: int = 6, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # [B, N, 3, heads, head_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)            # each [B, N, heads, head_dim]
        q = q.transpose(1, 2)                   # [B, heads, N, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale   # [B, heads, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(out))


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block (LayerNorm before each sub-layer)."""
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = Attention(dim, heads, attn_drop, proj_drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = MLP(dim, int(dim * mlp_ratio), dropout=proj_drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ─────────────────────────────────────────────
# 2.  Vision Transformer
# ─────────────────────────────────────────────

class LensingViT(nn.Module):
    """
    Vision Transformer for gravitational lensing parameter regression.

    Args
    ----
    img_size  : int   – spatial size of the input image (default 140)
    patch     : int   – patch size in pixels (default 14 → 10×10 = 100 patches)
    in_ch     : int   – number of input channels, 4 for GRIZ (default 4)
    dim       : int   – token embedding dimension (default 192)
    depth     : int   – number of Transformer blocks (default 6)
    heads     : int   – number of attention heads (default 6)
    mlp_ratio : float – MLP hidden dim multiplier (default 4.0)
    out_dim   : int   – number of regression targets (default 5)
    dropout   : float – dropout rate applied in MLP & projection layers
    """

    def __init__(
        self,
        img_size:  int   = 140,
        patch:     int   = 14,
        in_ch:     int   = 4,
        dim:       int   = 192,
        depth:     int   = 6,
        heads:     int   = 6,
        mlp_ratio: float = 4.0,
        out_dim:   int   = 5,
        dropout:   float = 0.1,
    ):
        super().__init__()
        assert img_size % patch == 0, "img_size must be divisible by patch"
        self.grid        = img_size // patch
        num_patches      = self.grid * self.grid

        # ── Patch embedding ──────────────────────────────────────
        # Conv2d with kernel=stride=patch acts as a non-overlapping patch tokeniser
        self.patch_embed = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)

        # ── Learnable tokens & positional embedding ───────────────
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.pos_drop  = nn.Dropout(dropout)

        # ── Transformer encoder ───────────────────────────────────
        self.blocks = nn.Sequential(*[
            TransformerBlock(dim, heads, mlp_ratio, attn_drop=dropout, proj_drop=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

        # ── Regression head ───────────────────────────────────────
        # Two-layer MLP head is more expressive than a single linear layer
        self.head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, out_dim),
        )

        self._init_weights()

    # ── Weight initialisation ────────────────────────────────────
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Forward pass ─────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, 4, 140, 140]  (GRIZ images, normalised)
        returns : [B, 5]       (θ_E, γ₁, γ₂, e₁, e₂)
        """
        B = x.shape[0]

        # Patchify  →  [B, dim, grid, grid]  →  [B, N, dim]
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1)          # [B, N+1, dim]

        # Add positional embedding
        x = self.pos_drop(x + self.pos_embed)

        # Transformer encoder
        x = self.blocks(x)
        x = self.norm(x)

        # Regression from CLS token
        return self.head(x[:, 0])               # [B, 5]


# ─────────────────────────────────────────────
# 3.  Dataset
# ─────────────────────────────────────────────

class LensingDataset(Dataset):
    """
    Minimal dataset wrapper.

    Expected layout (adjust `load_sample` to your actual file format):
        images : np.ndarray  shape [N, 4, 140, 140]  float32
        labels : np.ndarray  shape [N, 5]             float32
                             columns: [theta_E, gamma1, gamma2, e1, e2]

    Pass normalisation statistics (per-channel mean/std) computed on
    the TRAINING split only to avoid data leakage.
    """

    def __init__(
        self,
        images:   np.ndarray,
        labels:   np.ndarray,
        mean:     np.ndarray | None = None,
        std:      np.ndarray | None = None,
        augment:  bool = False,
    ):
        assert images.shape[0] == labels.shape[0]
        self.images  = torch.from_numpy(images.astype(np.float32))
        self.labels  = torch.from_numpy(labels.astype(np.float32))
        self.augment = augment

        # Per-channel normalisation  [4, 1, 1] broadcastable
        if mean is not None and std is not None:
            self.mean = torch.tensor(mean, dtype=torch.float32).view(4, 1, 1)
            self.std  = torch.tensor(std,  dtype=torch.float32).view(4, 1, 1)
        else:
            self.mean = self.images.mean(dim=(0, 2, 3)).view(4, 1, 1)
            self.std  = self.images.std(dim=(0, 2, 3)).view(4, 1, 1).clamp(min=1e-6)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = (self.images[idx] - self.mean) / self.std

        # ── Data augmentation (training only) ────────────────────
        if self.augment:
            # Random horizontal flip
            if torch.rand(1).item() > 0.5:
                img = img.flip(-1)
                # γ₁ flips sign under horizontal flip; e₁ flips sign
                # Adjust labels accordingly:
                # [theta_E, gamma1, gamma2, e1, e2] → [θE, -γ1, γ2, -e1, e2]
                lbl = self.labels[idx].clone()
                lbl[1] = -lbl[1]   # gamma1
                lbl[3] = -lbl[3]   # e1
            else:
                lbl = self.labels[idx]

            # Random vertical flip
            if torch.rand(1).item() > 0.5:
                img = img.flip(-2)
                lbl = lbl.clone()
                lbl[2] = -lbl[2]   # gamma2
                lbl[4] = -lbl[4]   # e2

            # Random 90° rotation (k ∈ {0,1,2,3})
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                img = torch.rot90(img, k, [-2, -1])
                # Rotate shear / ellipticity vectors by k×90°
                angle = k * math.pi / 2
                cos_a, sin_a = math.cos(2 * angle), math.sin(2 * angle)
                lbl = lbl.clone()
                g1, g2 = lbl[1].item(), lbl[2].item()
                e1, e2 = lbl[3].item(), lbl[4].item()
                lbl[1] = cos_a * g1 - sin_a * g2
                lbl[2] = sin_a * g1 + cos_a * g2
                lbl[3] = cos_a * e1 - sin_a * e2
                lbl[4] = sin_a * e1 + cos_a * e2
        else:
            lbl = self.labels[idx]

        return img, lbl


# ─────────────────────────────────────────────
# 4.  Loss
# ─────────────────────────────────────────────

class WeightedMSELoss(nn.Module):
    """
    Weighted MSE loss to handle parameters with very different scales.

    Default weights upweight θ_E (most physically critical) and down-weight
    the shear / ellipticity components which are typically smaller in
    magnitude.  Tune these to the variance of each parameter in your dataset.
    """
    def __init__(self, weights: list[float] | None = None):
        super().__init__()
        if weights is None:
            weights = [1.0, 0.5, 0.5, 0.5, 0.5]   # [theta_E, γ1, γ2, e1, e2]
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        sq_err = (pred - target) ** 2               # [B, 5]
        return (sq_err * self.weights).mean()


# ─────────────────────────────────────────────
# 5.  Training & evaluation utilities
# ─────────────────────────────────────────────

PARAM_NAMES = ["theta_E", "gamma1", "gamma2", "e1", "e2"]


def train_one_epoch(model, loader, optimizer, loss_fn, device, scheduler=None):
    model.train()
    total_loss = 0.0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        pred = model(imgs)
        loss = loss_fn(pred, lbls)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    if scheduler is not None:
        scheduler.step()
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss  = 0.0
    all_preds   = []
    all_targets = []
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        pred = model(imgs)
        total_loss += loss_fn(pred, lbls).item() * imgs.size(0)
        all_preds.append(pred.cpu())
        all_targets.append(lbls.cpu())

    preds   = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    rmse    = ((preds - targets) ** 2).mean(0).sqrt()   # per-parameter RMSE

    return total_loss / len(loader.dataset), dict(zip(PARAM_NAMES, rmse.tolist()))


def plot_predictions(preds, targets, save_path="predictions.png"):
    """Scatter-plot predicted vs true for each of the 5 parameters."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for i, (ax, name) in enumerate(zip(axes, PARAM_NAMES)):
        ax.scatter(targets[:, i], preds[:, i], s=2, alpha=0.3, rasterized=True)
        lo = min(targets[:, i].min(), preds[:, i].min())
        hi = max(targets[:, i].max(), preds[:, i].max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.5)
        ax.set_xlabel(f"True {name}", fontsize=11)
        ax.set_ylabel(f"Pred {name}", fontsize=11)
        ax.set_title(name, fontsize=12)
        r2 = np.corrcoef(targets[:, i], preds[:, i])[0, 1] ** 2
        ax.text(0.05, 0.92, f"R²={r2:.3f}", transform=ax.transAxes, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved prediction plots → {save_path}")


# ─────────────────────────────────────────────
# 6.  Main training script
# ─────────────────────────────────────────────

def main():
    # ── Configuration ────────────────────────────────────────────
    CFG = dict(
        # Data
        data_path    = "lensing_data.npz",   # update to your actual path
        img_size     = 140,
        in_ch        = 4,
        out_dim      = 5,
        # Model
        patch        = 14,    # → 10×10 = 100 patches
        dim          = 192,
        depth        = 6,
        heads        = 6,
        mlp_ratio    = 4.0,
        dropout      = 0.1,
        # Training
        epochs       = 100,
        batch_size   = 64,
        lr           = 3e-4,
        weight_decay = 0.05,
        val_frac     = 0.1,
        test_frac    = 0.1,
        num_workers  = 4,
        seed         = 42,
        save_dir     = "checkpoints",
    )

    torch.manual_seed(CFG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load data ─────────────────────────────────────────────────
    # Expects a .npz with keys 'images' [N,4,140,140] and 'labels' [N,5]
    # Adjust this block to your actual data format:
    print(f"Loading data from {CFG['data_path']} …")
    data    = np.load(CFG["data_path"])
    images  = data["images"]    # float32 or will be cast
    labels  = data["labels"]    # float32 or will be cast
    N       = len(images)
    print(f"  Total samples : {N}")

    # ── Train / val / test split ──────────────────────────────────
    n_val  = int(N * CFG["val_frac"])
    n_test = int(N * CFG["test_frac"])
    n_tr   = N - n_val - n_test

    rng   = np.random.default_rng(CFG["seed"])
    idx   = rng.permutation(N)
    tr_idx, val_idx, te_idx = idx[:n_tr], idx[n_tr:n_tr+n_val], idx[n_tr+n_val:]

    # Compute normalisation stats on TRAINING images only
    tr_imgs = images[tr_idx]
    mean = tr_imgs.mean(axis=(0, 2, 3))   # [4]
    std  = tr_imgs.std(axis=(0, 2, 3)).clip(1e-6)

    train_ds = LensingDataset(images[tr_idx], labels[tr_idx], mean, std, augment=True)
    val_ds   = LensingDataset(images[val_idx], labels[val_idx], mean, std, augment=False)
    test_ds  = LensingDataset(images[te_idx],  labels[te_idx],  mean, std, augment=False)

    print(f"  Train {len(train_ds)} | Val {len(val_ds)} | Test {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,
                              num_workers=CFG["num_workers"], pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=CFG["batch_size"], shuffle=False,
                              num_workers=CFG["num_workers"], pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=CFG["batch_size"], shuffle=False,
                              num_workers=CFG["num_workers"], pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────
    model = LensingViT(
        img_size  = CFG["img_size"],
        patch     = CFG["patch"],
        in_ch     = CFG["in_ch"],
        dim       = CFG["dim"],
        depth     = CFG["depth"],
        heads     = CFG["heads"],
        mlp_ratio = CFG["mlp_ratio"],
        out_dim   = CFG["out_dim"],
        dropout   = CFG["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters : {n_params:,}")

    # ── Optimiser & scheduler ─────────────────────────────────────
    # Separate weight decay: don't decay biases, LayerNorm, embeddings
    no_decay = {"bias", "norm", "cls_token", "pos_embed"}
    param_groups = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         "weight_decay": CFG["weight_decay"]},
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(param_groups, lr=CFG["lr"])
    scheduler = CosineAnnealingLR(optimizer, T_max=CFG["epochs"], eta_min=1e-6)

    loss_fn = WeightedMSELoss()

    # ── Training loop ─────────────────────────────────────────────
    save_dir = Path(CFG["save_dir"])
    save_dir.mkdir(exist_ok=True)

    best_val_loss = float("inf")
    history = {"train": [], "val": []}

    for epoch in range(1, CFG["epochs"] + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scheduler)
        val_loss, val_rmse = evaluate(model, val_loader, loss_fn, device)
        history["train"].append(tr_loss)
        history["val"].append(val_loss)

        rmse_str = "  ".join(f"{k}={v:.4f}" for k, v in val_rmse.items())
        print(f"Epoch {epoch:03d}/{CFG['epochs']}  "
              f"train={tr_loss:.5f}  val={val_loss:.5f}  |  RMSE:  {rmse_str}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt = {
                "epoch"      : epoch,
                "model"      : model.state_dict(),
                "optimizer"  : optimizer.state_dict(),
                "val_loss"   : val_loss,
                "val_rmse"   : val_rmse,
                "cfg"        : CFG,
                "norm_mean"  : mean,
                "norm_std"   : std,
            }
            torch.save(ckpt, save_dir / "best.pt")
            print(f"  ✓ Saved best checkpoint (val={val_loss:.5f})")

    # ── Test evaluation ───────────────────────────────────────────
    ckpt = torch.load(save_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    test_loss, test_rmse = evaluate(model, test_loader, loss_fn, device)
    print("\n── Test results ──────────────────────────────────────────")
    print(f"  Loss : {test_loss:.5f}")
    for k, v in test_rmse.items():
        print(f"  RMSE {k:>10s} : {v:.5f}")

    # ── Prediction scatter plots ──────────────────────────────────
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            all_preds.append(model(imgs.to(device)).cpu().numpy())
            all_targets.append(lbls.numpy())
    preds   = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    plot_predictions(preds, targets, save_path=str(save_dir / "predictions.png"))

    # ── Loss curve ────────────────────────────────────────────────
    plt.figure(figsize=(8, 4))
    plt.plot(history["train"], label="train")
    plt.plot(history["val"],   label="val")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.tight_layout()
    plt.savefig(str(save_dir / "loss_curve.png"), dpi=150)
    plt.close()
    print(f"Saved loss curve → {save_dir}/loss_curve.png")


# ─────────────────────────────────────────────
# 7.  Quick sanity check (no data needed)
# ─────────────────────────────────────────────

def sanity_check():
    """Run a forward + backward pass with random data to verify the model."""
    device = torch.device("cpu")
    model  = LensingViT().to(device)
    x      = torch.randn(4, 4, 140, 140)          # batch of 4 fake images
    y      = torch.randn(4, 5)                     # fake labels
    pred   = model(x)
    assert pred.shape == (4, 5), f"Expected (4,5), got {pred.shape}"
    loss   = F.mse_loss(pred, y)
    loss.backward()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Sanity check passed | output shape {pred.shape} | "
          f"{n_params:,} parameters | loss={loss.item():.4f}")


if __name__ == "__main__":
    import sys
    if "--check" in sys.argv:
        sanity_check()
    else:
        main()