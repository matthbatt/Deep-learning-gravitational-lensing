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
        augment:  bool = False,
    ):
        assert images.shape[0] == labels.shape[0]
        self.images  = torch.from_numpy(images.astype(np.float32))
        self.labels  = torch.from_numpy(labels.astype(np.float32))
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]

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