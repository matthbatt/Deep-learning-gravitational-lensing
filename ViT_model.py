import torch
import torch.nn as nn
import math

# -------------------------
# Basic Transformer Blocks
# -------------------------

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim, heads=6):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads)
        q, k, v = qkv.unbind(dim=2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# -------------------------
# Vision Transformer
# -------------------------

class TinyViT(nn.Module):
    def __init__(self, img_size=140, patch=4, in_ch=4, dim=192, depth=6, heads=6, out_dim=5):
        super().__init__()

        assert img_size % patch == 0, "Image size must be divisible by patch size"
        self.grid = img_size // patch
        num_patches = self.grid * self.grid

        # Patch embedding: 4 channels → dim
        self.patch_embed = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)

        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))

        # Transformer encoder
        self.blocks = nn.Sequential(*[
            TransformerBlock(dim, heads) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)

        # Regression head
        self.head = nn.Linear(dim, out_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]

        # Patchify
        x = self.patch_embed(x)          # [B, dim, grid, grid]
        x = x.flatten(2).transpose(1, 2) # [B, N, dim]

        # Add class token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1)

        # Add positional embedding
        x = x + self.pos_embed

        # Transformer
        x = self.blocks(x)
        x = self.norm(x)

        # Use CLS token for regression
        cls_out = x[:, 0]
        return self.head(cls_out)



