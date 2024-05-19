# code adapted from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit_for_small_dataset.py

from math import sqrt

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# b = batch size, c = channels, h = height, w = width, s = patch size, d = embedding dimension, p = number of patches
class Shortcut(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.identity = nn.Parameter(torch.eye(dim), requires_grad=False)

    def forward(self, x):
        return x @ self.identity.T


class ViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="mean",
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {"cls", "mean"}, "pool type must be either cls (cls token) or mean (mean pooling)"

        self.to_patch_embedding = SPT(dim=dim, patch_size=patch_size, channels=channels)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, img):
        """
        img: (b, c, h, w)
        """
        # (b, p, d)
        x = self.to_patch_embedding(img)
        b, p, _ = x.shape

        # (1, 1, d) -> (b, 1, d)
        cls_tokens = repeat(self.cls_token, "() 1 d -> b 1 d", b=b)

        # (b, p + 1, d)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (p + 1)]
        x = self.dropout(x)

        # (b, p+1, d)
        x = self.transformer(x)

        # (b, d)
        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        return self.mlp_head(x)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        LSA(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        Shortcut(dim),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                        Shortcut(dim),
                    ]
                )
            )

    def forward(self, x):
        for attn, shortcut1, ff, shortcut2 in self.layers:
            x = attn(x) + shortcut1(x)
            x = ff(x) + shortcut2(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class LSA(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.temperature = nn.Parameter(torch.log(torch.tensor(dim_head**-0.5)))

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        # (b, p, d)
        x = self.norm(x)

        # (b, p+1, (h*d))
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # (b, h, p+1, d)
        q, k, v = map(lambda t: rearrange(t, "b p (h d) -> b h p d", h=self.heads), (q, k, v))

        # (b, h, p+1, p+1)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.temperature.exp()

        # (p+1, p+1)
        mask = torch.eye(dots.shape[-1], device=dots.device, dtype=torch.bool)
        mask_value = -torch.finfo(dots.dtype).max  # - inf
        dots = dots.masked_fill(mask, mask_value)  # to avoid attending to the token itself

        # (b, h, p+1, p+1)
        attn = self.attend(dots)
        attn = self.dropout(attn)

        # (b, h, p+1, d)
        out = torch.matmul(attn, v)

        # (b, p, (h*d))
        out = rearrange(out, "b h p d -> b p (h d)")
        return self.to_out(out)


class SPT(nn.Module):
    def __init__(self, *, dim, patch_size, channels=3):
        super().__init__()
        patch_dim = patch_size * patch_size * 5 * channels

        self.to_patch_tokens = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
        )

    def forward(self, x):
        """
        x: (b, c, h, w)
        """
        shifts = ((1, -1, 0, 0), (-1, 1, 0, 0), (0, 0, 1, -1), (0, 0, -1, 1))
        shifted_x = list(map(lambda shift: F.pad(x, shift), shifts))
        x_with_shifts = torch.cat((x, *shifted_x), dim=1)
        return self.to_patch_tokens(x_with_shifts)
