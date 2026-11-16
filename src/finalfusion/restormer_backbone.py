"""Restormer-style multi-scale backbone for RAW-domain unified fusion.

Design rationale
----------------
The earlier small DenseNet backbone only models local neighbourhoods.  For a
CVPR-competitive "large model" we need (i) global, long-range context, (ii)
RAW/CFA detail preservation, and (iii) an explicit task-conditioning channel so
that a single set of weights serves IVIF, MEF and MFF.

We therefore adopt the efficient *transposed attention* formulation of
Restormer (Zamir et al., CVPR 2022):

* ``MDTA`` computes attention along the *channel* dimension instead of the
  spatial dimension, so its cost is quadratic in ``C/heads`` rather than in
  ``H*W``.  This is the key to running long-range attention on 256x256 RAW
  crops within a 24GB GPU.
* ``GDFN`` performs gated depth-wise feature transformation for controlled
  spatial mixing.
* A three-level encoder/decoder with skip connections keeps CFA-level detail
  (edges, noise texture, saturation) while the bottleneck exchanges global
  context.
* Task conditioning is injected with *adaptive LayerNorm* (AdaLN): a learned
  scale/shift is produced from the task embedding and applied before every
  attention block.  This preserves the "one model, three tasks" selling point
  without the parameter-hungry FiLM used by the small baseline.

The module is domain-agnostic: the same backbone accepts a 12-channel RAW
latent or a 12-channel RGB latent, which is what enables the controlled
RAW-vs-RGB same-backbone comparison.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for [B, C, H, W] feature maps."""

    def __init__(self, dim: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value.permute(0, 2, 3, 1)
        value = self.norm(value)
        return value.permute(0, 3, 1, 2)


class AdaLN(nn.Module):
    """Adaptive LayerNorm modulation from a global task-condition vector."""

    def __init__(self, dim: int, cond_dim: int) -> None:
        super().__init__()
        self.modulation = nn.Linear(cond_dim, 2 * dim)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        scale, shift = self.modulation(condition).chunk(2, dim=-1)
        scale = scale.view(1, -1, 1, 1) + 1.0
        shift = shift.view(1, -1, 1, 1)
        return value * scale + shift


class MDTA(nn.Module):
    """Multi-Dconv-head Transposed Attention.

    Attention is evaluated over channel heads rather than spatial positions,
    giving long-range feature interaction at ``O((C/heads)^2 * HW)`` instead of
    ``O((HW)^2)``.
    """

    def __init__(self, dim: int, heads: int = 8, cond_dim: int = 0, bias: bool = False) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("MDTA dim must be divisible by heads")
        self.heads = int(heads)
        self.dim = int(dim)
        self.head_dim = dim // heads
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_dw = nn.Conv2d(dim * 3, dim * 3, 3, 1, 1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)
        self.ada = AdaLN(dim, cond_dim) if cond_dim > 0 else None

    def forward(self, value: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        batch, _, height, width = value.shape
        if self.ada is not None and condition is not None:
            value = self.ada(value, condition)
        qkv = self.qkv_dw(self.qkv(value))
        qkv = qkv.reshape(batch, 3, self.heads, self.head_dim, height * width)
        query, key, value_tokens = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)
        attention = torch.matmul(query, key.transpose(-2, -1)) * self.temperature
        attention = attention.softmax(dim=-1)
        out = torch.matmul(attention, value_tokens)
        out = out.reshape(batch, self.dim, height, width)
        return self.project_out(out)


class GDFN(nn.Module):
    """Gated depth-wise feed-forward network."""

    def __init__(self, dim: int, expansion_factor: float = 2.66, bias: bool = False) -> None:
        super().__init__()
        hidden = int(dim * expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden * 2, 1, bias=bias)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, 3, 1, 1, groups=hidden * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.project_in(value)
        first, second = self.dwconv(value).chunk(2, dim=1)
        return self.project_out(F.gelu(first) * second)


class RestormerBlock(nn.Module):
    """One transposed-attention transformer block with optional AdaLN."""

    def __init__(self, dim: int, heads: int = 8, cond_dim: int = 0,
                 expansion_factor: float = 2.66, bias: bool = False) -> None:
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attention = MDTA(dim, heads, cond_dim, bias)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = GDFN(dim, expansion_factor, bias)

    def forward(self, value: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        value = value + self.attention(self.norm1(value), condition)
        value = value + self.ffn(self.norm2(value))
        return value


class RestormerUNet(nn.Module):
    """Three-level transformer U-Net for unified RAW fusion.

    Accepts a canonical 12-channel latent (two 6-channel source latents) and
    returns a 6-channel shared latent consumed by the task head.  A global task
    condition is injected via AdaLN in every block.
    """

    def __init__(
        self,
        in_channels: int = 12,
        out_channels: int = 6,
        dims: tuple[int, int, int] = (64, 128, 256),
        blocks: tuple[int, int, int] = (2, 2, 2),
        heads: int = 8,
        cond_dim: int = 32,
        expansion_factor: float = 2.66,
    ) -> None:
        super().__init__()
        if len(dims) != 3 or len(blocks) != 3:
            raise ValueError("dims and blocks must have three entries")
        self.dims = tuple(dims)
        self.stem = nn.Conv2d(in_channels, dims[0], 3, 1, 1)

        self.enc1 = nn.ModuleList([RestormerBlock(dims[0], heads, cond_dim, expansion_factor) for _ in range(blocks[0])])
        self.down1 = nn.Conv2d(dims[0], dims[1], 4, 2, 1)
        self.enc2 = nn.ModuleList([RestormerBlock(dims[1], heads, cond_dim, expansion_factor) for _ in range(blocks[1])])
        self.down2 = nn.Conv2d(dims[1], dims[2], 4, 2, 1)

        self.bottleneck = nn.ModuleList([RestormerBlock(dims[2], heads, cond_dim, expansion_factor) for _ in range(blocks[2])])

        self.up2 = nn.ConvTranspose2d(dims[2], dims[1], 2, 2)
        self.dec2 = nn.ModuleList([RestormerBlock(dims[1], heads, cond_dim, expansion_factor) for _ in range(blocks[1])])
        self.up1 = nn.ConvTranspose2d(dims[1], dims[0], 2, 2)
        self.dec1 = nn.ModuleList([RestormerBlock(dims[0], heads, cond_dim, expansion_factor) for _ in range(blocks[0])])

        self.head = nn.Conv2d(dims[0], out_channels, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, value: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        x0 = self.stem(value)
        x1 = self.enc1[0](x0, condition)
        for block in self.enc1[1:]:
            x1 = block(x1, condition)
        x2 = self.down1(x1)
        x2 = self.enc2[0](x2, condition)
        for block in self.enc2[1:]:
            x2 = block(x2, condition)
        x3 = self.down2(x2)
        for block in self.bottleneck:
            x3 = block(x3, condition)

        x = self.up2(x3)
        x = x + x2
        x = self.dec2[0](x, condition)
        for block in self.dec2[1:]:
            x = block(x, condition)
        x = self.up1(x)
        x = x + x1
        x = self.dec1[0](x, condition)
        for block in self.dec1[1:]:
            x = block(x, condition)
        return self.head(x)


__all__ = [
    "LayerNorm2d",
    "AdaLN",
    "MDTA",
    "GDFN",
    "RestormerBlock",
    "RestormerUNet",
]
