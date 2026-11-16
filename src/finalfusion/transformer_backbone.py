"""A transformer U-Net backbone for RAW-domain unified fusion.

The network downsamples the canonical 12-channel latent with convolutions,
applies several global self-attention blocks at a quarter resolution, then
upsamples back with skip connections.  It is domain-agnostic: the same backbone
accepts RAW or RGB feature inputs, which enables the controlled RAW-vs-RGB
comparison.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(self.conv(value), negative_slope=0.2, inplace=True)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 8, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        value = self.norm1(value)
        value = self.attn(value, value, value, need_weights=False)[0]
        value = value + residual
        residual = value
        value = self.mlp(self.norm2(value)) + residual
        return value


class TransformerUNet(nn.Module):
    """Convolutional encoder/decoder with a transformer bottleneck."""

    def __init__(self, in_channels: int = 12, out_channels: int = 6,
                 base: int = 64, bottleneck_blocks: int = 4, heads: int = 8) -> None:
        super().__init__()
        self.stem = nn.Sequential(ConvBlock(in_channels, base), ConvBlock(base, base))
        self.down1 = ConvBlock(base, base * 2, stride=2)
        self.down2 = ConvBlock(base * 2, base * 2, stride=2)
        self.bottleneck = nn.ModuleList([TransformerBlock(base * 2, heads) for _ in range(bottleneck_blocks)])
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                 ConvBlock(base * 2, base))
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                 ConvBlock(base, base))
        self.head = nn.Conv2d(base, out_channels, 1)
        for module in self.modules():
            if isinstance(module, nn.Conv2d) and module is not self.head:
                nn.init.kaiming_normal_(module.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        x0 = self.stem(value)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        batch, channels, height, width = x2.shape
        tokens = x2.flatten(2).transpose(1, 2)
        for block in self.bottleneck:
            tokens = block(tokens)
        x2 = tokens.transpose(1, 2).reshape(batch, channels, height, width)
        value = self.up1(x2) + x1
        value = self.up2(value) + x0
        return self.head(value)


__all__ = ["ConvBlock", "TransformerBlock", "TransformerUNet"]
