"""Shared RAW/CFA components used by all FinalFusion tasks."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

HADAMARD = ((1.0, 1.0, 1.0, 1.0), (1.0, -1.0, 1.0, -1.0),
            (1.0, 1.0, -1.0, -1.0), (1.0, -1.0, -1.0, 1.0))


def soft_clamp_unit(value: torch.Tensor, beta: float = 50.0) -> torch.Tensor:
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("soft-clamp beta must be finite and positive")
    return F.softplus(value, beta=beta) - F.softplus(value - 1.0, beta=beta)


def require_raw4(first: torch.Tensor, second: torch.Tensor) -> None:
    if first.shape != second.shape or first.ndim != 4 or first.shape[1] != 4:
        raise ValueError("Sources must be matching RAW4 tensors [B,4,H,W]")


def _neighbor(value: torch.Tensor, vertical: int, horizontal: int) -> torch.Tensor:
    height, width = value.shape[-2:]
    padded = F.pad(value, (1, 1, 1, 1), mode="replicate")
    top, left = 1 + vertical, 1 + horizontal
    return padded[..., top:top + height, left:left + width]


def _selector(hidden: int, explicit_init: bool) -> nn.Sequential:
    module = nn.Sequential(nn.Conv2d(5, hidden, 3, padding=1), nn.GELU(), nn.Conv2d(hidden, 2, 1))
    if explicit_init:
        nn.init.trunc_normal_(module[0].weight, mean=0.0, std=0.05, a=-0.1, b=0.1)
        nn.init.zeros_(module[0].bias)
    nn.init.zeros_(module[-1].weight)
    nn.init.zeros_(module[-1].bias)
    return module


class DirectionalGreenPredictor(nn.Module):
    def __init__(self, shared_selector: bool, hidden: int = 16) -> None:
        super().__init__()
        self.shared_selector = bool(shared_selector)
        if self.shared_selector:
            self.shared = _selector(hidden, explicit_init=False)
        else:
            self.at_red = _selector(hidden, explicit_init=True)
            self.at_blue = _selector(hidden, explicit_init=True)

    @staticmethod
    def _select(selector: nn.Module, horizontal: torch.Tensor, vertical: torch.Tensor,
                horizontal_gradient: torch.Tensor, vertical_gradient: torch.Tensor):
        evidence = torch.cat((horizontal, vertical, horizontal_gradient, vertical_gradient,
                              (horizontal - vertical).abs()), dim=1)
        logits = -torch.cat((horizontal_gradient, vertical_gradient), dim=1) + selector(evidence)
        weights = torch.softmax(logits, dim=1)
        return weights[:, :1] * horizontal + weights[:, 1:] * vertical, weights

    def forward(self, green_h: torch.Tensor, green_v: torch.Tensor):
        red_selector = self.shared if self.shared_selector else self.at_red
        blue_selector = self.shared if self.shared_selector else self.at_blue
        gh_r, gv_d = _neighbor(green_h, 0, 1), _neighbor(green_v, 1, 0)
        green_at_red, red_weights = self._select(red_selector, 0.5 * (green_h + gh_r),
                                                  0.5 * (green_v + gv_d), (green_h - gh_r).abs(), (green_v - gv_d).abs())
        gv_l, gh_u = _neighbor(green_v, 0, -1), _neighbor(green_h, -1, 0)
        green_at_blue, blue_weights = self._select(blue_selector, 0.5 * (green_v + gv_l),
                                                    0.5 * (green_h + gh_u), (green_v - gv_l).abs(), (green_h - gh_u).abs())
        return green_at_red, green_at_blue, red_weights, blue_weights


class CfaAnalysis(nn.Module):
    def __init__(self, shared_selector: bool) -> None:
        super().__init__()
        self.register_buffer("hadamard", torch.tensor(HADAMARD, dtype=torch.float32) / 2.0)
        self.green_predictor = DirectionalGreenPredictor(shared_selector)

    def forward(self, raw4: torch.Tensor):
        if raw4.ndim != 4 or raw4.shape[1] != 4:
            raise ValueError("CFA analysis expects canonical RAW4 [B,4,H,W]")
        red, green_h, green_v, blue = raw4.unbind(dim=1)
        green_at_red, green_at_blue, red_weights, blue_weights = self.green_predictor(green_h.unsqueeze(1), green_v.unsqueeze(1))
        achromatic = torch.cat((green_at_red, green_h.unsqueeze(1), green_v.unsqueeze(1), green_at_blue), dim=1)
        structure = torch.einsum("ij,bjhw->bihw", self.hadamard, achromatic)
        chroma = torch.cat((red.unsqueeze(1) - green_at_red, blue.unsqueeze(1) - green_at_blue), dim=1)
        return torch.cat((structure, chroma), dim=1), {"green_weights_red": red_weights, "green_weights_blue": blue_weights}


class CfaSynthesis(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("hadamard", torch.tensor(HADAMARD, dtype=torch.float32) / 2.0)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 4 or latent.shape[1] != 6:
            raise ValueError("CFA synthesis expects latent6 [B,6,H,W]")
        achromatic = torch.einsum("ij,bjhw->bihw", self.hadamard, latent[:, :4])
        return torch.cat((achromatic[:, :1] + latent[:, 4:5], achromatic[:, 1:3], achromatic[:, 3:4] + latent[:, 5:6]), dim=1)


class ResidualProjection(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.projection(value)


class TaskInputAdapter(nn.Module):
    def __init__(self, shared_selector: bool) -> None:
        super().__init__()
        self.analysis = CfaAnalysis(shared_selector)
        self.projection = ResidualProjection(12)

    def prepare_sources(self, source1: torch.Tensor, source2: torch.Tensor, metadata: dict[str, Any]):
        return source1, source2, dict(metadata)

    def alter_latents(self, first: torch.Tensor, second: torch.Tensor):
        return first, second, {}

    def forward(self, source1: torch.Tensor, source2: torch.Tensor, metadata: dict[str, Any]):
        require_raw4(source1, source2)
        first_raw, second_raw, context = self.prepare_sources(source1, source2, metadata)
        first_latent, first_aux = self.analysis(first_raw)
        second_latent_raw, second_aux = self.analysis(second_raw)
        first_latent, second_latent, extra = self.alter_latents(first_latent, second_latent_raw)
        context.update({"source1_model_raw4": first_raw, "source2_model_raw4": second_raw,
                        "source1_latent": first_latent, "source2_latent": second_latent,
                        "source2_latent_raw": second_latent_raw,
                        "source1_green_weights_red": first_aux["green_weights_red"],
                        "source1_green_weights_blue": first_aux["green_weights_blue"],
                        "source2_green_weights_red": second_aux["green_weights_red"],
                        "source2_green_weights_blue": second_aux["green_weights_blue"]})
        context.update(extra)
        return self.projection(torch.cat((first_latent, second_latent), dim=1)), context


class ReflectConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activate: bool = True) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3)
        self.activate = bool(activate)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.conv(F.pad(value, (1, 1, 1, 1), mode="reflect"))
        return F.leaky_relu(value, negative_slope=0.2, inplace=True) if self.activate else value


class DenseLayer(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int) -> None:
        super().__init__()
        self.layer = ReflectConv(in_channels, growth_rate)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.layer(value), value), dim=1)


class SharedU2FusionBackbone(nn.Module):
    def __init__(self, width: int = 44) -> None:
        super().__init__()
        self.stem = ReflectConv(12, width)
        channels = width
        dense = []
        for _ in range(5):
            dense.append(DenseLayer(channels, width))
            channels += width
        self.dense = nn.Sequential(*dense)
        self.decoder = nn.Sequential(ReflectConv(channels, 128), ReflectConv(128, 64), ReflectConv(64, 32), ReflectConv(32, 6, activate=False))
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu")
                nn.init.zeros_(module.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4 or value.shape[1] != 12:
            raise ValueError("shared backbone expects [B,12,H,W]")
        return self.decoder(self.dense(self.stem(value)))


class TaskOutputHead(nn.Module):
    def __init__(self, mef_external_order: bool = False) -> None:
        super().__init__()
        self.mef_external_order = bool(mef_external_order)
        self.projection = ResidualProjection(6)
        self.synthesis = CfaSynthesis()

    def forward(self, shared_latent: torch.Tensor):
        fused_latent = self.projection(shared_latent)
        fused_preclamp = self.synthesis(fused_latent)
        if self.mef_external_order:
            fused_preclamp = fused_preclamp.index_select(1, torch.as_tensor((0, 1, 3, 2), device=fused_preclamp.device))
        return {"shared_latent": shared_latent, "fused_latent": fused_latent,
                "fused_raw_preclamp": fused_preclamp, "fused_raw4": soft_clamp_unit(fused_preclamp)}
