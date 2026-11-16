"""MFF loss adapter for the unified training interface."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .support.raw_loss import Raw4GradientEnergyWeightedFusionLoss


class MffTaskLoss(nn.Module):
    def __init__(self, **options: Any) -> None:
        super().__init__()
        self.criterion = Raw4GradientEnergyWeightedFusionLoss(**options)

    def forward(self, output: dict[str, Any], source1: torch.Tensor, source2: torch.Tensor,
                metadata: dict[str, Any]):
        del metadata
        values = self.criterion(source1.float(), source2.float(), output["fused_raw4"].float())
        return values["total_loss"], {
            "loss": values["total_loss"].detach(),
            "pixel": values["pixel_loss"].detach(),
            "structure": values["ssim_loss"].detach(),
            "gradient": values["gradient_loss"].detach(),
            "sharpness": values["sharpness_loss"].detach(),
            "chroma": values["chroma_loss"].detach(),
            "weight1": values["weight1"].mean().detach(),
            "weight2": values["weight2"].mean().detach(),
            "weight_sum_error": (values["weight1"] + values["weight2"] - 1.0).abs().max().detach(),
        }
