"""MEF loss adapter for the unified training interface."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .support.losses import RelativeInnovationSpatialChromaFusionLoss


class MefTaskLoss(nn.Module):
    def __init__(self, noise_model: Any, **options: Any) -> None:
        super().__init__()
        self.criterion = RelativeInnovationSpatialChromaFusionLoss(noise_model, **options)

    def forward(self, output: dict[str, Any], source1: torch.Tensor, source2: torch.Tensor,
                metadata: dict[str, Any]):
        del metadata
        context = output["context"]
        return self.criterion(
            output["fused_raw4"], context["under_reference"], context["over_reference"],
            source1, source2, context["exposure_ratio"], fused_preclamp=output["fused_raw_preclamp"]
        )
