"""Spatially adaptive IVIF objective with separated intensity and color roles."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ..loss_utils import canonical_chroma, charbonnier, gradient_magnitude, local_mean, ssim_map


class IvifTaskLoss(nn.Module):
    def __init__(
        self,
        intensity_weight: float = 2.0,
        structure_weight: float = 1.0,
        gradient_weight: float = 1.0,
        chroma_weight: float = 1.0,
        artifact_weight: float = 0.1,
        activity_window: int = 7,
        activity_temperature: float = 1.0,
        flat_threshold: float = 0.02,
        artifact_margin: float = 0.01,
        charbonnier_epsilon: float = 1.0e-3,
        target_mode: str = "weighted",
        visible_anchor_weight: float = 0.0,
        ir_anchor_weight: float = 0.0,
    ) -> None:
        super().__init__()
        weights = (intensity_weight, structure_weight, gradient_weight, chroma_weight, artifact_weight)
        if min(weights) < 0.0:
            raise ValueError("IVIF loss weights must be non-negative")
        if activity_window < 3 or activity_window % 2 == 0 or activity_temperature <= 0.0:
            raise ValueError("IVIF activity window/temperature is invalid")
        if target_mode not in ("weighted", "select", "soft_select"):
            raise ValueError("target_mode must be weighted/select/soft_select")
        self.intensity_weight = float(intensity_weight)
        self.structure_weight = float(structure_weight)
        self.gradient_weight = float(gradient_weight)
        self.chroma_weight = float(chroma_weight)
        self.artifact_weight = float(artifact_weight)
        self.activity_window = int(activity_window)
        self.activity_temperature = float(activity_temperature)
        self.flat_threshold = float(flat_threshold)
        self.artifact_margin = float(artifact_margin)
        self.charbonnier_epsilon = float(charbonnier_epsilon)
        self.target_mode = str(target_mode)
        self.visible_anchor_weight = float(visible_anchor_weight)
        self.ir_anchor_weight = float(ir_anchor_weight)

    @staticmethod
    def _gray(raw4: torch.Tensor) -> torch.Tensor:
        return raw4.mean(dim=1, keepdim=True)

    def forward(
        self,
        output: dict[str, Any],
        source1: torch.Tensor,
        source2: torch.Tensor,
        metadata: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del metadata
        fused = output["fused_raw4"].float()
        visible, infrared = source1.float(), source2.float()
        fused_gray, visible_gray, infrared_gray = self._gray(fused), self._gray(visible), self._gray(infrared)

        with torch.no_grad():
            visible_gradient = gradient_magnitude(visible_gray)
            infrared_gradient = gradient_magnitude(infrared_gray)
            visible_activity = local_mean(visible_gradient.square(), self.activity_window)
            infrared_activity = local_mean(infrared_gradient.square(), self.activity_window)
            logits = torch.cat((torch.log(visible_activity + 1.0e-6), torch.log(infrared_activity + 1.0e-6)), dim=1)
            modality_weights = torch.softmax(logits / self.activity_temperature, dim=1)
            if self.target_mode == "select":
                select_visible = (visible_activity >= infrared_activity).float()
                visible_weight, infrared_weight = select_visible, 1.0 - select_visible
                intensity_target = visible_weight * visible_gray + infrared_weight * infrared_gray
                gradient_target = torch.maximum(visible_gradient, infrared_gradient)
            elif self.target_mode == "soft_select":
                visible_weight, infrared_weight = modality_weights[:, :1], modality_weights[:, 1:]
                intensity_target = visible_weight * visible_gray + infrared_weight * infrared_gray
                gradient_target = torch.maximum(visible_gradient, infrared_gradient)
            else:
                visible_weight, infrared_weight = modality_weights[:, :1], modality_weights[:, 1:]
                intensity_target = visible_weight * visible_gray + infrared_weight * infrared_gray
                gradient_target = visible_weight * visible_gradient + infrared_weight * infrared_gradient
            maximum_source_gradient = torch.maximum(visible_gradient, infrared_gradient)
            flat_mask = (maximum_source_gradient < self.flat_threshold).float()

        fused_gradient = gradient_magnitude(fused_gray)
        intensity = charbonnier(fused_gray - intensity_target, self.charbonnier_epsilon).mean()
        structure = (
            visible_weight * (1.0 - ssim_map(fused_gray, visible_gray))
            + infrared_weight * (1.0 - ssim_map(fused_gray, infrared_gray))
        ).mean()
        gradient = charbonnier(fused_gradient - gradient_target, self.charbonnier_epsilon).mean()
        chroma = charbonnier(canonical_chroma(fused) - canonical_chroma(visible), self.charbonnier_epsilon).mean()
        visible_anchor = charbonnier((fused_gray - visible_gray) * visible_weight, self.charbonnier_epsilon).mean()
        ir_anchor = charbonnier((fused_gray - infrared_gray) * infrared_weight, self.charbonnier_epsilon).mean()
        excess_gradient = torch.relu(fused_gradient - maximum_source_gradient - self.artifact_margin)
        artifact = (flat_mask * excess_gradient).sum() / flat_mask.sum().clamp_min(1.0)

        total = (
            self.intensity_weight * intensity
            + self.structure_weight * structure
            + self.gradient_weight * gradient
            + self.chroma_weight * chroma
            + self.artifact_weight * artifact
            + self.visible_anchor_weight * visible_anchor
            + self.ir_anchor_weight * ir_anchor
        )
        return total, {
            "loss": total.detach(),
            "intensity": intensity.detach(),
            "structure": structure.detach(),
            "gradient": gradient.detach(),
            "chroma": chroma.detach(),
            "artifact": artifact.detach(),
            "visible_anchor": visible_anchor.detach(),
            "ir_anchor": ir_anchor.detach(),
            "weight_visible": visible_weight.mean().detach(),
            "weight_infrared": infrared_weight.mean().detach(),
            "flat_fraction": flat_mask.mean().detach(),
        }
