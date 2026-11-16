"""Task loss dispatch; implementations live in the three task packages."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .ivif.loss import IvifTaskLoss
from .loss_utils import charbonnier, gradient_magnitude
from .mef.loss import MefTaskLoss
from .mff.loss import MffTaskLoss


class RenderingLossWrapper(nn.Module):
    """Adds a tone-mapped sRGB supervision on top of a task loss.

    The base task losses supervise the *linear* fused RAW.  That never tells the
    model that a linear value of ``0.2`` renders much darker than the sources and
    therefore looks wrong.  This wrapper supervises the rendered ``fused_srgb``
    (output by the v5 differentiable renderer) against a Reinhard-tone-mapped
    reference of the sources, which fixes the "output is too dark" root cause.
    """

    def __init__(self, base: nn.Module, weight: float = 1.0, epsilon: float = 1.0e-3) -> None:
        super().__init__()
        self.base = base
        self.weight = float(weight)
        self.epsilon = float(epsilon)

    def _rendering_loss(self, srgb: torch.Tensor, source1: torch.Tensor, source2: torch.Tensor):
        g1 = source1.mean(dim=1, keepdim=True)
        g2 = source2.mean(dim=1, keepdim=True)
        t1 = g1 / (1.0 + g1)
        t2 = g2 / (1.0 + g2)
        reference = 0.5 * (t1 + t2)
        gray = srgb.mean(dim=1, keepdim=True)
        brightness = charbonnier(gray - reference, self.epsilon).mean()
        gradient_source = torch.maximum(gradient_magnitude(t1), gradient_magnitude(t2))
        structure = charbonnier(gradient_magnitude(gray) - gradient_source, self.epsilon).mean()
        return brightness + structure, {
            "render_brightness": brightness.detach(),
            "render_structure": structure.detach(),
        }

    def forward(self, output: dict, source1: torch.Tensor, source2: torch.Tensor, metadata: dict):
        loss, stats = self.base(output, source1, source2, metadata)
        if "fused_srgb" in output and self.weight > 0.0:
            render_loss, render_stats = self._rendering_loss(output["fused_srgb"], source1, source2)
            loss = loss + self.weight * render_loss
            stats.update(render_stats)
        return loss, stats


class SemanticLossWrapper(nn.Module):
    """Adds a frozen-DINOv3 semantic alignment term on top of the IVIF loss."""

    def __init__(self, base: nn.Module, weight: float = 1.0) -> None:
        super().__init__()
        from .dino_semantic import DinoV3SemanticLoss

        self.base = base
        self.weight = float(weight)
        self.semantic = DinoV3SemanticLoss()

    def forward(self, output: dict, source1: torch.Tensor, source2: torch.Tensor, metadata: dict):
        loss, stats = self.base(output, source1, source2, metadata)
        if "visible_srgb" in output and "ir_rgb" in output and self.weight > 0.0:
            sem_loss, sem_stats = self.semantic(
                output["fused_srgb"], output["visible_srgb"], output["ir_rgb"]
            )
            loss = loss + self.weight * sem_loss
            stats.update(sem_stats)
        return loss, stats


def build_task_loss(task: str, config: dict[str, Any], device: torch.device) -> nn.Module:
    if task == "ivif":
        criterion: nn.Module = IvifTaskLoss(
            intensity_weight=config.get("intensity_weight", 2.0),
            structure_weight=config.get("structure_weight", 1.0),
            gradient_weight=config.get("gradient_weight", 1.0),
            chroma_weight=config.get("chroma_weight", 1.0),
            artifact_weight=config.get("artifact_weight", 0.1),
            activity_window=config.get("activity_window", 7),
            activity_temperature=config.get("activity_temperature", 1.0),
            flat_threshold=config.get("flat_threshold", 0.02),
            artifact_margin=config.get("artifact_margin", 0.01),
            charbonnier_epsilon=config.get("charbonnier_epsilon", 1.0e-3),
            target_mode=config.get("target_mode", "weighted"),
            visible_anchor_weight=config.get("visible_anchor_weight", 0.0),
            ir_anchor_weight=config.get("ir_anchor_weight", 0.0),
        )
    elif task == "mef":
        from .mef.support.noise import load_noise_model

        criterion = MefTaskLoss(
            load_noise_model(config["cache_root"]),
            pixel_weight=config.get("pixel_weight", 2.0),
            structure_weight=config.get("structure_weight", 1.0),
            detail_weight=config.get("detail_weight", 0.5),
            chroma_residual_weight=config.get("chroma_residual_weight", 1.0),
            range_weight=config.get("range_weight", 0.1),
            confidence_floor=config.get("confidence_floor", 0.25),
            range_margin=config.get("range_margin", 0.02),
            charbonnier_epsilon=config.get("charbonnier_epsilon", 1.0e-3),
            gradient_window=config.get("gradient_window", 7),
            noise_floor=config.get("noise_floor", 1.0e-4),
            saturation_start=config.get("saturation_start", 0.98),
            saturation_threshold=config.get("saturation_threshold", 0.995),
            innovation_weight=config.get("innovation_weight", 2.0),
            saturation_penalty=config.get("saturation_penalty", 6.0),
            selection_temperature=config.get("selection_temperature", 0.25),
            guided_radius=config.get("reliability_guided_radius", 5),
            guided_epsilon=config.get("reliability_guided_epsilon", 1.0e-3),
            strong_threshold=config.get("strong_selection_threshold", 0.8),
            open_size=config.get("morph_open_size", 3),
            close_size=config.get("morph_close_size", 5),
            restore_threshold=config.get("restore_selection_threshold", 0.95),
            chroma_black_valid_start=config.get("chroma_black_valid_start", 0.001),
            chroma_black_valid_end=config.get("chroma_black_valid_end", 0.012),
            chroma_guided_radius=config.get("chroma_guided_radius", 3),
            chroma_guided_epsilon=config.get("chroma_guided_epsilon", 2.0e-3),
            chroma_agreement_gate=config.get("chroma_agreement_gate", False),
            chroma_agreement_tolerance=config.get("chroma_agreement_tolerance", 0.5),
            loss_context=config.get("context_pad", 0),
        )
    elif task == "mff":
        criterion = MffTaskLoss(
            gradient_window=config.get("gradient_window", 11),
            ssim_window=config.get("ssim_window", 11),
            pixel_weight=config.get("pixel_weight", 2.0),
            structure_weight=config.get("structure_weight", 1.0),
            gradient_weight=config.get("gradient_weight", 1.0),
            sharpness_weight=config.get("sharpness_weight", 0.5),
            chroma_weight=config.get("chroma_weight", 0.5),
            ssim_sigma=config.get("ssim_sigma", 1.5),
            focus_epsilon=config.get("focus_epsilon", 1.0e-4),
            charbonnier_epsilon=config.get("charbonnier_epsilon", 1.0e-3),
        )
    else:
        raise ValueError("Unknown task {!r}".format(task))
    criterion = criterion.to(device)
    rendering_weight = float(config.get("rendering_weight", 0.0))
    if rendering_weight > 0.0:
        criterion = RenderingLossWrapper(criterion, weight=rendering_weight)
    semantic_weight = float(config.get("semantic_weight", 0.0))
    if semantic_weight > 0.0 and task == "ivif":
        criterion = SemanticLossWrapper(criterion, weight=semantic_weight)
    return criterion.to(device)


__all__ = [
    "IvifTaskLoss", "MefTaskLoss", "MffTaskLoss", "RenderingLossWrapper",
    "SemanticLossWrapper", "build_task_loss",
]
