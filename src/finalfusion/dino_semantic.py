"""Frozen DINOv3 semantic alignment for IVIF.

The fused output should preserve the *semantic* content of both sources
(visible texture and thermal saliency), not just low-level structure.  We align
both the global ``[CLS]`` token and the *spatial patch tokens* of the fused sRGB
with those of each source, with DINOv3 frozen so gradients flow only into the
fused image.  The patch tokens capture local semantics and therefore also help
downstream segmentation (where the global CLS token alone was insufficient).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class DinoV3SemanticLoss(nn.Module):
    """Global semantic alignment loss using a frozen DINOv3 ViT-S/16."""

    def __init__(
        self,
        repo_dir: str = "/root/autodl-tmp/rawprism/baselines/UniFusion/dinov3",
        ckpt: str = "/root/autodl-tmp/rawprism/baselines/UniFusion/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
        size: int = 224,
        weight_visible: float = 1.0,
        weight_ir: float = 0.5,
    ) -> None:
        super().__init__()
        self.size = int(size)
        self.weight_visible = float(weight_visible)
        self.weight_ir = float(weight_ir)
        model = torch.hub.load(repo_dir, "dinov3_vits16", source="local", weights=ckpt)
        for param in model.parameters():
            param.requires_grad = False
        self.backbone = model.eval()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def _feat(self, rgb: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.interpolate(rgb, size=(self.size, self.size), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        out = self.backbone.forward_features(x)
        return out["x_norm_clstoken"], out["x_norm_patchtokens"]

    def forward(
        self,
        fused_srgb: torch.Tensor,
        visible_srgb: torch.Tensor,
        ir_rgb: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        with torch.no_grad():
            cls_visible, patch_visible = self._feat(visible_srgb)
            cls_ir, patch_ir = self._feat(ir_rgb)
        cls_fused, patch_fused = self._feat(fused_srgb)
        cls_fused = torch.nn.functional.normalize(cls_fused, dim=-1)
        cls_visible = torch.nn.functional.normalize(cls_visible, dim=-1)
        cls_ir = torch.nn.functional.normalize(cls_ir, dim=-1)
        patch_fused = torch.nn.functional.normalize(patch_fused, dim=-1)
        patch_visible = torch.nn.functional.normalize(patch_visible, dim=-1)
        patch_ir = torch.nn.functional.normalize(patch_ir, dim=-1)
        loss_visible = (
            torch.nn.functional.mse_loss(cls_fused, cls_visible)
            + torch.nn.functional.mse_loss(patch_fused, patch_visible)
        )
        loss_ir = (
            torch.nn.functional.mse_loss(cls_fused, cls_ir)
            + torch.nn.functional.mse_loss(patch_fused, patch_ir)
        )
        total = self.weight_visible * loss_visible + self.weight_ir * loss_ir
        return total, {
            "semantic_visible": loss_visible.detach(),
            "semantic_ir": loss_ir.detach(),
        }


__all__ = ["DinoV3SemanticLoss"]

