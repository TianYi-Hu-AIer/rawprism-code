"""Focus-aware RAW4 loss with normalized activity weights and five complementary terms."""

from __future__ import annotations

import torch
from torch import nn

from ...loss_utils import canonical_chroma, charbonnier, gaussian_kernel, gradient_magnitude, local_mean, reflected_filter, ssim_map


class Raw4GradientEnergyWeightedFusionLoss(nn.Module):
    """Focus-weighted pixel, structure, gradient, sharpness, and chroma objective."""

    def __init__(
        self,
        gradient_window: int = 11,
        ssim_window: int = 11,
        pixel_weight: float = 2.0,
        structure_weight: float = 1.0,
        gradient_weight: float = 1.0,
        sharpness_weight: float = 0.5,
        chroma_weight: float = 0.5,
        ssim_sigma: float = 1.5,
        focus_epsilon: float = 1.0e-4,
        charbonnier_epsilon: float = 1.0e-3,
    ) -> None:
        super().__init__()
        if gradient_window < 3 or gradient_window % 2 == 0 or ssim_window < 3 or ssim_window % 2 == 0:
            raise ValueError("gradient and SSIM windows must be odd and at least 3")
        if min(pixel_weight, structure_weight, gradient_weight, sharpness_weight, chroma_weight) < 0.0:
            raise ValueError("MFF loss weights must be non-negative")
        if focus_epsilon <= 0.0 or charbonnier_epsilon <= 0.0:
            raise ValueError("MFF loss epsilons must be positive")
        self.gradient_window = int(gradient_window)
        self.ssim_window = int(ssim_window)
        self.pixel_weight = float(pixel_weight)
        self.structure_weight = float(structure_weight)
        self.gradient_weight = float(gradient_weight)
        self.sharpness_weight = float(sharpness_weight)
        self.chroma_weight = float(chroma_weight)
        self.ssim_sigma = float(ssim_sigma)
        self.focus_epsilon = float(focus_epsilon)
        self.charbonnier_epsilon = float(charbonnier_epsilon)

    @staticmethod
    def _gray(raw4: torch.Tensor) -> torch.Tensor:
        return raw4.mean(dim=1, keepdim=True)

    def _sharpness(self, image: torch.Tensor) -> torch.Tensor:
        blurred = reflected_filter(image, gaussian_kernel(5, 1.0, image))
        return (image - blurred).abs()

    def forward(self, first_raw4: torch.Tensor, second_raw4: torch.Tensor, fused_raw4: torch.Tensor):
        if first_raw4.shape != second_raw4.shape or first_raw4.shape != fused_raw4.shape:
            raise ValueError("MFF loss expects matching [B,4,H,W] tensors")
        if first_raw4.ndim != 4 or first_raw4.shape[1] != 4:
            raise ValueError("MFF loss expects [B,4,H,W] tensors")
        if min(first_raw4.shape[-2:]) < max(self.gradient_window, self.ssim_window):
            raise ValueError("RAW4 patch is smaller than the configured loss window")

        first, second, fused = first_raw4.float(), second_raw4.float(), fused_raw4.float()
        first_gray, second_gray, fused_gray = self._gray(first), self._gray(second), self._gray(fused)
        with torch.no_grad():
            first_gradient = gradient_magnitude(first_gray)
            second_gradient = gradient_magnitude(second_gray)
            first_energy = local_mean(first_gradient.square(), self.gradient_window)
            second_energy = local_mean(second_gradient.square(), self.gradient_window)
            denominator = first_energy + second_energy + 2.0 * self.focus_epsilon
            weight_first = (first_energy + self.focus_epsilon) / denominator
            weight_second = 1.0 - weight_first
            gradient_target = weight_first * first_gradient + weight_second * second_gradient
            sharpness_target = weight_first * self._sharpness(first_gray) + weight_second * self._sharpness(second_gray)
            chroma_target = weight_first * canonical_chroma(first) + weight_second * canonical_chroma(second)

        pixel = (
            weight_first * charbonnier(fused - first, self.charbonnier_epsilon)
            + weight_second * charbonnier(fused - second, self.charbonnier_epsilon)
        ).mean()
        structure = (
            weight_first * (1.0 - ssim_map(fused, first, self.ssim_window, self.ssim_sigma))
            + weight_second * (1.0 - ssim_map(fused, second, self.ssim_window, self.ssim_sigma))
        ).mean()
        gradient = charbonnier(gradient_magnitude(fused_gray) - gradient_target, self.charbonnier_epsilon).mean()
        sharpness = charbonnier(self._sharpness(fused_gray) - sharpness_target, self.charbonnier_epsilon).mean()
        chroma = charbonnier(canonical_chroma(fused) - chroma_target, self.charbonnier_epsilon).mean()
        total = (
            self.pixel_weight * pixel
            + self.structure_weight * structure
            + self.gradient_weight * gradient
            + self.sharpness_weight * sharpness
            + self.chroma_weight * chroma
        )
        return {
            "total_loss": total,
            "pixel_loss": pixel,
            "ssim_loss": structure,
            "gradient_loss": gradient,
            "sharpness_loss": sharpness,
            "chroma_loss": chroma,
            "weight1": weight_first,
            "weight2": weight_second,
        }
