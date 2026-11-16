"""Reference-domain fusion loss with spatial relative reliability and chroma projection."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from ...loss_utils import charbonnier, gradient_magnitude
from .chroma import SourceChromaProjection
from .noise import NoiseModel
from .reliability import RelativeInnovationSpatialReliability, require_packed


def _gaussian_kernel(size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coordinate = torch.arange(size, device=device, dtype=dtype) - size // 2
    grid_y, grid_x = torch.meshgrid(coordinate, coordinate, indexing="ij")
    kernel = torch.exp(-(grid_x.square() + grid_y.square()) / (2.0 * sigma * sigma))
    return (kernel / kernel.sum())[None, None]


def ssim_map(first: torch.Tensor, second: torch.Tensor, size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    if first.shape != second.shape or first.ndim != 4 or first.shape[1] != 1:
        raise ValueError("SSIM expects matching [B, 1, H, W] tensors")
    first, second = first.float(), second.float()
    kernel = _gaussian_kernel(size, sigma, first.device, first.dtype)
    radius = size // 2
    first_padded = F.pad(first, (radius, radius, radius, radius), mode="reflect")
    second_padded = F.pad(second, (radius, radius, radius, radius), mode="reflect")
    mean_first = F.conv2d(first_padded, kernel)
    mean_second = F.conv2d(second_padded, kernel)
    first_sq, second_sq = mean_first.square(), mean_second.square()
    product = mean_first * mean_second
    variance_first = (F.conv2d(first_padded.square(), kernel) - first_sq).clamp_min(0.0)
    variance_second = (F.conv2d(second_padded.square(), kernel) - second_sq).clamp_min(0.0)
    covariance = F.conv2d(first_padded * second_padded, kernel) - product
    c1, c2 = 0.01**2, 0.03**2
    return ((2.0 * product + c1) * (2.0 * covariance + c2)) / (
        (first_sq + second_sq + c1) * (variance_first + variance_second + c2)
    ).clamp_min(torch.finfo(first.dtype).eps)


def packed_ssim_map(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    require_packed(first, "first")
    require_packed(second, "second")
    if first.shape != second.shape:
        raise ValueError("Packed SSIM tensors must match")
    batch, channels, height, width = first.shape
    values = ssim_map(first.reshape(batch * channels, 1, height, width), second.reshape(batch * channels, 1, height, width))
    return values.reshape(batch, channels, values.shape[-2], values.shape[-1])


def spectral_angle_map(first: torch.Tensor, second: torch.Tensor, epsilon: float = 1.0e-8) -> torch.Tensor:
    require_packed(first, "first")
    require_packed(second, "second")
    if first.shape != second.shape:
        raise ValueError("SAM tensors must match")
    first, second = first.float(), second.float()
    first_norm = torch.linalg.vector_norm(first, dim=1, keepdim=True)
    second_norm = torch.linalg.vector_norm(second, dim=1, keepdim=True)
    cosine = (first * second).sum(dim=1, keepdim=True) / (first_norm * second_norm).clamp_min(epsilon)
    angle = torch.arccos(cosine.clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6))
    return torch.where((first_norm > epsilon) & (second_norm > epsilon), angle, torch.zeros_like(angle))


class RelativeInnovationSpatialChromaFusionLoss(torch.nn.Module):
    """Direct RAW4 supervision with explicit chroma-residual and range losses."""

    def __init__(
        self,
        noise_model: NoiseModel,
        pixel_weight: float = 2.0,
        structure_weight: float = 1.0,
        detail_weight: float = 0.5,
        chroma_residual_weight: float = 1.0,
        range_weight: float = 0.1,
        confidence_floor: float = 0.25,
        range_margin: float = 0.02,
        charbonnier_epsilon: float = 1.0e-3,
        gradient_window: int = 7,
        noise_floor: float = 1.0e-4,
        saturation_start: float = 0.98,
        saturation_threshold: float = 0.995,
        innovation_weight: float = 2.0,
        saturation_penalty: float = 6.0,
        selection_temperature: float = 0.25,
        guided_radius: int = 5,
        guided_epsilon: float = 1.0e-3,
        strong_threshold: float = 0.8,
        open_size: int = 3,
        close_size: int = 5,
        restore_threshold: float = 0.95,
        chroma_black_valid_start: float = 0.001,
        chroma_black_valid_end: float = 0.012,
        chroma_guided_radius: int = 3,
        chroma_guided_epsilon: float = 2.0e-3,
        chroma_agreement_gate: bool = False,
        chroma_agreement_tolerance: float = 0.5,
        loss_context: int = 16,
    ) -> None:
        super().__init__()
        if min(pixel_weight, structure_weight, detail_weight, chroma_residual_weight, range_weight) < 0.0:
            raise ValueError("Loss weights must be non-negative")
        if not 0.0 <= confidence_floor <= 1.0 or range_margin < 0.0 or charbonnier_epsilon <= 0.0:
            raise ValueError("MEF confidence/range/Charbonnier options are invalid")
        if loss_context < 0:
            raise ValueError("loss_context must be non-negative")
        self.pixel_weight = float(pixel_weight)
        self.structure_weight = float(structure_weight)
        self.detail_weight = float(detail_weight)
        self.chroma_residual_weight = float(chroma_residual_weight)
        self.range_weight = float(range_weight)
        self.confidence_floor = float(confidence_floor)
        self.range_margin = float(range_margin)
        self.charbonnier_epsilon = float(charbonnier_epsilon)
        self.loss_context = int(loss_context)
        self.reliability = RelativeInnovationSpatialReliability(
            noise_model,
            gradient_window=gradient_window,
            noise_floor=noise_floor,
            saturation_start=saturation_start,
            saturation_threshold=saturation_threshold,
            innovation_weight=innovation_weight,
            saturation_penalty=saturation_penalty,
            selection_temperature=selection_temperature,
            guided_radius=guided_radius,
            guided_epsilon=guided_epsilon,
            strong_threshold=strong_threshold,
            open_size=open_size,
            close_size=close_size,
            restore_threshold=restore_threshold,
        )
        self.chroma = SourceChromaProjection(
            black_valid_start=chroma_black_valid_start,
            black_valid_end=chroma_black_valid_end,
            saturation_start=saturation_start,
            saturation_end=saturation_threshold,
            guided_radius=chroma_guided_radius,
            guided_epsilon=chroma_guided_epsilon,
            agreement_gate=chroma_agreement_gate,
            agreement_tolerance=chroma_agreement_tolerance,
        )

    def source_reliability(
        self,
        under_native: torch.Tensor,
        over_native: torch.Tensor,
        over_to_under_ratio: torch.Tensor,
        under_reference: torch.Tensor | None = None,
        over_reference: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return self.reliability(under_native, over_native, over_to_under_ratio, under_reference, over_reference)

    def finalize_fused(
        self,
        fused_network: torch.Tensor,
        under_reference: torch.Tensor,
        over_reference: torch.Tensor,
        under_native: torch.Tensor,
        over_native: torch.Tensor,
        over_to_under_ratio: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        require_packed(fused_network, "fused_network")
        require_packed(under_reference, "under_reference")
        require_packed(over_reference, "over_reference")
        components = self.source_reliability(
            under_native, over_native, over_to_under_ratio, under_reference, over_reference
        )
        fused_final, chroma = self.chroma(
            fused_network, under_reference, over_reference, under_native, over_native
        )
        components.update({"chroma_" + key: value for key, value in chroma.items()})
        return fused_final, components

    def _central(self, value: torch.Tensor) -> torch.Tensor:
        if self.loss_context == 0:
            return value
        margin = self.loss_context
        if value.shape[-2] <= 2 * margin + 10 or value.shape[-1] <= 2 * margin + 10:
            raise ValueError("Input patch is too small for configured loss_context")
        return value[..., margin:-margin, margin:-margin]

    @staticmethod
    def _chroma_residuals(value: torch.Tensor) -> torch.Tensor:
        """Return canonical packed-RAW [R-G, B-G] residuals."""
        green = 0.5 * (value[:, 1:2] + value[:, 3:4])
        return torch.cat((value[:, 0:1] - green, value[:, 2:3] - green), dim=1)

    def forward(
        self,
        fused_network: torch.Tensor,
        under_reference: torch.Tensor,
        over_reference: torch.Tensor,
        under_native: torch.Tensor,
        over_native: torch.Tensor,
        over_to_under_ratio: torch.Tensor,
        fused_preclamp: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        require_packed(fused_network, "fused_network")
        components = self.source_reliability(
            under_native, over_native, over_to_under_ratio, under_reference, over_reference
        )
        under_target = under_reference.float().clamp(0.0, 1.0)
        over_target = over_reference.float().clamp(0.0, 1.0)
        target = components["under_weight"] * under_target + components["over_weight"] * over_target
        fused = self._central(fused_network.float())
        target = self._central(target)
        under_weight = self._central(components["under_weight"])
        over_weight = self._central(components["over_weight"])
        confidence = (under_weight - over_weight).abs()
        pixel_importance = self.confidence_floor + (1.0 - self.confidence_floor) * confidence
        pixel = (pixel_importance * charbonnier(fused - target, self.charbonnier_epsilon)).mean()
        structure = (
            under_weight * (1.0 - packed_ssim_map(fused, self._central(under_target)))
            + over_weight * (1.0 - packed_ssim_map(fused, self._central(over_target)))
        ).mean()
        fused_detail = gradient_magnitude(fused)
        detail_target = (
            under_weight * gradient_magnitude(self._central(under_target))
            + over_weight * gradient_magnitude(self._central(over_target))
        )
        detail = charbonnier(fused_detail - detail_target, self.charbonnier_epsilon).mean()
        chroma_residual = (
            self._chroma_residuals(fused) - self._chroma_residuals(target)
        )
        chroma_residual = charbonnier(chroma_residual, self.charbonnier_epsilon).mean()
        if fused_preclamp is None:
            range_loss = fused.new_zeros(())
        else:
            require_packed(fused_preclamp, "fused_preclamp")
            preclamp = self._central(fused_preclamp.float())
            range_loss = (
                F.relu(-preclamp - self.range_margin)
                + F.relu(preclamp - 1.0 - self.range_margin)
            ).mean()
        loss = (
            self.pixel_weight * pixel
            + self.structure_weight * structure
            + self.detail_weight * detail
            + self.chroma_residual_weight * chroma_residual
            + self.range_weight * range_loss
        )
        stats = {
            "loss": loss.detach(),
            "pixel": pixel.detach(),
            "structure": structure.detach(),
            "detail": detail.detach(),
            "chroma_residual": chroma_residual.detach(),
            "range": range_loss.detach(),
            "confidence": confidence.mean().detach(),
            "weight_under": under_weight.mean().detach(),
            "weight_over": over_weight.mean().detach(),
            "weight_under_raw": self._central(components["under_weight_raw"]).mean().detach(),
            "weight_over_raw": self._central(components["over_weight_raw"]).mean().detach(),
            "innovation_under": self._central(components["under_innovation"]).mean().detach(),
            "innovation_over": self._central(components["over_innovation"]).mean().detach(),
            "snr_under": self._central(components["under_snr"]).mean().detach(),
            "snr_over": self._central(components["over_snr"]).mean().detach(),
            "saturation_under": self._central(components["under_saturation"]).mean().detach(),
            "saturation_over": self._central(components["over_saturation"]).mean().detach(),
            "strong_under_fraction": self._central(components["strong_under_mask"]).mean().detach(),
            "strong_over_fraction": self._central(components["strong_over_mask"]).mean().detach(),
            "target_mean": target.mean().detach(),
            "network_mean": fused.mean().detach(),
        }
        return loss, stats


# Kept as an import-compatible name for scripts that refer to the old class.
GeometricMeanSNRFourChannelFusionLoss = RelativeInnovationSpatialChromaFusionLoss
