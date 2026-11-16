"""Source-only shared reliability for four-channel packed RAW fusion."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from .noise import NoiseModel


EPSILON = 1.0e-12


def require_packed(tensor: torch.Tensor, name: str) -> None:
    if tensor.ndim != 4 or tensor.shape[1] != 4:
        raise ValueError("{} must have packed RAW shape [B, 4, H, W]".format(name))


def reflected_mean(value: torch.Tensor, window_size: int) -> torch.Tensor:
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")
    radius = window_size // 2
    if radius == 0:
        return value.float()
    padded = F.pad(value.float(), (radius, radius, radius, radius), mode="reflect")
    return F.avg_pool2d(padded, window_size, stride=1)


def local_gradient_power(image: torch.Tensor, window_size: int) -> torch.Tensor:
    """Sobel power averaged locally in packed coordinates."""
    require_packed(image, "image")
    image = image.float()
    channels = image.shape[1]
    kernel_x = image.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    kernel_y = kernel_x.t()
    gradient_x = F.conv2d(image, kernel_x[None, None].expand(channels, -1, -1, -1), padding=1, groups=channels)
    gradient_y = F.conv2d(image, kernel_y[None, None].expand(channels, -1, -1, -1), padding=1, groups=channels)
    return reflected_mean(gradient_x.square() + gradient_y.square(), window_size)


def native_saturation(native: torch.Tensor, start: float, end: float, window_size: int) -> torch.Tensor:
    require_packed(native, "native")
    if not 0.0 <= start < end <= 1.0:
        raise ValueError("Require 0 <= saturation start < end <= 1")
    fraction = ((native.float() - start) / (end - start)).clamp(0.0, 1.0)
    return reflected_mean(fraction, window_size)


def _guided_filter(guide: torch.Tensor, source: torch.Tensor, radius: int, epsilon: float) -> torch.Tensor:
    """Differentiable guided filter; callers use it with source-only tensors."""
    if guide.ndim != 4 or source.ndim != 4 or guide.shape != source.shape or guide.shape[1] != 1:
        raise ValueError("guided filter expects matching [B, 1, H, W] tensors")
    if radius < 0 or epsilon <= 0.0:
        raise ValueError("guided filter radius/epsilon must be non-negative/positive")
    size = 2 * radius + 1
    mean_guide = reflected_mean(guide, size)
    mean_source = reflected_mean(source, size)
    covariance = reflected_mean(guide * source, size) - mean_guide * mean_source
    variance = reflected_mean(guide * guide, size) - mean_guide.square()
    slope = covariance / (variance + epsilon)
    intercept = mean_source - slope * mean_guide
    return reflected_mean(slope, size) * guide + reflected_mean(intercept, size)


def _max_filter(mask: torch.Tensor, size: int) -> torch.Tensor:
    radius = size // 2
    return F.max_pool2d(F.pad(mask, (radius, radius, radius, radius), mode="reflect"), size, stride=1)


def _erode(mask: torch.Tensor, size: int) -> torch.Tensor:
    return 1.0 - _max_filter(1.0 - mask, size)


def _open_then_close(mask: torch.Tensor, open_size: int, close_size: int) -> torch.Tensor:
    opened = _max_filter(_erode(mask, open_size), open_size)
    return _erode(_max_filter(opened, close_size), close_size)


class RelativeInnovationSpatialReliability(torch.nn.Module):
    """Relative SNR, source-exclusive detail, and spatially coherent Q maps.

    Four CFA channels form source scores before the low-temperature sigmoid,
    yielding one shared Q map for all output channels.  This module has no
    learnable parameter and is evaluated in ``torch.no_grad`` by design.
    """

    def __init__(
        self,
        noise_model: NoiseModel,
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
    ) -> None:
        super().__init__()
        if gradient_window < 1 or gradient_window % 2 == 0:
            raise ValueError("gradient_window must be a positive odd integer")
        if noise_floor <= 0.0 or innovation_weight < 0.0 or saturation_penalty < 0.0:
            raise ValueError("noise floor must be positive; score weights must be non-negative")
        if selection_temperature <= 0.0 or guided_radius < 0 or guided_epsilon <= 0.0:
            raise ValueError("selection/guided-filter parameters are invalid")
        if not 0.5 < strong_threshold < 1.0 or not 0.5 < restore_threshold < 1.0:
            raise ValueError("strong and restore thresholds must be in (0.5, 1)")
        if open_size < 1 or close_size < 1 or open_size % 2 == 0 or close_size % 2 == 0:
            raise ValueError("morphology sizes must be positive odd integers")
        self.register_buffer("under_a", torch.tensor(noise_model.under_a, dtype=torch.float32).view(1, 4, 1, 1))
        self.register_buffer("under_b", torch.tensor(noise_model.under_b, dtype=torch.float32).view(1, 4, 1, 1))
        self.register_buffer("over_a", torch.tensor(noise_model.over_a, dtype=torch.float32).view(1, 4, 1, 1))
        self.register_buffer("over_b", torch.tensor(noise_model.over_b, dtype=torch.float32).view(1, 4, 1, 1))
        self.gradient_window = int(gradient_window)
        self.noise_floor = float(noise_floor)
        self.saturation_start = float(saturation_start)
        self.saturation_threshold = float(saturation_threshold)
        self.innovation_weight = float(innovation_weight)
        self.saturation_penalty = float(saturation_penalty)
        self.selection_temperature = float(selection_temperature)
        self.guided_radius = int(guided_radius)
        self.guided_epsilon = float(guided_epsilon)
        self.strong_threshold = float(strong_threshold)
        self.open_size = int(open_size)
        self.close_size = int(close_size)
        self.restore_threshold = float(restore_threshold)

    def native_variance(self, native: torch.Tensor, exposure: str) -> torch.Tensor:
        require_packed(native, "native")
        if exposure == "under":
            slope, intercept = self.under_a, self.under_b
        elif exposure == "over":
            slope, intercept = self.over_a, self.over_b
        else:
            raise ValueError("exposure must be under or over")
        return (slope * native.float().clamp_min(0.0) + intercept).clamp_min(self.noise_floor**2)

    @torch.no_grad()
    def forward(
        self,
        under_native: torch.Tensor,
        over_native: torch.Tensor,
        over_to_under_ratio: torch.Tensor,
        under_reference: torch.Tensor | None = None,
        over_reference: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        require_packed(under_native, "under_native")
        require_packed(over_native, "over_native")
        if under_native.shape != over_native.shape:
            raise ValueError("Under and over native tensors must match")
        ratio = over_to_under_ratio.float().reshape(-1, 1, 1, 1)
        if ratio.shape[0] != under_native.shape[0] or torch.any(ratio <= 0.0):
            raise ValueError("over_to_under_ratio must provide one positive value per item")
        under_scale = torch.sqrt(ratio)
        over_scale = 1.0 / under_scale
        under_reference = under_native.float() * under_scale if under_reference is None else under_reference.float()
        over_reference = over_native.float() * over_scale if over_reference is None else over_reference.float()
        require_packed(under_reference, "under_reference")
        require_packed(over_reference, "over_reference")

        under_variance = reflected_mean(self.native_variance(under_native, "under") * ratio, self.gradient_window)
        over_variance = reflected_mean(self.native_variance(over_native, "over") / ratio, self.gradient_window)
        under_gradient = local_gradient_power(under_reference, self.gradient_window)
        over_gradient = local_gradient_power(over_reference, self.gradient_window)
        under_noise_power = 24.0 * under_variance
        over_noise_power = 24.0 * over_variance
        under_snr = under_gradient / under_noise_power.clamp_min(self.noise_floor**2)
        over_snr = over_gradient / over_noise_power.clamp_min(self.noise_floor**2)
        under_detail = (under_gradient - under_noise_power).clamp_min(0.0)
        over_detail = (over_gradient - over_noise_power).clamp_min(0.0)
        detail_total = (under_detail + over_detail).clamp_min(EPSILON)
        under_innovation = (under_detail - over_detail).clamp_min(0.0) / detail_total
        over_innovation = (over_detail - under_detail).clamp_min(0.0) / detail_total
        under_saturation = native_saturation(under_native, self.saturation_start, self.saturation_threshold, self.gradient_window)
        over_saturation = native_saturation(over_native, self.saturation_start, self.saturation_threshold, self.gradient_window)

        under_score = (
            torch.log1p(under_snr).mean(dim=1, keepdim=True)
            + self.innovation_weight * under_innovation.mean(dim=1, keepdim=True)
            - self.saturation_penalty * under_saturation.mean(dim=1, keepdim=True)
        )
        over_score = (
            torch.log1p(over_snr).mean(dim=1, keepdim=True)
            + self.innovation_weight * over_innovation.mean(dim=1, keepdim=True)
            - self.saturation_penalty * over_saturation.mean(dim=1, keepdim=True)
        )
        score_difference_raw = over_score - under_score
        over_weight_raw = torch.sigmoid(score_difference_raw / self.selection_temperature)

        guide = 0.5 * (under_reference.mean(dim=1, keepdim=True) + over_reference.mean(dim=1, keepdim=True))
        guide_scale = torch.quantile(guide.flatten(start_dim=1), 0.995, dim=1).reshape(-1, 1, 1, 1).clamp_min(EPSILON)
        guide = (guide / guide_scale).clamp(0.0, 1.0)
        score_difference_smooth = _guided_filter(guide, score_difference_raw, self.guided_radius, self.guided_epsilon)
        over_weight_smooth = torch.sigmoid(score_difference_smooth / self.selection_temperature)
        under_weight_smooth = 1.0 - over_weight_smooth
        strong_over = _open_then_close((over_weight_smooth > self.strong_threshold).float(), self.open_size, self.close_size)
        strong_under = _open_then_close((under_weight_smooth > self.strong_threshold).float(), self.open_size, self.close_size)
        margin = self.selection_temperature * math.log(self.restore_threshold / (1.0 - self.restore_threshold))
        score_difference = torch.where(strong_over > 0.5, score_difference_smooth.clamp_min(margin), score_difference_smooth)
        score_difference = torch.where(strong_under > 0.5, score_difference.clamp_max(-margin), score_difference)
        over_weight = torch.sigmoid(score_difference / self.selection_temperature)
        under_weight = 1.0 - over_weight
        return {
            "under_weight": under_weight,
            "over_weight": over_weight,
            "under_weight_raw": 1.0 - over_weight_raw,
            "over_weight_raw": over_weight_raw,
            "under_weight_smooth": under_weight_smooth,
            "over_weight_smooth": over_weight_smooth,
            "strong_under_mask": strong_under,
            "strong_over_mask": strong_over,
            "score_difference_raw": score_difference_raw,
            "score_difference_smooth": score_difference_smooth,
            "score_difference": score_difference,
            "under_snr": under_snr,
            "over_snr": over_snr,
            "under_innovation": under_innovation,
            "over_innovation": over_innovation,
            "under_saturation": under_saturation,
            "over_saturation": over_saturation,
            "under_gradient_power": under_gradient,
            "over_gradient_power": over_gradient,
            "under_variance_reference": under_variance,
            "over_variance_reference": over_variance,
        }
