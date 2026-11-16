"""Shared differentiable primitives for task-specific fusion losses."""

from __future__ import annotations

import torch
from torch.nn import functional as F


EPSILON = 1.0e-8


def charbonnier(value: torch.Tensor, epsilon: float = 1.0e-3) -> torch.Tensor:
    if epsilon <= 0.0:
        raise ValueError("Charbonnier epsilon must be positive")
    return torch.sqrt(value.square() + epsilon * epsilon)


def gaussian_kernel(size: int, sigma: float, reference: torch.Tensor) -> torch.Tensor:
    if size < 3 or size % 2 == 0 or sigma <= 0.0:
        raise ValueError("Gaussian size must be odd and >= 3; sigma must be positive")
    coordinates = torch.arange(size, device=reference.device, dtype=reference.dtype) - size // 2
    grid_y, grid_x = torch.meshgrid(coordinates, coordinates, indexing="ij")
    kernel = torch.exp(-(grid_x.square() + grid_y.square()) / (2.0 * sigma * sigma))
    return (kernel / kernel.sum()).view(1, 1, size, size)


def reflected_filter(value: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    radius = kernel.shape[-1] // 2
    channels = value.shape[1]
    expanded = kernel.expand(channels, 1, -1, -1)
    return F.conv2d(F.pad(value, (radius, radius, radius, radius), mode="reflect"), expanded, groups=channels)


def local_mean(value: torch.Tensor, size: int) -> torch.Tensor:
    kernel = value.new_ones((1, 1, size, size)) / float(size * size)
    return reflected_filter(value, kernel)


def ssim_map(first: torch.Tensor, second: torch.Tensor, size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    if first.shape != second.shape or first.ndim != 4:
        raise ValueError("SSIM expects matching BCHW tensors")
    kernel = gaussian_kernel(size, sigma, first)
    mean_first = reflected_filter(first, kernel)
    mean_second = reflected_filter(second, kernel)
    variance_first = (reflected_filter(first.square(), kernel) - mean_first.square()).clamp_min(0.0)
    variance_second = (reflected_filter(second.square(), kernel) - mean_second.square()).clamp_min(0.0)
    covariance = reflected_filter(first * second, kernel) - mean_first * mean_second
    numerator = (2.0 * mean_first * mean_second + 0.01 ** 2) * (2.0 * covariance + 0.03 ** 2)
    denominator = (mean_first.square() + mean_second.square() + 0.01 ** 2) * (
        variance_first + variance_second + 0.03 ** 2
    )
    return numerator / denominator.clamp_min(torch.finfo(first.dtype).eps)


def sobel_gradients(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if value.ndim != 4:
        raise ValueError("Sobel gradients expect BCHW tensors")
    sobel_x = value.new_tensor(((-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0))).view(1, 1, 3, 3)
    sobel_y = value.new_tensor(((-1.0, -2.0, -1.0), (0.0, 0.0, 0.0), (1.0, 2.0, 1.0))).view(1, 1, 3, 3)
    return reflected_filter(value, sobel_x), reflected_filter(value, sobel_y)


def gradient_magnitude(value: torch.Tensor, epsilon: float = EPSILON) -> torch.Tensor:
    gradient_x, gradient_y = sobel_gradients(value)
    return torch.sqrt(gradient_x.square() + gradient_y.square() + epsilon)


def canonical_chroma(value: torch.Tensor) -> torch.Tensor:
    """Return [R-G, B-G] from canonical RAW4 [R,Gh,Gv,B]."""
    if value.ndim != 4 or value.shape[1] != 4:
        raise ValueError("Canonical chroma expects RAW4 [R,Gh,Gv,B]")
    green = 0.5 * (value[:, 1:2] + value[:, 2:3])
    return torch.cat((value[:, 0:1] - green, value[:, 3:4] - green), dim=1)
