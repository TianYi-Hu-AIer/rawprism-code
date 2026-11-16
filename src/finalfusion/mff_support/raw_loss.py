"""Exact RawLab MFF local Sobel-weighted SSIM + MSE objective in PyTorch."""

import torch
from torch import nn
from torch.nn import functional as functional


EPSILON = 1e-8


def _reflect_filter(image, kernel):
    radius = kernel.shape[-1] // 2
    return functional.conv2d(functional.pad(image, (radius, radius, radius, radius), mode="reflect"), kernel)


class Raw4GradientEnergyWeightedFusionLoss(nn.Module):
    """Unchanged RawLab MFF loss, applied after latent6 synthesis to RAW4."""

    def __init__(self, gradient_window=11, ssim_window=11, mse_weight=20.0, ssim_sigma=1.5):
        super().__init__()
        if gradient_window < 3 or gradient_window % 2 != 1 or ssim_window < 3 or ssim_window % 2 != 1:
            raise ValueError("gradient and SSIM windows must be odd and at least 3")
        self.gradient_window = int(gradient_window)
        self.ssim_window = int(ssim_window)
        self.mse_weight = float(mse_weight)
        self.ssim_sigma = float(ssim_sigma)
        self.register_buffer("sobel_x", torch.tensor(((-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0))).view(1, 1, 3, 3))
        self.register_buffer("sobel_y", torch.tensor(((-1.0, -2.0, -1.0), (0.0, 0.0, 0.0), (1.0, 2.0, 1.0))).view(1, 1, 3, 3))

    def _gaussian_window(self, device, dtype):
        coordinates = torch.arange(self.ssim_window, device=device, dtype=dtype) - self.ssim_window // 2
        x, y = torch.meshgrid(coordinates, coordinates, indexing="ij")
        window = torch.exp(-(x.square() + y.square()) / (2.0 * self.ssim_sigma * self.ssim_sigma))
        return (window / window.sum()).view(1, 1, self.ssim_window, self.ssim_window)

    def _ssim_map(self, first, second):
        window = self._gaussian_window(first.device, first.dtype)
        mean_first, mean_second = functional.conv2d(first, window), functional.conv2d(second, window)
        variance_first = functional.conv2d(first.square(), window) - mean_first.square()
        variance_second = functional.conv2d(second.square(), window) - mean_second.square()
        covariance = functional.conv2d(first * second, window) - mean_first * mean_second
        numerator = (2.0 * mean_first * mean_second + 0.01 ** 2) * (2.0 * covariance + 0.03 ** 2)
        denominator = (mean_first.square() + mean_second.square() + 0.01 ** 2) * (variance_first + variance_second + 0.03 ** 2)
        return numerator / (denominator + EPSILON)

    def _local_gradient_energy(self, image):
        energy = _reflect_filter(image, self.sobel_x).square() + _reflect_filter(image, self.sobel_y).square()
        average = torch.ones((1, 1, self.gradient_window, self.gradient_window), device=image.device, dtype=image.dtype)
        average /= float(self.gradient_window * self.gradient_window)
        return _reflect_filter(energy, average)

    def forward(self, first_raw4, second_raw4, fused_raw4):
        if first_raw4.shape != second_raw4.shape or first_raw4.shape != fused_raw4.shape:
            raise ValueError("MFF loss expects matching [B,4,H,W] tensors")
        if first_raw4.ndim != 4 or first_raw4.shape[1] != 4:
            raise ValueError("MFF loss expects [B,4,H,W] tensors")
        if min(first_raw4.shape[-2:]) < max(self.gradient_window, self.ssim_window):
            raise ValueError("RAW4 patch is smaller than the configured loss window")
        batch, _, height, width = first_raw4.shape
        first = first_raw4.reshape(batch * 4, 1, height, width)
        second = second_raw4.reshape(batch * 4, 1, height, width)
        fused = fused_raw4.reshape(batch * 4, 1, height, width)
        energy_first = self._local_gradient_energy(first)
        energy_second = self._local_gradient_energy(second)
        weight_first = (energy_first / (energy_first + energy_second + EPSILON)).detach()
        weight_second = (energy_second / (energy_first + energy_second + EPSILON)).detach()
        mse = (weight_first * (fused - first).square() + weight_second * (fused - second).square()).mean()
        radius = self.ssim_window // 2
        ssim_map = self._ssim_map(first, fused)
        cropped_first = weight_first[:, :, radius:-radius, radius:-radius]
        cropped_second = weight_second[:, :, radius:-radius, radius:-radius]
        ssim = (cropped_first * (1.0 - ssim_map) + cropped_second * (1.0 - self._ssim_map(second, fused))).mean()
        return {
            "total_loss": ssim + self.mse_weight * mse,
            "ssim_loss": ssim,
            "mse_loss": mse,
            "weight1": weight_first,
            "weight2": weight_second,
        }
