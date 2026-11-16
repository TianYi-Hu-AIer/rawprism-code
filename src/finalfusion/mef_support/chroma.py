"""Brightness-preserving source chromaticity projection in packed RAW space."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from .reliability import EPSILON, reflected_mean, require_packed


def _smoothstep(value: torch.Tensor, start: float, end: float) -> torch.Tensor:
    fraction = ((value - start) / (end - start)).clamp(0.0, 1.0)
    return fraction.square() * (3.0 - 2.0 * fraction)


def _guided_filter_channelwise(guide: torch.Tensor, source: torch.Tensor, radius: int, epsilon: float) -> torch.Tensor:
    """Guided filter four independent chromaticity channels with one guide."""
    size = 2 * radius + 1
    guide4 = guide.expand(-1, 4, -1, -1)
    mean_guide = reflected_mean(guide4, size)
    mean_source = reflected_mean(source, size)
    covariance = reflected_mean(guide4 * source, size) - mean_guide * mean_source
    variance = reflected_mean(guide4.square(), size) - mean_guide.square()
    slope = covariance / (variance + epsilon)
    intercept = mean_source - slope * mean_guide
    return reflected_mean(slope, size) * guide4 + reflected_mean(intercept, size)


class SourceChromaProjection(torch.nn.Module):
    """Project fused packed RAW onto a source-derived four-channel chromaticity.

    The source chromaticity is built in the geometric-mean reference domain.
    Only valid, non-saturated native measurements contribute.  The projection
    preserves ``mean_c(fused)`` exactly before any later sensor-range clipping.
    """

    def __init__(
        self,
        black_valid_start: float = 0.001,
        black_valid_end: float = 0.012,
        saturation_start: float = 0.98,
        saturation_end: float = 0.995,
        guided_radius: int = 3,
        guided_epsilon: float = 2.0e-3,
        agreement_gate: bool = False,
        agreement_tolerance: float = 0.5,
    ) -> None:
        super().__init__()
        if not 0.0 <= black_valid_start < black_valid_end <= 1.0:
            raise ValueError("black-valid thresholds must satisfy 0 <= start < end <= 1")
        if not 0.0 <= saturation_start < saturation_end <= 1.0:
            raise ValueError("saturation thresholds must satisfy 0 <= start < end <= 1")
        if guided_radius < 0 or guided_epsilon <= 0.0 or agreement_tolerance <= 0.0:
            raise ValueError("guided/agreement parameters are invalid")
        self.black_valid_start = float(black_valid_start)
        self.black_valid_end = float(black_valid_end)
        self.saturation_start = float(saturation_start)
        self.saturation_end = float(saturation_end)
        self.guided_radius = int(guided_radius)
        self.guided_epsilon = float(guided_epsilon)
        self.agreement_gate = bool(agreement_gate)
        self.agreement_tolerance = float(agreement_tolerance)

    @staticmethod
    def chromaticity(value: torch.Tensor) -> torch.Tensor:
        return value / value.mean(dim=1, keepdim=True).clamp_min(EPSILON)

    def source_validity(self, native: torch.Tensor) -> torch.Tensor:
        require_packed(native, "native")
        luminance = native.float().mean(dim=1, keepdim=True)
        peak = native.float().amax(dim=1, keepdim=True)
        return _smoothstep(luminance, self.black_valid_start, self.black_valid_end) * (
            1.0 - _smoothstep(peak, self.saturation_start, self.saturation_end)
        )

    @torch.no_grad()
    def target_chromaticity(
        self,
        fused_reference: torch.Tensor,
        under_reference: torch.Tensor,
        over_reference: torch.Tensor,
        under_native: torch.Tensor,
        over_native: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        for name, value in (
            ("fused_reference", fused_reference), ("under_reference", under_reference),
            ("over_reference", over_reference), ("under_native", under_native), ("over_native", over_native),
        ):
            require_packed(value, name)
        under_chroma = self.chromaticity(under_reference.float())
        over_chroma = self.chromaticity(over_reference.float())
        fused_chroma = self.chromaticity(fused_reference.float())
        under_validity = self.source_validity(under_native)
        over_validity = self.source_validity(over_native)
        total_validity = under_validity + over_validity
        target_log = (
            under_validity * torch.log(under_chroma.clamp_min(EPSILON))
            + over_validity * torch.log(over_chroma.clamp_min(EPSILON))
        ) / total_validity.clamp_min(EPSILON)
        target_log = torch.where(total_validity > EPSILON, target_log, torch.log(fused_chroma.clamp_min(EPSILON)))
        guide = 0.5 * (under_reference.float().mean(dim=1, keepdim=True) + over_reference.float().mean(dim=1, keepdim=True))
        guide_scale = torch.quantile(guide.flatten(start_dim=1), 0.995, dim=1).reshape(-1, 1, 1, 1).clamp_min(EPSILON)
        guide = (guide / guide_scale).clamp(0.0, 1.0)
        target = torch.exp(_guided_filter_channelwise(guide, target_log, self.guided_radius, self.guided_epsilon))
        target = target / target.mean(dim=1, keepdim=True).clamp_min(EPSILON)
        source_difference = torch.abs(torch.log(under_chroma.clamp_min(EPSILON)) - torch.log(over_chroma.clamp_min(EPSILON))).mean(dim=1, keepdim=True)
        agreement = torch.exp(-source_difference / self.agreement_tolerance)
        alpha = agreement if self.agreement_gate else torch.ones_like(agreement)
        alpha = alpha * (total_validity > EPSILON).float()
        selected = (1.0 - alpha) * fused_chroma + alpha * target
        selected = selected / selected.mean(dim=1, keepdim=True).clamp_min(EPSILON)
        return {
            "target_chromaticity": selected,
            "source_target_chromaticity": target,
            "under_validity": under_validity,
            "over_validity": over_validity,
            "combined_validity": total_validity,
            "source_chromaticity_difference": source_difference,
            "agreement": agreement,
            "projection_alpha": alpha,
        }

    def forward(
        self,
        fused_reference: torch.Tensor,
        under_reference: torch.Tensor,
        over_reference: torch.Tensor,
        under_native: torch.Tensor,
        over_native: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        components = self.target_chromaticity(
            fused_reference, under_reference, over_reference, under_native, over_native
        )
        source_projected = fused_reference.float().mean(dim=1, keepdim=True) * components["target_chromaticity"]
        # With no valid source chromaticity, retain both the fused value and
        # its identity gradient rather than reconstructing from a detached ratio.
        alpha = components["projection_alpha"]
        projected = alpha * source_projected + (1.0 - alpha) * fused_reference.float()
        components["projection_mean_absolute_change"] = torch.abs(projected - fused_reference.float()).mean(
            dim=1, keepdim=True
        )
        return projected, components
