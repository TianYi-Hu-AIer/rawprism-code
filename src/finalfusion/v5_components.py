"""RAWPrism v5 components.

The three novel pieces that make v5 a "reliability-field" architecture rather
than a black-box fusion network:

1. :func:`physical_reliability_priors` -- closed-form, *physical* cues computed
   from the linear RAW (SNR / saturation / focus / spectral saliency), NOT from
   network latents.  This is the information RGB-domain methods have destroyed.
2. :class:`ReliabilityFieldTransformer` -- a task-conditioned module that turns
   physics (query) + semantic context (key/value) into two per-pixel, per-source
   reliability maps.  These maps are the paper's explainable visualisation.
3. :class:`PhysicsAnchoredFusion` -- ``u = w1*z1 + w2*z2 + g(w1*z1, w2*z2)`` with
   a zero-initialised residual ``g``: at init the model is exactly the physically
   optimal reliability-weighted radiance average, and learning only refines it.
4. :class:`DifferentiableRawRenderer` -- fused linear RAW -> sRGB through a
   differentiable tone-mapping head, fixing the "output is too dark" root cause.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def _sobel_power(x: torch.Tensor) -> torch.Tensor:
    """Per-channel Sobel gradient power, returned as [B, C, H, W]."""
    kx = x.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    ky = kx.t()
    c = x.shape[1]
    gx = F.conv2d(x, kx[None, None].expand(c, -1, -1, -1), padding=1, groups=c)
    gy = F.conv2d(x, ky[None, None].expand(c, -1, -1, -1), padding=1, groups=c)
    return gx.square() + gy.square()


def physical_reliability_priors(
    source1_raw: torch.Tensor,
    source2_raw: torch.Tensor,
    task: str,
    noise: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Closed-form physical reliability cues for one source pair.

    Args:
        source1_raw, source2_raw: packed linear RAW4 tensors ``[B, 4, H, W]``.
        task: one of ``"mef"``, ``"mff"``, ``"ivif"``.
        noise: optional dict with per-source Poisson-Gaussian coefficients
            ``{"a1", "b1", "a2", "b2"}`` each ``[1, 4, 1, 1]`` (MEF only).

    Returns a ``[B, 8, H, W]`` tensor:
        [g1, s1, e1, x1, g2, s2, e2, x2] where
        g = gradient power (focus/texture), s = saturation/validity,
        e = local energy, x = task-specific physics (MEF: SNR proxy; else 0).
    """
    if source1_raw.ndim != 4 or source1_raw.shape[1] != 4:
        raise ValueError("source1_raw must be packed RAW4 [B,4,H,W]")
    if source1_raw.shape != source2_raw.shape:
        raise ValueError("source1_raw and source2_raw must match")

    def _per_source(raw: torch.Tensor, a: torch.Tensor | None, b: torch.Tensor | None):
        grad = _sobel_power(raw).mean(dim=1, keepdim=True)
        grad = grad / (grad.mean(dim=(2, 3), keepdim=True) + 1e-6)
        sat = ((raw < 0.02) | (raw > 0.98)).float().mean(dim=1, keepdim=True)
        energy = raw.mean(dim=1, keepdim=True)
        if task == "mef":
            if a is not None and b is not None:
                variance = (a * raw.clamp_min(0.0) + b).clamp_min(1e-8)
            else:
                variance = (raw + 1e-3).clamp_min(1e-8)
            snr = raw / variance.sqrt()
            physics = snr.mean(dim=1, keepdim=True)
        elif task == "mff":
            physics = grad
        elif task == "ivif":
            physics = energy
        else:
            physics = torch.zeros_like(energy)
        return grad, sat, energy, physics

    a1 = noise.get("a1") if noise else None
    b1 = noise.get("b1") if noise else None
    a2 = noise.get("a2") if noise else None
    b2 = noise.get("b2") if noise else None
    g1, s1, e1, x1 = _per_source(source1_raw, a1, b1)
    g2, s2, e2, x2 = _per_source(source2_raw, a2, b2)
    return torch.cat((g1, s1, e1, x1, g2, s2, e2, x2), dim=1)


class FiLM(nn.Module):
    """Feature-wise linear modulation from a global condition vector."""

    def __init__(self, cond_dim: int, channels: int) -> None:
        super().__init__()
        self.scale = nn.Linear(cond_dim, channels)
        self.shift = nn.Linear(cond_dim, channels)

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        scale = self.scale(condition)
        shift = self.shift(condition)
        scale = scale.view(*scale.shape, 1, 1) + 1.0
        shift = shift.view(*shift.shape, 1, 1)
        return value * scale + shift


class ReliabilityFieldTransformer(nn.Module):
    """Task-conditioned reliability-field estimator.

    Physics (query) decides *where* each source is trustworthy; semantic context
    (key/value) corrects for "this dark region is a semantically important
    target".  The output is two per-pixel reliability maps ``w1, w2``.
    """

    def __init__(
        self,
        prior_channels: int = 8,
        latent_channels: int = 6,
        cond_dim: int = 32,
        hidden: int = 32,
        use_semantic: bool = False,
    ) -> None:
        super().__init__()
        self.use_semantic = bool(use_semantic)
        self.phys_proj = nn.Conv2d(prior_channels, hidden, 3, padding=1)
        ctx_in = 2 * latent_channels
        self.ctx_proj = nn.Conv2d(ctx_in, hidden, 3, padding=1)
        self.film = FiLM(cond_dim, hidden)
        self.fuse = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.to_logits = nn.Conv2d(hidden, 2, 1)
        nn.init.zeros_(self.to_logits.weight)
        nn.init.zeros_(self.to_logits.bias)

    def forward(
        self,
        priors: torch.Tensor,
        latent1: torch.Tensor,
        latent2: torch.Tensor,
        condition: torch.Tensor,
        semantic1: torch.Tensor | None = None,
        semantic2: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        physics = F.gelu(self.phys_proj(priors))
        context = F.gelu(self.ctx_proj(torch.cat((latent1, latent2), dim=1)))
        value = F.gelu(self.film(self.fuse(physics + context), condition))
        if self.use_semantic and semantic1 is not None and semantic2 is not None:
            sem = F.interpolate(semantic1 + semantic2, size=value.shape[-2:], mode="bilinear", align_corners=False)
            value = value + F.gelu(self.film(sem, condition))
        logits = self.to_logits(value)
        weights = torch.softmax(logits, dim=1)
        return weights[:, 0:1], weights[:, 1:2]


class PhysicsAnchoredFusion(nn.Module):
    """``u = w1*z1 + w2*z2 + g(w1*z1, w2*z2)`` with a zero-init residual."""

    def __init__(self, latent_channels: int = 6) -> None:
        super().__init__()
        self.residual = nn.Conv2d(2 * latent_channels, latent_channels, 1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(
        self,
        latent1: torch.Tensor,
        latent2: torch.Tensor,
        weight1: torch.Tensor,
        weight2: torch.Tensor,
    ) -> torch.Tensor:
        base = weight1 * latent1 + weight2 * latent2
        return base + self.residual(torch.cat((weight1 * latent1, weight2 * latent2), dim=1))


class DifferentiableRawRenderer(nn.Module):
    """Fused linear RAW4 -> sRGB through a differentiable tone-mapping head.

    The physical part (demosaic-averaging the green diagonals, then a Reinhard
    tone map) is fixed; a small learnable per-channel gain/bias lets training
    refine the display while staying anchored to a correct tone curve.
    """

    def __init__(self, learnable_tone: bool = True) -> None:
        super().__init__()
        self.learnable_tone = bool(learnable_tone)
        self.gain = nn.Parameter(torch.ones(1, 3, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, raw4: torch.Tensor) -> torch.Tensor:
        if raw4.ndim != 4 or raw4.shape[1] != 4:
            raise ValueError("renderer expects packed RAW4 [B,4,H,W]")
        r, gh, gv, b = raw4.unbind(dim=1)
        g = 0.5 * (gh + gv)
        rgb = torch.stack((r, g, b), dim=1).clamp(0.0, 1.0)
        if self.learnable_tone:
            rgb = (rgb * self.gain + self.bias).clamp(0.0, 1.0)
        return rgb / (1.0 + rgb)


__all__ = [
    "physical_reliability_priors",
    "FiLM",
    "ReliabilityFieldTransformer",
    "PhysicsAnchoredFusion",
    "DifferentiableRawRenderer",
]
