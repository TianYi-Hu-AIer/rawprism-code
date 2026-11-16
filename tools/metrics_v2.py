"""Validated fusion metrics (protocol v2) for RAWPrism.

Why this file exists
--------------------
The released evaluator (``finalfusion/metrics.py`` + ``scripts/extra_metrics.py``)
does not implement the definitions it names:

* Q_ABF never uses the fused gradient magnitude, so a constant image scores 0.9966.
* SCD is ``cos(F-A, A) + cos(F-B, B)`` instead of the authors' cross-difference
  Pearson form ``corr(F-B, A) + corr(F-A, B)``.
* VIFp averages per-scale ratios instead of pooling numerator/denominator
  information across scales (LIVE ``vifp_mscale.m``).

This module implements the published definitions, keeps the same gray/border
conventions as the rest of the codebase, and is validated against an independent
reference on synthetic arrays (see ``tools/local/validate_metrics_v2.py``).

Conventions
-----------
* Images are float, luminance in [0, 1] (RGB is converted with Rec.601 weights,
  the same as the rest of the released code).
* Metrics are computed at a single declared resolution; callers must guarantee
  that sources and fused share the same field of view.
* Degenerate inputs (constant images, zero gradients) return defined values
  instead of NaN, and the choice is documented per function.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, sobel

EPS = 1.0e-10

# Xydeas & Petrovic (2000) constants, as used by the reference implementations.
GAMMA_G, KAPPA_G, SIGMA_G = 0.9994, -15.0, 0.5
GAMMA_A, KAPPA_A, SIGMA_A = 0.9879, -22.0, 0.8


def to_gray(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 1:
        return image[..., 0]
    if image.ndim == 3 and image.shape[2] == 3:
        return 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    if image.ndim == 3 and image.shape[2] == 4:
        # RGBA: drop alpha, never average it into luminance.
        return 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    raise ValueError("expected [H,W], [H,W,1] or [H,W,3], got {}".format(image.shape))


# ---------------------------------------------------------------- Q^{AB/F}
def _strength_orientation(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sobel gradient magnitude and *undirected* edge orientation in [0, pi)."""
    gray = to_gray(image)
    gx = sobel(gray, axis=1) / 8.0
    gy = sobel(gray, axis=0) / 8.0
    return np.hypot(gx, gy), np.mod(np.arctan2(gy, gx), np.pi)


def _q_gradient(source_strength: np.ndarray, fused_strength: np.ndarray) -> np.ndarray:
    """Relative-strength term: uses BOTH source and fused magnitude."""
    largest = np.maximum(source_strength, fused_strength)
    smallest = np.minimum(source_strength, fused_strength)
    relative = np.divide(smallest, largest, out=np.ones_like(largest), where=largest > EPS)
    return GAMMA_G / (1.0 + np.exp(KAPPA_G * (relative - SIGMA_G)))


def _q_orientation(source_orientation: np.ndarray, fused_orientation: np.ndarray) -> np.ndarray:
    """Orientation-preservation term with correct pi-periodic wrap-around."""
    difference = np.abs(source_orientation - fused_orientation)
    difference = np.mod(difference, np.pi)
    distance = np.minimum(difference, np.pi - difference)  # in [0, pi/2]
    relative = 1.0 - distance / (np.pi / 2.0)
    return GAMMA_A / (1.0 + np.exp(KAPPA_A * (relative - SIGMA_A)))


def q_abf(source_a: np.ndarray, source_b: np.ndarray, fused: np.ndarray) -> float:
    """Xydeas-Petrovic Q^{AB/F} edge-preservation index (range [0, ~0.976] for identity).

    Identity input gives ~0.975, not 1.0: that is the published constant choice
    (Gamma_g, Gamma_a < 1), not a defect.  A structure-destroying output must
    score near 0.
    """
    strength_f, orientation_f = _strength_orientation(fused)
    numerator = 0.0
    denominator = EPS
    for source in (source_a, source_b):
        strength_s, orientation_s = _strength_orientation(source)
        q = _q_gradient(strength_s, strength_f) * _q_orientation(orientation_s, orientation_f)
        numerator += float(np.sum(q * strength_s))
        denominator += float(np.sum(strength_s))
    return numerator / denominator


# --------------------------------------------------------------------- SCD
def scd(source_a: np.ndarray, source_b: np.ndarray, fused: np.ndarray) -> float:
    """Sum of correlated differences (Aslantas & Bendes).

    ``corr(F-B, A) + corr(F-A, B)`` with Pearson correlation (mean removed).
    Correlations of a constant difference map are undefined; such a term
    contributes 0.0 and the reason is reported by the caller, never silently
    replaced by a cosine.
    """
    a, b, f = to_gray(source_a), to_gray(source_b), to_gray(fused)
    return _pearson(f - b, a) + _pearson(f - a, b)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean()
    y = y - y.mean()
    denominator = float(np.linalg.norm(x.ravel()) * np.linalg.norm(y.ravel()))
    if denominator <= EPS:
        return 0.0
    return float(np.sum(x * y) / denominator)


# ------------------------------------------------------------------- VIFp
def _gaussian_window(size: int, sigma: float) -> np.ndarray:
    axis = np.arange(-(size // 2), size // 2 + 1, dtype=np.float64)
    g = np.exp(-(axis**2) / (2.0 * sigma**2))
    g /= g.sum()
    return np.outer(g, g)


def _to_u8(image: np.ndarray) -> np.ndarray:
    return np.round(np.clip(to_gray(image), 0.0, 1.0) * 255.0)


def vifp(reference: np.ndarray, distorted: np.ndarray, scales: int = 4, sigma_nsq: float = 2.0) -> float:
    """Pixel-domain VIF, LIVE ``vifp_mscale.m`` semantics.

    The scale information is accumulated (``sum(num)/sum(den)``) before the
    ratio, exactly as in the authors' MATLAB code.  Values above 1 are allowed.
    """
    from scipy.signal import convolve2d

    ref = _to_u8(reference)
    dst = _to_u8(distorted)
    total_num = 0.0
    total_den = 0.0
    for scale in range(1, scales + 1):
        size = 2 ** (4 - scale + 1) + 1
        window = _gaussian_window(size, size / 5.0)
        if scale > 1:
            ref = convolve2d(ref, window, mode="valid")[::2, ::2]
            dst = convolve2d(dst, window, mode="valid")[::2, ::2]
        mu1 = convolve2d(ref, window, mode="valid")
        mu2 = convolve2d(dst, window, mode="valid")
        sigma1_sq = convolve2d(ref * ref, window, mode="valid") - mu1 * mu1
        sigma2_sq = convolve2d(dst * dst, window, mode="valid") - mu2 * mu2
        sigma12 = convolve2d(ref * dst, window, mode="valid") - mu1 * mu2

        sigma1_sq = np.maximum(sigma1_sq, 0.0)
        sigma2_sq = np.maximum(sigma2_sq, 0.0)
        g = sigma12 / (sigma1_sq + EPS)
        sv_sq = sigma2_sq - g * sigma12

        low1 = sigma1_sq < EPS
        g[low1] = 0.0
        sv_sq[low1] = sigma2_sq[low1]
        sigma1_sq[low1] = 0.0
        low2 = sigma2_sq < EPS
        g[low2] = 0.0
        sv_sq[low2] = 0.0
        negative = g < 0.0
        sv_sq[negative] = sigma2_sq[negative]
        g[negative] = 0.0
        sv_sq = np.maximum(sv_sq, EPS)

        total_num += float(np.sum(np.log10(1.0 + g * g * sigma1_sq / (sv_sq + sigma_nsq))))
        total_den += float(np.sum(np.log10(1.0 + sigma1_sq / sigma_nsq)))
    return total_num / max(total_den, EPS)


def vif_fusion(source_a: np.ndarray, source_b: np.ndarray, fused: np.ndarray) -> float:
    """Source-averaged VIF.  The averaging is our declared fusion choice."""
    return 0.5 * (vifp(source_a, fused) + vifp(source_b, fused))


# ------------------------------------------------------- EN / SD / MI (kept)
def entropy(image: np.ndarray) -> float:
    histogram, _ = np.histogram(np.clip(to_gray(image), 0.0, 1.0), bins=256, range=(0.0, 1.0))
    p = histogram / max(histogram.sum(), 1)
    p = p[p > 0.0]
    return float(-np.sum(p * np.log2(p)))


def std(image: np.ndarray) -> float:
    return float(to_gray(image).std())


def mutual_information(first: np.ndarray, second: np.ndarray) -> float:
    a = np.round(np.clip(to_gray(first), 0.0, 1.0) * 255).astype(np.uint8)
    b = np.round(np.clip(to_gray(second), 0.0, 1.0) * 255).astype(np.uint8)
    joint, _, _ = np.histogram2d(a.ravel(), b.ravel(), bins=256, range=[[0, 255], [0, 255]])
    joint = joint / joint.sum()
    pa = joint.sum(axis=1)
    pb = joint.sum(axis=0)
    ha = -np.sum(pa[pa > 0] * np.log2(pa[pa > 0]))
    hb = -np.sum(pb[pb > 0] * np.log2(pb[pb > 0]))
    hj = -np.sum(joint[joint > 0] * np.log2(joint[joint > 0]))
    return float(ha + hb - hj)


def mi_fusion(source_a: np.ndarray, source_b: np.ndarray, fused: np.ndarray) -> float:
    return mutual_information(source_a, fused) + mutual_information(source_b, fused)


# ------------------------------------------------------------- SSIM / PSNR
def ssim(reference: np.ndarray, fused: np.ndarray, sigma: float = 1.5, truncate: float = 4.0) -> float:
    """Gaussian-window SSIM with the same border convention on both images.

    ``truncate`` is explicit so this is not silently a different window than the
    11x11 valid SSIM used elsewhere.
    """
    a, b = to_gray(reference), to_gray(fused)
    mu1 = gaussian_filter(a, sigma, truncate=truncate)
    mu2 = gaussian_filter(b, sigma, truncate=truncate)
    sigma1_sq = gaussian_filter(a * a, sigma, truncate=truncate) - mu1 * mu1
    sigma2_sq = gaussian_filter(b * b, sigma, truncate=truncate) - mu2 * mu2
    sigma12 = gaussian_filter(a * b, sigma, truncate=truncate) - mu1 * mu2
    c1, c2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1 * mu1 + mu2 * mu2 + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return float(ssim_map.mean())


def psnr(reference: np.ndarray, fused: np.ndarray) -> float:
    mse = float(np.mean((to_gray(reference) - to_gray(fused)) ** 2))
    if mse <= EPS:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


METRIC_VERSION = "rawprism-metrics-v2"


def evaluate_pair(source_a: np.ndarray, source_b: np.ndarray, fused: np.ndarray) -> dict[str, float]:
    """No-reference suite on one common field of view.

    VIF is *not* included here: its four-scale local statistics cost ~40 s on a
    12 MP image, and the protocol computes VIF/SSIM/PSNR/CC on a declared
    1024 max-side grid (see tools/evaluate_v2.py::extras).
    """
    return {
        "EN": entropy(fused),
        "SD": std(fused),
        "Q_ABF": q_abf(source_a, source_b, fused),
        "MI": mi_fusion(source_a, source_b, fused),
        "SCD": scd(source_a, source_b, fused),
    }


__all__ = [
    "METRIC_VERSION", "to_gray", "q_abf", "scd", "vifp", "vif_fusion",
    "entropy", "std", "mi_fusion", "mutual_information", "ssim", "psnr",
    "evaluate_pair",
]
