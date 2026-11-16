"""Reference and no-reference fusion metrics used in RAWPrism evaluation.

All image inputs are single-channel or 3-channel float arrays in [0, 1].
Metrics are implemented in NumPy so they can run on CPU independently of the
training loop.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import sobel


EPSILON = 1.0e-10


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.float64)
    if image.ndim == 3 and image.shape[2] == 3:
        return (0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2])
    if image.ndim == 3 and image.shape[2] == 1:
        return image[..., 0].astype(np.float64)
    raise ValueError("Expected 2-D or 3-channel image, got {}".format(image.shape))


def psnr(reference: np.ndarray, fused: np.ndarray) -> float:
    reference, fused = to_gray(reference), to_gray(fused)
    mse = float(np.mean((reference - fused) ** 2))
    if mse <= EPSILON:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def ssim(reference: np.ndarray, fused: np.ndarray, window: int = 11, sigma: float = 1.5) -> float:
    from scipy.ndimage import gaussian_filter

    reference, fused = to_gray(reference), to_gray(fused)
    mu1 = gaussian_filter(reference, sigma)
    mu2 = gaussian_filter(fused, sigma)
    mu1_sq, mu2_sq = mu1 * mu1, mu2 * mu2
    mu12 = mu1 * mu2
    sigma1_sq = gaussian_filter(reference * reference, sigma) - mu1_sq
    sigma2_sq = gaussian_filter(fused * fused, sigma) - mu2_sq
    sigma12 = gaussian_filter(reference * fused, sigma) - mu12
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu12 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return float(ssim_map.mean())


def entropy(image: np.ndarray) -> float:
    gray = to_gray(image)
    histogram, _ = np.histogram(gray, bins=256, range=(0.0, 1.0))
    probabilities = histogram / (histogram.sum() + EPSILON)
    probabilities = probabilities[probabilities > 0.0]
    return float(-np.sum(probabilities * np.log2(probabilities + EPSILON)))


def std(image: np.ndarray) -> float:
    return float(to_gray(image).std())


def _edge_strength_and_orientation(image: np.ndarray):
    gray = to_gray(image)
    gx = sobel(gray, axis=1)
    gy = sobel(gray, axis=0)
    strength = np.sqrt(gx * gx + gy * gy)
    orientation = np.arctan2(gy, gx)
    return strength, orientation


def q_abf(source_a: np.ndarray, source_b: np.ndarray, fused: np.ndarray) -> float:
    """Xydeas-Petrovic edge-preservation metric Q^{AB/F}."""
    strength_a, orientation_a = _edge_strength_and_orientation(source_a)
    strength_b, orientation_b = _edge_strength_and_orientation(source_b)
    strength_f, orientation_f = _edge_strength_and_orientation(fused)
    total_a = strength_a.sum() + EPSILON
    total_b = strength_b.sum() + EPSILON
    af_diff = np.abs(orientation_a - orientation_f)
    af_angle = np.minimum(af_diff, np.pi - af_diff)
    g_af = 0.9994 / (1.0 + np.exp(15.0 * (af_angle - np.pi / 2.0)))
    a_af = 1.0 / (1.0 + np.exp(10.0 * (strength_a - 0.8)))
    q_af = np.sum(g_af * a_af * strength_a) / total_a
    bf_diff = np.abs(orientation_b - orientation_f)
    bf_angle = np.minimum(bf_diff, np.pi - bf_diff)
    g_bf = 0.9994 / (1.0 + np.exp(15.0 * (bf_angle - np.pi / 2.0)))
    a_bf = 1.0 / (1.0 + np.exp(10.0 * (strength_b - 0.8)))
    q_bf = np.sum(g_bf * a_bf * strength_b) / total_b
    return float((q_af * total_a + q_bf * total_b) / (total_a + total_b))


def mutual_information(first: np.ndarray, second: np.ndarray) -> float:
    first, second = to_gray(first), to_gray(second)
    first = np.round(first * 255).astype(np.uint8)
    second = np.round(second * 255).astype(np.uint8)
    joint, _, _ = np.histogram2d(first.ravel(), second.ravel(), bins=256, range=[[0, 255], [0, 255]])
    joint = joint / joint.sum()
    first_marginal = joint.sum(axis=1)
    second_marginal = joint.sum(axis=0)
    first_entropy = -np.sum(first_marginal * np.log2(first_marginal + EPSILON))
    second_entropy = -np.sum(second_marginal * np.log2(second_marginal + EPSILON))
    joint_entropy = -np.sum(joint * np.log2(joint + EPSILON))
    return float(first_entropy + second_entropy - joint_entropy)


def mi_fusion(source_a: np.ndarray, source_b: np.ndarray, fused: np.ndarray) -> float:
    return float(mutual_information(source_a, fused) + mutual_information(source_b, fused))


def scd(source_a: np.ndarray, source_b: np.ndarray, fused: np.ndarray) -> float:
    """Sum of correlated differences between fused and each source."""
    a, b, f = to_gray(source_a), to_gray(source_b), to_gray(fused)
    diff_a = f - a
    diff_b = f - b
    denom_a = np.linalg.norm(diff_a.ravel()) * np.linalg.norm(a.ravel()) + EPSILON
    denom_b = np.linalg.norm(diff_b.ravel()) * np.linalg.norm(b.ravel()) + EPSILON
    corr_a = float(np.sum(diff_a.ravel() * a.ravel()) / denom_a)
    corr_b = float(np.sum(diff_b.ravel() * b.ravel()) / denom_b)
    return corr_a + corr_b


def evaluate_pair(source_a: np.ndarray, source_b: np.ndarray, fused: np.ndarray,
                  reference: np.ndarray | None = None) -> dict[str, float]:
    result = {
        "EN": entropy(fused),
        "SD": std(fused),
        "Q_ABF": q_abf(source_a, source_b, fused),
        "MI": mi_fusion(source_a, source_b, fused),
        "SCD": scd(source_a, source_b, fused),
    }
    if reference is not None:
        result["PSNR"] = psnr(reference, fused)
        result["SSIM"] = ssim(reference, fused)
    return result


__all__ = ["psnr", "ssim", "entropy", "std", "q_abf", "mutual_information",
           "mi_fusion", "scd", "evaluate_pair"]
