"""Inference-only, brightness-preserving source chromaticity correction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re

import cv2
import numpy as np


EPSILON = 1.0e-8


@dataclass(frozen=True)
class HardSourceChromaConfig:
    black_valid: float = 0.012
    saturation_valid: float = 0.970
    snr_start: float = 12.0
    snr_end: float = 36.0
    reliable_threshold: float = 0.20
    guided_radius: int = 3
    guided_epsilon: float = 1.0e-3
    morph_open: int = 3
    morph_close: int = 5
    edge_quantile: float = 0.85

    def validate(self) -> None:
        if not 0.0 < self.black_valid < self.saturation_valid < 1.0:
            raise ValueError("Require 0 < black_valid < saturation_valid < 1")
        if not 0.0 < self.snr_start < self.snr_end:
            raise ValueError("Require 0 < snr_start < snr_end")
        if not 0.0 <= self.reliable_threshold <= 1.0:
            raise ValueError("reliable_threshold must be in [0, 1]")
        if self.guided_radius < 0 or self.guided_epsilon <= 0.0:
            raise ValueError("Invalid guided-filter parameters")
        if min(self.morph_open, self.morph_close) < 1 or self.morph_open % 2 == 0 or self.morph_close % 2 == 0:
            raise ValueError("Morphology sizes must be positive odd integers")
        if not 0.0 < self.edge_quantile < 1.0:
            raise ValueError("edge_quantile must be in (0, 1)")

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def metadata_path(roots: list[str | Path], scene: str, exposure: str) -> Path:
    """Resolve one numeric metadata file from old nested or new flat layouts."""
    attempted = []
    for root in (Path(value).expanduser() for value in roots):
        for base in (root / scene / "dng_components", root / scene):
            candidate = base / exposure / (scene + "_metadata_numeric.json")
            attempted.append(str(candidate))
            if candidate.is_file():
                return candidate
    raise FileNotFoundError("No {} metadata for {}. Tried: {}".format(exposure, scene, "; ".join(attempted)))


def metadata_noise_profile(path: str | Path) -> tuple[float, float]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Cannot read metadata {}".format(path)) from error
    values = payload[0] if isinstance(payload, list) else payload
    tag = values.get("EXIF:SubIFD:Image:NoiseProfile", {})
    encoded = tag.get("val", tag) if isinstance(tag, dict) else tag
    numbers = [float(value) for value in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(encoded))]
    if len(numbers) < 2 or min(numbers[:2]) < 0.0:
        raise ValueError("Missing or invalid DNG NoiseProfile in {}".format(path))
    return numbers[0], numbers[1]


def _smoothstep(value: np.ndarray, start: float, end: float) -> np.ndarray:
    fraction = np.clip((value - start) / (end - start), 0.0, 1.0)
    return fraction * fraction * (3.0 - 2.0 * fraction)


def _source_quality(native: np.ndarray, scale: float, noise: tuple[float, float], config: HardSourceChromaConfig):
    variance = noise[0] * np.maximum(native, 0.0) + noise[1]
    snr = native / np.sqrt(np.maximum(variance, EPSILON))
    low_snr = np.quantile(snr, 0.25, axis=0).astype(np.float32)
    valid = ((native >= config.black_valid) & (native <= config.saturation_valid)).all(axis=0)
    confidence = _smoothstep(low_snr, config.snr_start, config.snr_end).astype(np.float32)
    return valid, low_snr, confidence * valid.astype(np.float32)


def _nearest_labels(label: np.ndarray, seed: np.ndarray) -> np.ndarray:
    if not np.any(seed):
        return label.copy()
    _, identifiers = cv2.distanceTransformWithLabels((~seed).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    lookup = np.zeros(int(identifiers.max()) + 1, dtype=np.uint8)
    lookup[identifiers[seed]] = label[seed]
    result = label.copy()
    result[~seed] = lookup[identifiers[~seed]]
    return result


def _guided_filter(guide: np.ndarray, source: np.ndarray, radius: int, epsilon: float) -> np.ndarray:
    size = 2 * radius + 1
    mean_g = cv2.boxFilter(guide, -1, (size, size), borderType=cv2.BORDER_REFLECT_101)
    mean_s = cv2.boxFilter(source, -1, (size, size), borderType=cv2.BORDER_REFLECT_101)
    covariance = cv2.boxFilter(guide * source, -1, (size, size), borderType=cv2.BORDER_REFLECT_101) - mean_g * mean_s
    variance = cv2.boxFilter(guide * guide, -1, (size, size), borderType=cv2.BORDER_REFLECT_101) - mean_g * mean_g
    slope = covariance / (variance + epsilon)
    intercept = mean_s - slope * mean_g
    return cv2.boxFilter(slope, -1, (size, size), borderType=cv2.BORDER_REFLECT_101) * guide + cv2.boxFilter(intercept, -1, (size, size), borderType=cv2.BORDER_REFLECT_101)


def _morphology(label: np.ndarray, guide: np.ndarray, config: HardSourceChromaConfig) -> tuple[np.ndarray, np.ndarray]:
    opened = cv2.morphologyEx(label.astype(np.uint8) * 255, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.morph_open, config.morph_open)), borderType=cv2.BORDER_REFLECT_101)
    proposed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.morph_close, config.morph_close)), borderType=cv2.BORDER_REFLECT_101) > 127
    magnitude = np.hypot(cv2.Sobel(guide, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(guide, cv2.CV_32F, 0, 1, ksize=3))
    edge = magnitude >= float(np.quantile(magnitude, config.edge_quantile))
    result = label.copy()
    result[~edge] = proposed[~edge]
    return result, edge


def hard_source_chroma_correction(
    fused_network: np.ndarray, under_native: np.ndarray, over_native: np.ndarray,
    under_reference_scale: float, over_reference_scale: float,
    under_noise: tuple[float, float], over_noise: tuple[float, float],
    config: HardSourceChromaConfig = HardSourceChromaConfig(),
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Keep network brightness and select all final CFA ratios from one source."""
    config.validate()
    values = [np.asarray(value, dtype=np.float32) for value in (fused_network, under_native, over_native)]
    if any(value.ndim != 3 or value.shape[0] != 4 for value in values) or len({value.shape for value in values}) != 1:
        raise ValueError("network and source arrays must share packed RAW shape [4,H,W]")
    fused, under, over = values
    under_valid, under_snr, under_confidence = _source_quality(under, under_reference_scale, under_noise, config)
    over_valid, over_snr, over_confidence = _source_quality(over, over_reference_scale, over_noise, config)
    seed = (under_confidence >= config.reliable_threshold) | (over_confidence >= config.reliable_threshold)
    raw_over = over_confidence >= under_confidence
    filled = _nearest_labels(raw_over.astype(np.uint8), seed).astype(bool)
    guide = fused.mean(axis=0)
    guide = np.clip(guide / max(float(np.quantile(guide, 0.995)), EPSILON), 0.0, 1.0).astype(np.float32)
    filtered = _guided_filter(guide, 2.0 * filled.astype(np.float32) - 1.0, config.guided_radius, config.guided_epsilon) >= 0.0
    selected_over, edge = _morphology(filtered, guide, config)
    strong = seed & (np.abs(over_confidence - under_confidence) >= 0.25)
    selected_over[strong] = raw_over[strong]
    saturated_over = (over > config.saturation_valid).any(axis=0)
    forced_under = selected_over & saturated_over
    selected_over[forced_under] = False
    post, _ = _morphology(selected_over, guide, config)
    reenforced = post & saturated_over
    post[saturated_over] = False
    changed = post != selected_over
    selected_over = post
    selected = np.where(selected_over[None], over * over_reference_scale, under * under_reference_scale)
    selected_mean = selected.mean(axis=0, keepdims=True)
    luminance = fused.mean(axis=0, keepdims=True)
    corrected = (luminance * selected / np.maximum(selected_mean, EPSILON)).astype(np.float32)
    return corrected, {
        "selected_over": selected_over, "under_low_percentile_snr": under_snr, "over_low_percentile_snr": over_snr,
        "under_all_channel_valid": under_valid, "over_all_channel_valid": over_valid, "both_unreliable": ~seed,
        "over_native_saturated": saturated_over, "forced_under_from_saturated_over": forced_under,
        "post_saturation_morphology_changed": changed, "post_saturation_reenforced": reenforced,
        "edge_protected": edge, "zero_selected_source": selected_mean[0] <= EPSILON,
        "brightness_absolute_error": np.abs(corrected.mean(axis=0) - luminance[0]),
    }
