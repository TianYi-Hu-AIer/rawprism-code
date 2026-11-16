"""Metadata-free, scene-level relative exposure estimation for linear RAW."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class ExposureReference:
    """Mapping from native under/over coordinates to a geometric-mean reference."""

    over_to_under_ratio: float
    under_to_reference: float
    over_to_reference: float
    valid_pixel_count: int
    valid_pixel_fraction: float
    log_ratio_p02: float
    log_ratio_p50: float
    log_ratio_p98: float
    min_luminance: float
    max_luminance: float
    trim_fraction: float
    estimation_mode: str = "strict_overlap"
    requested_min_luminance: float = 0.02
    requested_max_luminance: float = 0.95

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "ExposureReference":
        return cls(**value)


def packed_luminance(packed_linear: np.ndarray) -> np.ndarray:
    """Return a scalar packed-RAW exposure-estimation proxy without torch."""
    packed_linear = np.asarray(packed_linear)
    if packed_linear.ndim != 3 or packed_linear.shape[0] != 4:
        raise ValueError("Expected normalized packed RAW with shape [4, H, W]")
    return packed_linear.astype(np.float64, copy=False).mean(axis=0)


def _joint_log_ratios(
    under_y: np.ndarray,
    over_y: np.ndarray,
    min_luminance: float,
    max_luminance: float,
) -> np.ndarray:
    valid = (
        (under_y >= min_luminance)
        & (under_y <= max_luminance)
        & (over_y >= min_luminance)
        & (over_y <= max_luminance)
    )
    log_ratios = np.log(over_y[valid]) - np.log(under_y[valid])
    return log_ratios[np.isfinite(log_ratios)]


def _make_reference(
    log_ratios: np.ndarray,
    total_pixels: int,
    min_luminance: float,
    max_luminance: float,
    trim_fraction: float,
    estimation_mode: str,
    requested_min_luminance: float,
    requested_max_luminance: float,
) -> ExposureReference:
    low, high = np.quantile(log_ratios, [trim_fraction, 1.0 - trim_fraction])
    trimmed = log_ratios[(log_ratios >= low) & (log_ratios <= high)]
    fitted_log_ratio = float(trimmed.mean())
    ratio = float(math.exp(fitted_log_ratio))
    if not math.isfinite(ratio) or ratio <= 0.0:
        raise ValueError("Estimated a non-positive or non-finite exposure ratio")
    root_ratio = math.sqrt(ratio)
    return ExposureReference(
        over_to_under_ratio=ratio,
        under_to_reference=root_ratio,
        over_to_reference=1.0 / root_ratio,
        valid_pixel_count=int(trimmed.size),
        valid_pixel_fraction=float(trimmed.size / total_pixels),
        log_ratio_p02=float(np.quantile(log_ratios, 0.02)),
        log_ratio_p50=float(np.quantile(log_ratios, 0.50)),
        log_ratio_p98=float(np.quantile(log_ratios, 0.98)),
        min_luminance=float(min_luminance),
        max_luminance=float(max_luminance),
        trim_fraction=float(trim_fraction),
        estimation_mode=estimation_mode,
        requested_min_luminance=float(requested_min_luminance),
        requested_max_luminance=float(requested_max_luminance),
    )


def estimate_exposure_reference(
    under_linear: np.ndarray,
    over_linear: np.ndarray,
    min_luminance: float = 0.02,
    max_luminance: float = 0.95,
    trim_fraction: float = 0.02,
    min_valid_pixels: int = 1024,
) -> ExposureReference:
    """Estimate one robust exposure ratio for an entire aligned scene.

    The first attempt uses jointly well-exposed pixels selected by the caller's
    thresholds.  Extreme brackets sometimes have no overlap in that strict
    interval, although both frames still contain usable signal.  Only in that
    case, a recorded fallback uses the conservative RAW-safe interval
    ``0.001 <= Y <= 0.995``.  If even this has insufficient overlap, the pair
    contains no data-only basis for a reliable exposure ratio and is rejected.
    """
    if under_linear.shape != over_linear.shape:
        raise ValueError("Under and over packed RAW shapes must match")
    if not 0.0 < min_luminance < max_luminance <= 1.0:
        raise ValueError("Require 0 < min_luminance < max_luminance <= 1")
    if not 0.0 <= trim_fraction < 0.5:
        raise ValueError("trim_fraction must be in [0, 0.5)")
    if min_valid_pixels < 1:
        raise ValueError("min_valid_pixels must be positive")

    under_y = packed_luminance(under_linear)
    over_y = packed_luminance(over_linear)
    strict_logs = _joint_log_ratios(under_y, over_y, min_luminance, max_luminance)
    if strict_logs.size >= min_valid_pixels:
        return _make_reference(
            strict_logs,
            under_y.size,
            min_luminance,
            max_luminance,
            trim_fraction,
            "strict_overlap",
            min_luminance,
            max_luminance,
        )

    relaxed_minimum = min(min_luminance, 0.001)
    relaxed_maximum = max(max_luminance, 0.995)
    relaxed_logs = _joint_log_ratios(under_y, over_y, relaxed_minimum, relaxed_maximum)
    if relaxed_logs.size < min_valid_pixels:
        raise ValueError(
            "Exposure estimation has only {} strict and {} relaxed jointly valid pixels; need at least {}. "
            "This pair has no usable common linear RAW range after black subtraction.".format(
                strict_logs.size, relaxed_logs.size, min_valid_pixels
            )
        )
    return _make_reference(
        relaxed_logs,
        under_y.size,
        relaxed_minimum,
        relaxed_maximum,
        trim_fraction,
        "relaxed_overlap",
        min_luminance,
        max_luminance,
    )
