"""Blind Poisson-Gaussian noise calibration for cached packed linear RAW."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .raw import RawDataError, RawPair


NOISE_MODEL_FILENAME = "noise_model.json"
_MEDIAN_CHI2_DOF1 = 0.4549364231


@dataclass(frozen=True)
class NoiseModel:
    """Variance model ``a*x + b`` for native normalized packed RAW values."""

    under_a: list[float]
    under_b: list[float]
    over_a: list[float]
    over_b: list[float]
    method: str
    samples_per_scene: int
    flat_fraction: float
    training_scenes: list[str]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "NoiseModel":
        model = cls(**value)
        for name in ("under_a", "under_b", "over_a", "over_b"):
            values = getattr(model, name)
            if len(values) != 4 or any(not np.isfinite(item) or item < 0.0 for item in values):
                raise ValueError("{} must contain four finite non-negative values".format(name))
        return model


def noise_model_path(cache_root: str | Path) -> Path:
    """Accept either a cache/output directory or an explicit noise-model file."""
    location = Path(cache_root)
    return location if location.suffix.lower() == ".json" else location / NOISE_MODEL_FILENAME


def write_noise_model(cache_root: str | Path, model: NoiseModel) -> Path:
    path = noise_model_path(cache_root)
    path.write_text(json.dumps(model.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def load_noise_model(cache_root: str | Path) -> NoiseModel:
    path = noise_model_path(cache_root)
    if not path.is_file():
        raise RawDataError("Missing {}. Re-run prepare_cache.py to estimate the RAW noise model.".format(path))
    try:
        return NoiseModel.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RawDataError("Invalid noise model {}: {}".format(path, error)) from error


def fit_noise_model(
    pairs: Sequence[RawPair],
    cache_root: str | Path,
    black_level: float,
    white_level: float,
    samples_per_scene: int = 2048,
    flat_fraction: float = 0.25,
    bins: int = 24,
    seed: int = 1337,
) -> NoiseModel:
    """Fit a conservative blind variance model using sampled low-gradient pixels.

    A four-neighbour residual is used only in locally flat regions.  Its median
    squared value estimates the local noise variance after correcting for the
    residual filter and the median of a chi-square random variable.  This is
    not a camera-calibrated PTC; it is a reproducible no-extra-metadata proxy
    for source reliability.
    """
    if not pairs:
        raise ValueError("Need at least one training scene to fit the noise model")
    if samples_per_scene < 64:
        raise ValueError("samples_per_scene must be at least 64")
    if not 0.01 <= flat_fraction <= 1.0:
        raise ValueError("flat_fraction must be in [0.01, 1]")
    if bins < 4:
        raise ValueError("bins must be at least four")
    if not white_level > black_level:
        raise ValueError("white_level must exceed black_level")
    scale = float(white_level - black_level)
    cache_root = Path(cache_root)
    rng = np.random.default_rng(seed)
    coefficients: dict[str, tuple[list[float], list[float]]] = {}
    for exposure in ("under", "over"):
        means: list[list[np.ndarray]] = [[] for _ in range(4)]
        variances: list[list[np.ndarray]] = [[] for _ in range(4)]
        for pair in pairs:
            source = np.load(cache_root / pair.scene / (exposure + ".npy"), mmap_mode="r")
            if source.ndim != 3 or source.shape[0] != 4 or min(source.shape[1:]) < 3:
                raise RawDataError("Invalid packed cache for {} {}".format(pair.scene, exposure))
            height, width = source.shape[1:]
            count = min(samples_per_scene, (height - 2) * (width - 2))
            rows = rng.integers(1, height - 1, size=count)
            cols = rng.integers(1, width - 1, size=count)
            for channel in range(4):
                image = (np.asarray(source[channel], dtype=np.float32) - black_level) / scale
                image = np.clip(image, 0.0, 1.0)
                center = image[rows, cols]
                neighbours = (
                    image[rows - 1, cols]
                    + image[rows + 1, cols]
                    + image[rows, cols - 1]
                    + image[rows, cols + 1]
                ) * 0.25
                residual = center - neighbours
                gradient_proxy = np.abs(center - neighbours)
                valid = (center > 0.005) & (center < 0.98)
                if not np.any(valid):
                    continue
                threshold = np.quantile(gradient_proxy[valid], flat_fraction)
                selected = valid & (gradient_proxy <= threshold)
                means[channel].append(center[selected])
                # For independent equal-variance samples, this residual has
                # variance 1.25*sigma^2. Median residual^2 is chi-square
                # biased by the factor below.
                variances[channel].append(np.square(residual[selected]) / (1.25 * _MEDIAN_CHI2_DOF1))
        a_values, b_values = [], []
        for channel in range(4):
            if not means[channel]:
                a_values.append(0.0)
                b_values.append(1e-6)
                continue
            mean = np.concatenate(means[channel])
            variance = np.concatenate(variances[channel])
            a_value, b_value = _fit_binned_nonnegative_line(mean, variance, bins)
            a_values.append(a_value)
            b_values.append(b_value)
        coefficients[exposure] = (a_values, b_values)
    return NoiseModel(
        under_a=coefficients["under"][0],
        under_b=coefficients["under"][1],
        over_a=coefficients["over"][0],
        over_b=coefficients["over"][1],
        method="blind_flat_residual_poisson_gaussian_v1",
        samples_per_scene=int(samples_per_scene),
        flat_fraction=float(flat_fraction),
        training_scenes=[pair.scene for pair in pairs],
    )


def _fit_binned_nonnegative_line(mean: np.ndarray, variance: np.ndarray, bins: int) -> tuple[float, float]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    x_values, y_values = [], []
    for index in range(bins):
        include = (mean >= edges[index]) & (mean < edges[index + 1])
        if np.count_nonzero(include) < 32:
            continue
        x_values.append(float(np.median(mean[include])))
        y_values.append(float(np.median(variance[include])))
    if len(x_values) < 2:
        return 0.0, max(float(np.median(variance)), 1e-8)
    design = np.stack([np.asarray(x_values), np.ones(len(x_values))], axis=1)
    slope, intercept = np.linalg.lstsq(design, np.asarray(y_values), rcond=None)[0]
    return max(float(slope), 0.0), max(float(intercept), 1e-8)
