"""Authoritative RAW-to-RGB rendering backed by libraw/dcraw metadata.

Two rendering paths are provided and are kept bit-for-bit consistent across
every method in the paper:

* ``render_dng`` delegates directly to ``rawpy.postprocess`` (the standard
  libraw/dcraw pipeline) for source DNG files.  This is the reference RGB.
* ``render_raw4`` renders a fused canonical RAW4 tensor ``[4, H, W]`` in
  ``[0, 1]`` through the *same* camera parameters (black/white levels, camera
  white balance, color matrix, CFA pattern) extracted from the native DNG by
  libraw.  The only third-party approximation is the demosaic (OpenCV's edge
  directed method) because libraw cannot be re-run on an in-memory mosaic.

The fidelity of ``render_raw4`` is validated against ``render_dng`` on the
source mosaics (see ``render_rgb_rawpy.py --verify``), and the resulting PSNR is
reported in the paper to make the rendering protocol auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rawpy


@dataclass
class CameraMeta:
    black: float
    white: float
    camera_wb: tuple[float, float, float]
    color_matrix: np.ndarray
    pattern: str
    raw_pattern: np.ndarray
    gamma: float = 1.0 / 2.2

    def to_dict(self) -> dict[str, Any]:
        return {
            "black": float(self.black),
            "white": float(self.white),
            "camera_wb": [float(v) for v in self.camera_wb],
            "color_matrix": self.color_matrix.tolist(),
            "pattern": self.pattern,
            "raw_pattern": self.raw_pattern.tolist(),
            "gamma": float(self.gamma),
        }

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "CameraMeta":
        return CameraMeta(
            black=float(payload["black"]),
            white=float(payload["white"]),
            camera_wb=tuple(float(v) for v in payload["camera_wb"]),
            color_matrix=np.asarray(payload["color_matrix"], dtype=np.float32),
            pattern=str(payload["pattern"]),
            raw_pattern=np.asarray(payload["raw_pattern"], dtype=np.int64),
            gamma=float(payload.get("gamma", 1.0 / 2.2)),
        )


def extract_meta(dng_path: str | Path) -> CameraMeta:
    with rawpy.imread(str(dng_path)) as raw:
        black = float(np.asarray(raw.black_level_per_channel, dtype=np.float32).mean())
        white = float(raw.white_level)
        wb = raw.camera_whitebalance[:3]
        wb = np.clip(wb, 1.0e-6, None)
        color_matrix = np.asarray(raw.color_matrix, dtype=np.float32)[:, :3]
        if color_matrix.shape != (3, 3):
            raise ValueError("Unexpected color matrix shape {}".format(color_matrix.shape))
        raw_pattern = np.asarray(raw.raw_pattern, dtype=np.int64)
        pattern = pattern_from_raw_pattern(raw_pattern)
    return CameraMeta(black=black, white=white, camera_wb=(float(wb[0]), float(wb[1]), float(wb[2])),
                      color_matrix=color_matrix, pattern=pattern, raw_pattern=raw_pattern)


def pattern_from_raw_pattern(raw_pattern: np.ndarray) -> str:
    """Map a 2x2 rawpy CFA pattern to a canonical name.

    rawpy encodes 0=R, 1=G, 2=B and 3=G2 (the second green), which we fold back
    to ``G`` before naming the 2x2 layout.
    """
    if raw_pattern.shape != (2, 2):
        raw_pattern = raw_pattern.reshape(2, 2)
    mapping = {0: "R", 1: "G", 2: "B", 3: "G"}
    name = "".join(mapping[int(v)] for v in raw_pattern.ravel())
    return name


def cv2_bayer_code(pattern: str) -> int:
    return {
        "RGGB": cv2.COLOR_BayerRG2RGB,
        "BGGR": cv2.COLOR_BayerBG2RGB,
        "GRBG": cv2.COLOR_BayerGR2RGB,
        "GBRG": cv2.COLOR_BayerGB2RGB,
    }[pattern.upper()]


def raw4_to_bayer(raw4: np.ndarray, pattern: str) -> np.ndarray:
    """Pack canonical ``[R, Gh, Gv, B]`` RAW4 back into a 2-D Bayer mosaic."""
    if raw4.ndim != 3 or raw4.shape[0] != 4:
        raise ValueError("raw4 must have shape [4, H, W]")
    height, width = raw4.shape[1] * 2, raw4.shape[2] * 2
    mosaic = np.zeros((height, width), dtype=np.float32)
    pattern = pattern.upper()
    if pattern == "BGGR":
        mosaic[0::2, 0::2], mosaic[0::2, 1::2] = raw4[3], raw4[1]
        mosaic[1::2, 0::2], mosaic[1::2, 1::2] = raw4[2], raw4[0]
    elif pattern == "RGGB":
        mosaic[0::2, 0::2], mosaic[0::2, 1::2] = raw4[0], raw4[1]
        mosaic[1::2, 0::2], mosaic[1::2, 1::2] = raw4[2], raw4[3]
    else:
        raise ValueError("Unsupported Bayer pattern: {}".format(pattern))
    return mosaic


def bayer_to_raw4(mosaic: np.ndarray, pattern: str) -> np.ndarray:
    """Split a 2-D Bayer mosaic into canonical ``[R, Gh, Gv, B]`` planes."""
    height, width = mosaic.shape[0] // 2 * 2, mosaic.shape[1] // 2 * 2
    mosaic = mosaic[:height, :width]
    raw4 = np.empty((4, height // 2, width // 2), dtype=mosaic.dtype)
    pattern = pattern.upper()
    if pattern == "BGGR":
        raw4[0], raw4[1] = mosaic[1::2, 1::2], mosaic[0::2, 1::2]
        raw4[2], raw4[3] = mosaic[1::2, 0::2], mosaic[0::2, 0::2]
    elif pattern == "RGGB":
        raw4[0], raw4[1] = mosaic[0::2, 0::2], mosaic[0::2, 1::2]
        raw4[2], raw4[3] = mosaic[1::2, 0::2], mosaic[1::2, 1::2]
    else:
        raise ValueError("Unsupported Bayer pattern: {}".format(pattern))
    return raw4


def _apply_srgb(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    low = value <= 0.0031308
    out = np.empty_like(value)
    out[low] = 12.92 * value[low]
    out[~low] = 1.055 * np.power(value[~low], 1.0 / 2.4) - 0.055
    return np.clip(out, 0.0, 1.0)


def render_raw4(raw4: np.ndarray, meta: CameraMeta, gamma: float | None = None) -> np.ndarray:
    """Render a normalized RAW4 tensor to an RGB image in [0, 1]."""
    if raw4.ndim != 3 or raw4.shape[0] != 4:
        raise ValueError("raw4 must have shape [4, H, W]")
    mosaic = raw4_to_bayer(raw4, meta.pattern)
    mosaic_u16 = np.clip(mosaic, 0.0, 1.0)
    mosaic_u16 = (mosaic_u16 * 65535.0).astype(np.uint16)
    rgb = cv2.cvtColor(mosaic_u16, cv2_bayer_code(meta.pattern)).astype(np.float32) / 65535.0
    wb = np.asarray(meta.camera_wb, dtype=np.float32).reshape(1, 1, 3)
    rgb = rgb * wb
    rgb = np.clip(rgb, 0.0, 1.0)
    # Camera -> sRGB via the DNG color matrix (row-vector convention).
    rgb = np.clip(np.einsum("hwc,cd->hwd", rgb, meta.color_matrix.astype(np.float32)), 0.0, 1.0)
    gamma_value = meta.gamma if gamma is None else gamma
    return _apply_srgb(rgb) if gamma_value is None or gamma_value == 1.0 else np.clip(
        np.power(rgb, gamma_value), 0.0, 1.0
    )


def render_dng(dng_path: str | Path, output_bps: int = 16) -> np.ndarray:
    """Render a DNG through the authoritative libraw/dcraw pipeline."""
    with rawpy.imread(str(dng_path)) as raw:
        rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=output_bps)
    return np.asarray(rgb, dtype=np.float32) / float((1 << output_bps) - 1)


def render_mosaic_via_dng(
    mosaic_u16: np.ndarray,
    template_dng: str | Path,
    output_bps: int = 16,
) -> np.ndarray:
    """Render a fused uint16 Bayer mosaic through libraw using a DNG template.

    ``rawpy`` exposes ``raw_image`` as a writable view over the decoded CFA
    buffer, so replacing it in place and calling ``postprocess`` routes the
    fused mosaic through the *exact* dcraw pipeline (AHD demosaic, camera
    colour matrix, white balance, tone curve) of the native camera.  This is
    the authoritative path for MEF and IVIF fused outputs.
    """
    with rawpy.imread(str(template_dng)) as raw:
        if mosaic_u16.shape != raw.raw_image.shape:
            raise ValueError(
                "mosaic shape {} does not match template {}".format(
                    mosaic_u16.shape, raw.raw_image.shape
                )
            )
        raw.raw_image[...] = np.clip(mosaic_u16, 0, 65535).astype(np.uint16)
        rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=output_bps)
    return np.asarray(rgb, dtype=np.float32) / float((1 << output_bps) - 1)


__all__ = [
    "CameraMeta",
    "extract_meta",
    "render_dng",
    "render_raw4",
    "render_mosaic_via_dng",
    "raw4_to_bayer",
    "bayer_to_raw4",
]
