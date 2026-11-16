"""A minimal, reproducible RAW-to-RGB evaluation ISP.

The same function is applied to every fused RAW4 and every source RAW so that
quantitative comparison is not confounded by different rendering pipelines.
Steps: Bayer demosaic -> white balance -> simple gamma -> clip to [0, 1].
"""

from __future__ import annotations

import cv2
import numpy as np


# camera white balance gains [R, G, B] used for the field-shot iPhone 17 Pro
# datasets (MEF/MFF); IVIF uses neutral gains because its visible branch is
# reconstructed RAW from already-rendered RGB.
DEFAULT_WB = (1.0, 1.0, 1.0)
IPHONE_WB = (2.6845703125, 1.0, 1.7258301973342896)


def raw4_to_bayer(raw4: np.ndarray, pattern: str = "BGGR") -> np.ndarray:
    """Pack canonical [R, Gh, Gv, B] RAW4 back into a 2-D Bayer mosaic."""
    if raw4.ndim != 3 or raw4.shape[0] != 4:
        raise ValueError("raw4 must have shape [4, H, W]")
    pattern = pattern.upper()
    h, w = raw4.shape[1] * 2, raw4.shape[2] * 2
    mosaic = np.zeros((h, w), dtype=np.float32)
    if pattern == "BGGR":
        mosaic[0::2, 0::2] = raw4[3]
        mosaic[0::2, 1::2] = raw4[1]
        mosaic[1::2, 0::2] = raw4[2]
        mosaic[1::2, 1::2] = raw4[0]
    elif pattern == "RGGB":
        mosaic[0::2, 0::2] = raw4[0]
        mosaic[0::2, 1::2] = raw4[1]
        mosaic[1::2, 0::2] = raw4[2]
        mosaic[1::2, 1::2] = raw4[3]
    else:
        raise ValueError("Unsupported Bayer pattern: {}".format(pattern))
    return mosaic


def render_raw4(raw4: np.ndarray, pattern: str = "BGGR", white_balance: tuple[float, float, float] = IPHONE_WB,
                gamma: float = 1.0 / 2.2) -> np.ndarray:
    """Render a normalized RAW4 tensor [4,H,W] in [0,1] to an RGB image in [0,1]."""
    mosaic = raw4_to_bayer(raw4, pattern)
    mosaic_u16 = np.clip(mosaic, 0.0, 1.0)
    if pattern == "BGGR":
        rgb = cv2.cvtColor((mosaic_u16 * 65535).astype(np.uint16), cv2.COLOR_BayerBG2RGB).astype(np.float32) / 65535.0
    elif pattern == "RGGB":
        rgb = cv2.cvtColor((mosaic_u16 * 65535).astype(np.uint16), cv2.COLOR_BayerRG2RGB).astype(np.float32) / 65535.0
    else:
        raise ValueError("Unsupported Bayer pattern: {}".format(pattern))
    rgb = rgb * np.asarray(white_balance, dtype=np.float32).reshape(1, 1, 3)
    rgb = np.clip(rgb, 0.0, None)
    rgb = np.power(rgb, gamma)
    return np.clip(rgb, 0.0, 1.0)


__all__ = ["raw4_to_bayer", "render_raw4", "DEFAULT_WB", "IPHONE_WB"]
