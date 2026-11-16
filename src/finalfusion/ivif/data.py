"""IVIF dataset and RAW input helpers."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def _pack_rggb(image: np.ndarray) -> np.ndarray:
    height, width = image.shape
    image = image[:height // 2 * 2, :width // 2 * 2]
    return np.stack((image[0::2, 0::2], image[0::2, 1::2], image[1::2, 0::2], image[1::2, 1::2])).astype(np.float32)


def _read_ivif_pair(scene: Path, black_level: float, white_level: float) -> tuple[np.ndarray, np.ndarray]:
    import cv2
    import tifffile

    visible = np.asarray(tifffile.imread(str(scene / "vis_raw.dng")), dtype=np.float32)
    if visible.ndim != 2:
        raise ValueError("{} visible DNG is not a single Bayer plane".format(scene))
    visible = np.clip((visible - black_level) / (white_level - black_level), 0.0, 1.0)
    infrared = cv2.imread(str(scene / "ir.png"), cv2.IMREAD_UNCHANGED)
    if infrared is None:
        raise ValueError("Failed to read {}/ir.png".format(scene))
    if infrared.ndim == 3:
        infrared = cv2.cvtColor(infrared, cv2.COLOR_BGR2GRAY)
    divisor = 65535.0 if infrared.dtype == np.uint16 else 255.0 if infrared.dtype == np.uint8 else 1.0
    infrared = np.clip(infrared.astype(np.float32) / divisor, 0.0, 1.0)
    height, width = min(visible.shape[0], infrared.shape[0]) // 2 * 2, min(visible.shape[1], infrared.shape[1]) // 2 * 2
    if min(height, width) < 2:
        raise ValueError("{} has no common even RAW area".format(scene))
    return visible[:height, :width], infrared[:height, :width]


class IvifDataset(Dataset):
    def __init__(self, scenes: list[Path], patch_size_bayer: int | None, training: bool,
                 black_level: float, white_level: float) -> None:
        self.scenes, self.patch_size_bayer, self.training = list(scenes), patch_size_bayer, bool(training)
        self.black_level, self.white_level = float(black_level), float(white_level)
        if not self.scenes or not self.white_level > self.black_level:
            raise ValueError("IVIF needs scenes and white_level > black_level")

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, index: int) -> dict[str, Any]:
        scene = self.scenes[index]
        visible, infrared = _read_ivif_pair(scene, self.black_level, self.white_level)
        full_visible, full_infrared = _pack_rggb(visible), _pack_rggb(infrared)
        gains = full_infrared.mean(axis=(1, 2)) / (full_visible.mean(axis=(1, 2)) + 1.0e-6)
        if self.training and self.patch_size_bayer:
            crop_h, crop_w = min(visible.shape[0], self.patch_size_bayer) // 2 * 2, min(visible.shape[1], self.patch_size_bayer) // 2 * 2
            max_y, max_x = visible.shape[0] - crop_h, visible.shape[1] - crop_w
            top = random.randrange(max_y // 2 + 1) * 2 if max_y else 0
            left = random.randrange(max_x // 2 + 1) * 2 if max_x else 0
            visible, infrared = visible[top:top + crop_h, left:left + crop_w], infrared[top:top + crop_h, left:left + crop_w]
        visible_raw4, infrared_raw4 = _pack_rggb(visible), _pack_rggb(infrared)
        visible_raw4 = np.clip(visible_raw4 * gains[:, None, None], 0.0, 1.0).astype(np.float32)
        return {"source1": torch.from_numpy(visible_raw4), "source2": torch.from_numpy(infrared_raw4), "scene": scene.name}
