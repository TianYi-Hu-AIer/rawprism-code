"""Scene splits, packed-RAW cache access, and crop datasets."""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import random
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .exposure import ExposureReference
from .raw import RawDataError, RawPair


def split_pairs(
    pairs: Sequence[RawPair],
    seed: int,
    val_fraction: float,
    test_fraction: float,
) -> dict[str, list[RawPair]]:
    if not 0.0 <= val_fraction < 1.0 or not 0.0 <= test_fraction < 1.0:
        raise ValueError("Split fractions must be in [0, 1)")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be less than one")
    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    test_count = max(1, round(total * test_fraction)) if test_fraction else 0
    val_count = max(1, round(total * val_fraction)) if val_fraction else 0
    train_count = total - val_count - test_count
    if train_count < 1:
        raise ValueError("The split leaves no training scenes")
    return {
        "train": shuffled[:train_count],
        "val": shuffled[train_count:train_count + val_count],
        "test": shuffled[train_count + val_count:],
    }


def split_pairs_from_json(pairs: Sequence[RawPair], path: str | Path) -> dict[str, list[RawPair]]:
    """Reuse a prior scene-level split while rediscovering paths from ``data-root``.

    The supplied JSON may contain stale absolute file paths from another
    machine.  Only its scene identifiers are authoritative; current paths and
    Bayer labels always come from a fresh ``discover_pairs(data_root)`` call.
    """
    split_path = Path(path).expanduser().resolve()
    try:
        content = json.loads(split_path.read_text(encoding="utf-8"))
        requested = content["splits"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise RawDataError("Invalid split JSON {}: {}".format(split_path, error)) from error
    if set(requested) != {"train", "val", "test"}:
        raise RawDataError("Split JSON must contain exactly train, val, and test lists")
    available = {pair.scene: pair for pair in pairs}
    split: dict[str, list[RawPair]] = {}
    all_scenes: list[str] = []
    for name in ("train", "val", "test"):
        records = requested[name]
        if not isinstance(records, list):
            raise RawDataError("Split '{}' must be a list".format(name))
        scenes = []
        for record in records:
            scene = record.get("scene") if isinstance(record, dict) else record
            if not isinstance(scene, str) or not scene:
                raise RawDataError("Split '{}' contains an invalid scene record".format(name))
            scenes.append(scene)
        missing = sorted(set(scenes) - set(available))
        if missing:
            raise RawDataError("Split '{}' names scenes absent from data-root: {}".format(name, missing))
        split[name] = [available[scene] for scene in scenes]
        all_scenes.extend(scenes)
    duplicates = sorted({scene for scene in all_scenes if all_scenes.count(scene) > 1})
    if duplicates:
        raise RawDataError("A scene appears more than once in split JSON: {}".format(duplicates))
    unassigned = sorted(set(available) - set(all_scenes))
    if unassigned:
        raise RawDataError(
            "Data root contains scenes absent from split JSON: {}".format(unassigned)
        )
    if not split["train"] or not split["val"] or not split["test"]:
        raise RawDataError("Split JSON must leave at least one scene in train, val, and test")
    return split


def write_split(path: str | Path, split: dict[str, list[RawPair]], seed: int) -> None:
    content = {
        "schema_version": 1,
        "seed": seed,
        "splits": {name: [pair.to_dict() for pair in pairs] for name, pairs in split.items()},
    }
    Path(path).write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")


def _open_packed(path: str) -> np.ndarray:
    """Prefer mmap, but support filesystems where mmap is unavailable."""
    try:
        return np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError):
        return np.load(path, allow_pickle=False)


def cache_path(cache_root: str | Path, scene: str, exposure: str) -> Path:
    return Path(cache_root) / scene / (exposure + ".npy")


def overlapping_tile_origins(height: int, width: int, patch_size: int, stride: int) -> tuple[tuple[int, int], ...]:
    """Return an end-aligned, fully covering grid of ``(top, left)`` origins.

    The last origin on each axis is explicitly aligned to the image edge.  With
    ``stride <= patch_size`` this means every packed pixel is covered at least
    once, including non-divisible image borders.
    """
    if min(height, width, patch_size, stride) <= 0:
        raise ValueError("Tile height, width, patch_size, and stride must be positive")
    if patch_size > height or patch_size > width:
        raise ValueError("Patch does not fit the packed image")
    if stride > patch_size:
        raise ValueError("tile stride must not exceed patch size; coverage would have gaps")

    def origins_for_axis(length: int) -> list[int]:
        maximum = length - patch_size
        origins = list(range(0, maximum + 1, stride))
        if origins[-1] != maximum:
            origins.append(maximum)
        return origins

    return tuple((top, left) for top in origins_for_axis(height) for left in origins_for_axis(width))


def _scene_cycle_permutation(count: int, scene: str, seed: int, cycle: int) -> np.ndarray:
    """Create a reproducible scene-specific no-replacement ordering per cycle."""
    if count < 1 or cycle < 0:
        raise ValueError("count must be positive and cycle must be non-negative")
    token = "{}:{}:{}".format(seed, scene, cycle).encode("utf-8")
    derived_seed = int.from_bytes(hashlib.blake2b(token, digest_size=8).digest(), "little")
    return np.random.default_rng(derived_seed).permutation(count)


class PackedRawPairDataset(Dataset):
    """Paired RAW crops from a cyclic, overlapping, no-replacement tile grid."""

    def __init__(
        self,
        pairs: Sequence[RawPair],
        cache_root: str | Path,
        black_level: float,
        white_level: float,
        patch_size: int,
        patches_per_scene: int,
        training: bool,
        tile_stride: int = 64,
        sampling_seed: int = 1337,
        context_pad: int = 16,
    ) -> None:
        if patch_size <= 10:
            raise ValueError("patch_size must be larger than the SSIM window")
        self.pairs = list(pairs)
        self.cache_root = Path(cache_root)
        self.black_level = float(black_level)
        self.white_level = float(white_level)
        self.patch_size = int(patch_size)
        self.patches_per_scene = int(patches_per_scene)
        self.training = bool(training)
        self.tile_stride = int(tile_stride)
        self.sampling_seed = int(sampling_seed)
        self.context_pad = int(context_pad)
        # A multiprocessing value keeps persistent DataLoader workers in sync.
        self._epoch = mp.Value("q", 0)
        if not self.white_level > self.black_level:
            raise ValueError("white_level must be greater than black_level")
        if self.patches_per_scene < 1:
            raise ValueError("patches_per_scene must be positive")
        if self.context_pad < 0:
            raise ValueError("context_pad must be non-negative")
        if not self.pairs:
            raise ValueError("Dataset has no pairs")
        if not self.cache_root.is_dir():
            raise RawDataError("Packed cache root does not exist: {}".format(self.cache_root))
        self.exposure_references = _read_exposure_references(self.cache_root)
        for pair in self.pairs:
            for exposure in ("under", "over"):
                path = cache_path(self.cache_root, pair.scene, exposure)
                if not path.is_file():
                    raise RawDataError("Missing packed cache file: {}".format(path))
            if pair.packed_height < patch_size or pair.packed_width < patch_size:
                raise RawDataError(
                    "Patch {} does not fit packed scene {} of size {}x{}".format(
                        patch_size, pair.scene, pair.packed_height, pair.packed_width
                    )
                )
            if pair.scene not in self.exposure_references:
                raise RawDataError("Missing exposure reference for scene {}".format(pair.scene))
        self._tile_origins = {
            pair.scene: overlapping_tile_origins(
                pair.packed_height, pair.packed_width, self.patch_size, self.tile_stride
            )
            for pair in self.pairs
        }

    def __len__(self) -> int:
        return len(self.pairs) * (self.patches_per_scene if self.training else 1)

    def set_epoch(self, epoch: int) -> None:
        """Advance deterministic training sampling, including persistent workers."""
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        with self._epoch.get_lock():
            self._epoch.value = int(epoch)

    def coverage_summary(self) -> dict[str, int]:
        """Expose the grid cardinality used by the current fixed-resolution dataset."""
        counts = {len(origins) for origins in self._tile_origins.values()}
        if len(counts) != 1:
            raise RawDataError("Training scenes have mixed packed dimensions; inspect per-scene tile grids")
        tile_count = counts.pop()
        return {
            "patch_size": self.patch_size,
            "context_pad": self.context_pad,
            "model_patch_size": self.patch_size + 2 * self.context_pad,
            "tile_stride": self.tile_stride,
            "candidate_tiles_per_scene": tile_count,
            "tiles_per_scene_per_epoch": self.patches_per_scene,
            "epochs_per_full_coverage_cycle": int(np.ceil(tile_count / self.patches_per_scene)),
        }

    def tile_origin(self, pair: RawPair, sample_index: int, epoch: int | None = None) -> tuple[int, int]:
        """Return one origin from the scene grid without replacement within a cycle."""
        if not self.training:
            raise RuntimeError("Validation datasets use their fixed center crop")
        if not 0 <= sample_index < self.patches_per_scene:
            raise ValueError("sample_index must be within one epoch's tiles_per_scene")
        if epoch is None:
            epoch = int(self._epoch.value)
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        origins = self._tile_origins[pair.scene]
        absolute_index = epoch * self.patches_per_scene + sample_index
        cycle, index_in_cycle = divmod(absolute_index, len(origins))
        permutation = _scene_cycle_permutation(len(origins), pair.scene, self.sampling_seed, cycle)
        return origins[int(permutation[index_in_cycle])]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        pair = self.pairs[index % len(self.pairs)]
        under = _open_packed(str(cache_path(self.cache_root, pair.scene, "under")))
        over = _open_packed(str(cache_path(self.cache_root, pair.scene, "over")))
        expected_shape = (4, pair.packed_height, pair.packed_width)
        if under.shape != expected_shape or over.shape != expected_shape:
            raise RawDataError("Packed cache shape mismatch for {}".format(pair.scene))
        if self.training:
            sample_index = index // len(self.pairs)
            top, left = self.tile_origin(pair, sample_index)
        else:
            max_y = pair.packed_height - self.patch_size
            max_x = pair.packed_width - self.patch_size
            top, left = max_y // 2, max_x // 2
        under_crop = _crop_with_reflect_context(under, top, left, self.patch_size, self.context_pad)
        over_crop = _crop_with_reflect_context(over, top, left, self.patch_size, self.context_pad)
        scale = self.white_level - self.black_level
        under_linear = np.clip((np.asarray(under_crop, dtype=np.float32) - self.black_level) / scale, 0.0, 1.0)
        over_linear = np.clip((np.asarray(over_crop, dtype=np.float32) - self.black_level) / scale, 0.0, 1.0)
        return {
            "under": torch.from_numpy(np.ascontiguousarray(under_linear)),
            "over": torch.from_numpy(np.ascontiguousarray(over_linear)),
            "over_to_under_ratio": torch.tensor(
                self.exposure_references[pair.scene].over_to_under_ratio, dtype=torch.float32
            ),
            "scene": pair.scene,
    }


def _crop_with_reflect_context(
    source: np.ndarray,
    top: int,
    left: int,
    patch_size: int,
    context_pad: int,
) -> np.ndarray:
    """Extract a fixed core tile plus reflection context without padding the full scene."""
    if source.ndim != 3 or source.shape[0] != 4:
        raise RawDataError("Expected packed cache array [4,H,W]")
    height, width = source.shape[-2:]
    y0, x0 = top - context_pad, left - context_pad
    y1, x1 = top + patch_size + context_pad, left + patch_size + context_pad
    clipped_y0, clipped_x0 = max(y0, 0), max(x0, 0)
    clipped_y1, clipped_x1 = min(y1, height), min(x1, width)
    tile = np.asarray(source[:, clipped_y0:clipped_y1, clipped_x0:clipped_x1])
    padding = (
        (0, 0),
        (clipped_y0 - y0, y1 - clipped_y1),
        (clipped_x0 - x0, x1 - clipped_x1),
    )
    if any(value for axis in padding for value in axis):
        tile = np.pad(tile, padding, mode="reflect")
    expected = (4, patch_size + 2 * context_pad, patch_size + 2 * context_pad)
    if tile.shape != expected:
        raise RawDataError("Context crop shape {} does not match {}".format(tile.shape, expected))
    return tile


def _read_exposure_references(cache_root: Path) -> dict[str, ExposureReference]:
    path = cache_root / "exposure_reference.json"
    if not path.is_file():
        raise RawDataError(
            "Missing {}. Re-run prepare_cache.py with --black-level and --white-level to create "
            "scene-level geometric-mean exposure references.".format(path)
        )
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
        references = {
            scene: ExposureReference.from_dict(value)
            for scene, value in content["scenes"].items()
        }
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RawDataError("Invalid exposure reference manifest {}: {}".format(path, error)) from error
    if not references:
        raise RawDataError("Exposure reference manifest contains no scenes: {}".format(path))
    return references


def worker_init_fn(worker_id: int) -> None:
    """Retain a deterministic worker seed for non-sampling NumPy operations."""
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed + worker_id)
