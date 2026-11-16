"""Dataset adapters that normalize three existing RAW tasks for joint training."""

from __future__ import annotations

import json
import multiprocessing as mp
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % 2 ** 32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _pack_rggb(image: np.ndarray) -> np.ndarray:
    height, width = image.shape
    image = image[:height // 2 * 2, :width // 2 * 2]
    return np.stack((image[0::2, 0::2], image[0::2, 1::2], image[1::2, 0::2], image[1::2, 1::2])).astype(
        np.float32
    )


def _read_ivif_pair(scene: Path, black_level: float, white_level: float) -> tuple[np.ndarray, np.ndarray]:
    import cv2
    import tifffile

    visible = np.asarray(tifffile.imread(str(scene / "vis_raw.dng")), dtype=np.float32)
    if visible.ndim != 2:
        raise ValueError("{} visible DNG is not a single Bayer plane".format(scene))
    visible = np.clip((visible - black_level) / (white_level - black_level), 0.0, 1.0)
    infrared = cv2.imread(str(scene / "ir.png"), cv2.IMREAD_UNCHANGED)
    if infrared is None:
        raise ValueError("Failed to read {}".format(scene / "ir.png"))
    if infrared.ndim == 3:
        infrared = cv2.cvtColor(infrared, cv2.COLOR_BGR2GRAY)
    if infrared.dtype == np.uint16:
        divisor = 65535.0
    elif infrared.dtype == np.uint8:
        divisor = 255.0
    elif np.issubdtype(infrared.dtype, np.integer):
        divisor = float(np.iinfo(infrared.dtype).max)
    else:
        divisor = 1.0
    infrared = np.clip(infrared.astype(np.float32) / divisor, 0.0, 1.0)
    height = min(visible.shape[0], infrared.shape[0]) // 2 * 2
    width = min(visible.shape[1], infrared.shape[1]) // 2 * 2
    if min(height, width) < 2:
        raise ValueError("{} has no common even RAW area".format(scene))
    return visible[:height, :width], infrared[:height, :width]


class IvifDataset(Dataset):
    def __init__(
        self,
        scenes: list[Path],
        patch_size_bayer: int | None,
        training: bool,
        black_level: float,
        white_level: float,
    ) -> None:
        self.scenes = list(scenes)
        self.patch_size_bayer = patch_size_bayer
        self.training = bool(training)
        self.black_level = float(black_level)
        self.white_level = float(white_level)
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
            crop_h = min(visible.shape[0], self.patch_size_bayer) // 2 * 2
            crop_w = min(visible.shape[1], self.patch_size_bayer) // 2 * 2
            max_y, max_x = visible.shape[0] - crop_h, visible.shape[1] - crop_w
            top = random.randrange(max_y // 2 + 1) * 2 if max_y else 0
            left = random.randrange(max_x // 2 + 1) * 2 if max_x else 0
            visible = visible[top:top + crop_h, left:left + crop_w]
            infrared = infrared[top:top + crop_h, left:left + crop_w]
        visible_raw4, infrared_raw4 = _pack_rggb(visible), _pack_rggb(infrared)
        visible_raw4 = np.clip(visible_raw4 * gains[:, None, None], 0.0, 1.0).astype(np.float32)
        return {
            "source1": torch.from_numpy(visible_raw4),
            "source2": torch.from_numpy(infrared_raw4),
            "scene": scene.name,
        }


class MffDataset(Dataset):
    _PHASE_OFFSET = {"train": 0, "validation": 1_000_003}

    def __init__(
        self,
        store: Any,
        scene_names: list[str],
        patch_size: int,
        patches_per_scene: int,
        seed: int,
        phase: str,
    ) -> None:
        if phase not in self._PHASE_OFFSET or not scene_names:
            raise ValueError("MFF phase must be train/validation with non-empty scenes")
        self.store = store
        self.scene_names = list(scene_names)
        self.patch_size = int(patch_size)
        self.patches_per_scene = int(patches_per_scene)
        self.seed = int(seed)
        self.phase = phase
        self._epoch = mp.Value("q", 0)

    def set_epoch(self, epoch: int) -> None:
        with self._epoch.get_lock():
            self._epoch.value = int(epoch)

    def __len__(self) -> int:
        return len(self.scene_names) * self.patches_per_scene

    def __getitem__(self, index: int) -> dict[str, Any]:
        from .mff.support.raw_dataset import canonical_plane_indices

        scene_index = index % len(self.scene_names)
        scene = self.scene_names[scene_index]
        reference, moving, valid, pattern = self.store.get(scene)
        indices = canonical_plane_indices(pattern)
        reference, moving = reference[list(indices)], moving[list(indices)]
        height, width = valid.shape
        if min(height, width) < self.patch_size:
            raise ValueError("MFF scene {} is smaller than patch size".format(scene))
        rng = np.random.RandomState(
            self.seed + self._PHASE_OFFSET[self.phase] + 7919 * int(self._epoch.value) + 104729 * index
        )
        max_y, max_x = height - self.patch_size, width - self.patch_size
        top = left = 0
        for _ in range(64):
            top = int(rng.randint(max_y + 1))
            left = int(rng.randint(max_x + 1))
            if np.all(valid[top:top + self.patch_size, left:left + self.patch_size] >= 0.999):
                break
        else:
            raise RuntimeError("Could not find a fully valid MFF crop for {}".format(scene))
        first = np.clip(reference[:, top:top + self.patch_size, left:left + self.patch_size], 0.0, 1.0)
        second = np.clip(moving[:, top:top + self.patch_size, left:left + self.patch_size], 0.0, 1.0)
        if self.phase == "train" and rng.rand() < 0.5:
            first, second = second, first
        return {
            "source1": torch.from_numpy(np.ascontiguousarray(first, dtype=np.float32)),
            "source2": torch.from_numpy(np.ascontiguousarray(second, dtype=np.float32)),
            "scene": scene,
        }


def _loader(dataset: Dataset, config: dict[str, Any], seed: int, training: bool) -> DataLoader:
    workers = int(config.get("num_workers", 0))
    return DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 1)),
        shuffle=training,
        num_workers=workers,
        pin_memory=bool(config.get("pin_memory", torch.cuda.is_available())),
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )


def _ivif_loaders(config: dict[str, Any], seed: int) -> tuple[DataLoader, DataLoader]:
    root = Path(config["dataset_root"])
    scene_root = root / "train"
    scenes = sorted(
        scene for scene in scene_root.iterdir()
        if scene.is_dir() and (scene / "vis_raw.dng").is_file() and (scene / "ir.png").is_file()
    )
    if len(scenes) < 2:
        raise FileNotFoundError("IVIF train split needs at least two vis_raw.dng/ir.png scenes")
    manifest = Path(config.get("validation_manifest", PROJECT_ROOT / "finalfusion_runs" / "ivif_validation.txt"))
    val_count = min(int(config.get("validation_count", 80)), len(scenes) - 1)
    if manifest.is_file():
        validation_names = {line.strip() for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()}
        if len(validation_names) != val_count or validation_names - {scene.name for scene in scenes}:
            raise ValueError("IVIF validation manifest is incompatible with the dataset")
    else:
        validation_names = {scene.name for scene in random.Random(seed).sample(scenes, val_count)}
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("\n".join(sorted(validation_names)) + "\n", encoding="utf-8")
    train_scenes = [scene for scene in scenes if scene.name not in validation_names]
    val_scenes = [scene for scene in scenes if scene.name in validation_names]
    common = (float(config.get("black_level", 16)), float(config.get("white_level", 4095)))
    train = IvifDataset(train_scenes, int(config.get("patch_size_bayer", 512)), True, *common)
    validation = IvifDataset(val_scenes, None, False, *common)
    return _loader(train, config, seed, True), _loader(validation, {**config, "batch_size": 1}, seed, False)


def _mef_loaders(config: dict[str, Any], seed: int) -> tuple[DataLoader, DataLoader]:
    from .mef.support.data import PackedRawPairDataset, split_pairs_from_json
    from .mef.support.raw import read_pair_manifest

    split_json = Path(config.get("split_json", Path(__file__).with_name("mef_scene_split.json")))
    manifest = Path(config["cache_root"]) / "manifest.json"
    pairs = read_pair_manifest(manifest)
    split = split_pairs_from_json(pairs, split_json)
    arguments = {
        "cache_root": config["cache_root"],
        "black_level": float(config["black_level"]),
        "white_level": float(config["white_level"]),
        "patch_size": int(config.get("patch_size", 128)),
        "tile_stride": int(config.get("tile_stride", 64)),
        "sampling_seed": seed,
        "context_pad": int(config.get("context_pad", 0)),
    }
    train = PackedRawPairDataset(
        split["train"], patches_per_scene=int(config.get("patches_per_scene", 4)), training=True, **arguments
    )
    validation = PackedRawPairDataset(split["val"], patches_per_scene=1, training=False, **arguments)
    return _loader(train, config, seed, True), _loader(validation, config, seed, False)


def _mff_loaders(config: dict[str, Any], seed: int) -> tuple[DataLoader, DataLoader]:
    from .mff.support.raw_dataset import RawPairStore, load_or_create_scene_split

    split_path = Path(config.get("split_json", PROJECT_ROOT / "finalfusion_runs" / "mff_scene_split.json"))
    split, _ = load_or_create_scene_split(
        config["data_root"], split_path, seed, config.get("quality_policy", "all")
    )
    store = RawPairStore(config["data_root"], int(config.get("cache_scenes", 0)))
    patch_size = int(config.get("patch_size", 512))
    train = MffDataset(
        store, split["splits"]["train"], patch_size, int(config.get("patches_per_scene", 16)), seed, "train"
    )
    validation = MffDataset(
        store,
        split["splits"]["validation"],
        patch_size,
        int(config.get("validation_patches_per_scene", 4)),
        seed,
        "validation",
    )
    return _loader(train, config, seed, True), _loader(validation, config, seed, False)


def build_task_loaders(task: str, config: dict[str, Any], seed: int) -> tuple[DataLoader, DataLoader]:
    builders = {"ivif": _ivif_loaders, "mef": _mef_loaders, "mff": _mff_loaders}
    try:
        return builders[task](config, seed)
    except KeyError as error:
        if task not in builders:
            raise ValueError("Unknown task {!r}".format(task)) from error
        raise


def standardize_batch(task: str, batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, ...]:
    if task == "mef":
        source1 = batch["under"].float().to(device, non_blocking=True)
        source2 = batch["over"].float().to(device, non_blocking=True)
        metadata = {"exposure_ratio": batch["over_to_under_ratio"].float().to(device, non_blocking=True)}
    else:
        source1 = batch["source1"].float().to(device, non_blocking=True)
        source2 = batch["source2"].float().to(device, non_blocking=True)
        metadata = {}
    return source1, source2, metadata


def set_loader_epoch(loader: DataLoader, epoch: int) -> None:
    setter = getattr(loader.dataset, "set_epoch", None)
    if setter is not None:
        setter(epoch)


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
