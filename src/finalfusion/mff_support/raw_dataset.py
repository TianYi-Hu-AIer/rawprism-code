"""RawLab MFF scene split and NPZ loader, retaining data rules for RAW4 batches."""

from __future__ import absolute_import, division, print_function

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np


PAIR_FILENAME = "registered_raw_pair.npz"
METADATA_FILENAME = "registration_metadata.json"
CANONICAL_CFA_PLANE_LABELS = ("R", "G1", "G2", "B")


def validate_cfa_pattern(cfa_pattern, context):
    values = np.asarray(cfa_pattern).reshape(-1)
    if values.size != 4 or not np.all(np.isfinite(values)) or not np.all(values == np.floor(values)):
        raise ValueError("{}: cfa_pattern must contain four finite integer values".format(context))
    pattern = tuple(int(value) for value in values)
    if sorted(pattern) != [0, 1, 1, 2]:
        raise ValueError("{}: unsupported CFA pattern {}".format(context, pattern))
    return pattern


def canonical_plane_indices(cfa_pattern):
    """Canonical R/G1/G2/B to stored packed-plane indices, with no interpolation."""
    pattern = validate_cfa_pattern(cfa_pattern, "canonical_plane_indices")
    green = tuple(index for index, color in enumerate(pattern) if color == 1)
    return (pattern.index(0), green[0], green[1], pattern.index(2))


def cfa_plane_labels(cfa_pattern):
    pattern = validate_cfa_pattern(cfa_pattern, "cfa_plane_labels")
    labels, green = [], 0
    for color in pattern:
        if color == 0:
            labels.append("R")
        elif color == 2:
            labels.append("B")
        else:
            green += 1
            labels.append("G{}".format(green))
    return tuple(labels)


def discover_scenes(data_root, quality_policy="all"):
    root = Path(data_root)
    if quality_policy not in ("all", "accepted"):
        raise ValueError("quality_policy must be all or accepted")
    scenes = []
    for scene_dir in sorted(root.glob("scene_*")):
        pair_path = scene_dir / PAIR_FILENAME
        if not pair_path.is_file():
            continue
        metadata_path = scene_dir / METADATA_FILENAME
        metadata = {}
        if metadata_path.is_file():
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        quality = metadata.get("registration_quality", {})
        accepted = bool(quality.get("accepted_for_dataset", True))
        if quality_policy == "accepted" and not accepted:
            continue
        scenes.append({"scene": scene_dir.name, "accepted_for_dataset": accepted, "quality_status": quality.get("status", "unknown"), "recommended_action": quality.get("recommended_action", "unknown")})
    if not scenes:
        raise FileNotFoundError("No registered_raw_pair.npz files were found under {}".format(root))
    return scenes


def build_scene_split(scenes, seed):
    if len(scenes) < 10:
        raise ValueError("At least ten scenes are required for 80/10/10 split")
    indices = np.arange(len(scenes))
    np.random.RandomState(int(seed)).shuffle(indices)
    train_count, validation_count = int(len(scenes) * 0.80), int(len(scenes) * 0.10)
    names = [record["scene"] for record in scenes]
    return {
        "format": "finalfusion_mff_scene_split_v1",
        "seed": int(seed),
        "counts": {"total": len(scenes), "train": train_count, "validation": validation_count, "test": len(scenes) - train_count - validation_count},
        "splits": {"train": [names[index] for index in indices[:train_count]], "validation": [names[index] for index in indices[train_count:train_count + validation_count]], "test": [names[index] for index in indices[train_count + validation_count:]]},
        "scene_quality": {record["scene"]: record for record in scenes},
    }


def load_or_create_scene_split(data_root, split_path, seed, quality_policy="all"):
    split_path = Path(split_path)
    if split_path.is_file():
        with split_path.open("r", encoding="utf-8") as handle:
            split = json.load(handle)
        if set(split.get("splits", {})) != {"train", "validation", "test"}:
            raise ValueError("Existing split must contain train/validation/test lists")
        if split.get("quality_policy") not in (None, quality_policy):
            raise ValueError("Existing split has incompatible quality_policy")
        return split, False
    split = build_scene_split(discover_scenes(data_root, quality_policy), seed)
    split["data_root_at_creation"], split["quality_policy"] = str(Path(data_root).resolve()), quality_policy
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with split_path.open("w", encoding="utf-8") as handle:
        json.dump(split, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    return split, True


class RawPairStore(object):
    def __init__(self, data_root, cache_scenes=0):
        self.data_root, self.cache_scenes, self.cache = Path(data_root), int(cache_scenes), OrderedDict()
        if self.cache_scenes < 0:
            raise ValueError("cache-scenes must be zero (all) or positive")

    def get(self, scene_name):
        if scene_name in self.cache:
            self.cache.move_to_end(scene_name)
            return self.cache[scene_name]
        path = self.data_root / scene_name / PAIR_FILENAME
        with np.load(str(path), allow_pickle=False) as archive:
            required = {"reference_packed", "moving_registered_packed", "moving_valid_mask", "cfa_pattern"}
            missing = required.difference(archive.files)
            if missing:
                raise KeyError("{} is missing keys {}".format(path, sorted(missing)))
            reference = np.asarray(archive["reference_packed"], dtype=np.float32).copy()
            moving = np.asarray(archive["moving_registered_packed"], dtype=np.float32).copy()
            valid = np.asarray(archive["moving_valid_mask"], dtype=np.float32).copy()
            pattern = validate_cfa_pattern(np.asarray(archive["cfa_pattern"]).copy(), scene_name)
        if reference.ndim != 3 or reference.shape[0] != 4 or moving.shape != reference.shape or valid.shape != reference.shape[1:]:
            raise ValueError("{} has inconsistent packed RAW pair shapes".format(scene_name))
        if not np.isfinite(reference).all() or not np.isfinite(moving).all():
            raise ValueError("{} contains non-finite RAW values".format(scene_name))
        pair = (reference, moving, valid, pattern)
        self.cache[scene_name] = pair
        if self.cache_scenes > 0:
            while len(self.cache) > self.cache_scenes:
                self.cache.popitem(last=False)
        return pair

    def validate_scene_patterns(self, scene_names):
        patterns = {}
        for scene_name in scene_names:
            with np.load(str(self.data_root / scene_name / PAIR_FILENAME), allow_pickle=False) as archive:
                patterns[scene_name] = validate_cfa_pattern(archive["cfa_pattern"], scene_name)
        return patterns


def _sample_valid_crop(valid_mask, patch_size, rng, all_valid=False):
    height, width = valid_mask.shape
    if height < patch_size or width < patch_size:
        raise ValueError("Packed RAW {}x{} smaller than patch {}".format(height, width, patch_size))
    max_y, max_x = height - patch_size, width - patch_size
    if all_valid:
        return int(rng.randint(max_y + 1)), int(rng.randint(max_x + 1))
    for _ in range(64):
        y, x = int(rng.randint(max_y + 1)), int(rng.randint(max_x + 1))
        if np.all(valid_mask[y:y + patch_size, x:x + patch_size] >= 0.999):
            return y, x
    raise RuntimeError("Could not sample a fully valid RAW patch")


class Raw4PatchBatcher(object):
    """MFF's scene/grouped valid-patch sampling, emitting full canonical RAW4 pairs."""

    _PHASE_OFFSETS = {"train": 0, "validation": 1000003, "test": 2000003}

    def __init__(self, store, scene_names, patch_size=512, patches_per_scene=4, seed=20260728, phase="train"):
        if phase not in self._PHASE_OFFSETS or not scene_names:
            raise ValueError("Unknown phase or empty scene list")
        self.store, self.scene_names = store, list(scene_names)
        self.patch_size, self.patches_per_scene, self.seed, self.phase = int(patch_size), int(patches_per_scene), int(seed), phase
        if self.patch_size <= 0 or self.patches_per_scene <= 0:
            raise ValueError("patch_size and patches_per_scene must be positive")

    @property
    def samples_per_epoch(self):
        return len(self.scene_names) * self.patches_per_scene

    def iter_batches(self, epoch, batch_size, training):
        rng = np.random.RandomState(self.seed + self._PHASE_OFFSETS[self.phase] + 7919 * (int(epoch) if training else 0))
        scenes = list(self.scene_names)
        if training:
            rng.shuffle(scenes)
        first_batch, second_batch = [], []
        for scene_name in scenes:
            reference, moving, valid, pattern = self.store.get(scene_name)
            indices = canonical_plane_indices(pattern)
            reference, moving = reference[list(indices)], moving[list(indices)]
            all_valid = bool(np.all(valid >= 0.999))
            for _ in range(self.patches_per_scene):
                y, x = _sample_valid_crop(valid, self.patch_size, rng, all_valid)
                first = np.clip(reference[:, y:y + self.patch_size, x:x + self.patch_size], 0.0, 1.0)
                second = np.clip(moving[:, y:y + self.patch_size, x:x + self.patch_size], 0.0, 1.0)
                if training and rng.rand() < 0.5:
                    first, second = second, first
                first_batch.append(np.transpose(first, (1, 2, 0)))
                second_batch.append(np.transpose(second, (1, 2, 0)))
                if len(first_batch) == batch_size:
                    yield np.asarray(first_batch, np.float32), np.asarray(second_batch, np.float32)
                    first_batch, second_batch = [], []
        if first_batch:
            yield np.asarray(first_batch, np.float32), np.asarray(second_batch, np.float32)
