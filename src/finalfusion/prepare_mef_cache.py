#!/usr/bin/env python3
"""Create efficient packed-RAW uint16 caches without reading camera metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .mef.support.data import split_pairs_from_json
from .mef.support.exposure import estimate_exposure_reference
from .mef.support.noise import fit_noise_model, write_noise_model
from .mef.support.raw import discover_pairs, normalize_raw, pack_bayer, read_raw_mosaic


PACKAGED_SPLIT_JSON = Path(__file__).with_name("mef_scene_split.json").resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="scene_xxxx/under|over RAW TIFF directory")
    parser.add_argument("--cache-root", required=True, help="Destination for packed uint16 .npy files")
    parser.add_argument("--black-level", required=True, type=float)
    parser.add_argument("--white-level", required=True, type=float)
    parser.add_argument("--exposure-min-luminance", type=float, default=0.02)
    parser.add_argument("--exposure-max-luminance", type=float, default=0.95)
    parser.add_argument("--exposure-trim-fraction", type=float, default=0.02)
    parser.add_argument("--exposure-min-valid-pixels", type=int, default=1024)
    parser.add_argument("--noise-seed", type=int, default=1337)
    parser.add_argument("--noise-samples-per-scene", type=int, default=2048)
    parser.add_argument("--noise-flat-fraction", type=float, default=0.25)
    parser.add_argument("--noise-bins", type=int, default=24)
    parser.add_argument("--skip-noise-fit", action="store_true",
                        help="Skip noise_model.json generation; train.py will reject the cache")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing cache files")
    parser.add_argument("--check-only", action="store_true",
                        help="Validate cache coverage and split provenance without writing files")
    return parser.parse_args()


def training_split_fingerprint(scene_names: list[str]) -> str:
    payload = "\n".join(sorted(scene_names)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_cache(
    pairs,
    cache_root: Path,
    black_level: float,
    white_level: float,
) -> list[str]:
    """Return lightweight cache errors without reading full packed arrays."""
    errors: list[str] = []
    pair_names = {pair.scene for pair in pairs}
    split = split_pairs_from_json(pairs, PACKAGED_SPLIT_JSON)
    train_names = [pair.scene for pair in split["train"]]
    expected_fingerprint = training_split_fingerprint(train_names)

    try:
        manifest = json.loads((cache_root / "manifest.json").read_text(encoding="utf-8"))
        cached_names = {record["scene"] for record in manifest["pairs"]}
        noise = manifest.get("noise_model", {})
        if cached_names != pair_names:
            errors.append("manifest scene coverage differs from data-root")
        if noise.get("training_scene_fingerprint") != expected_fingerprint:
            errors.append("noise model was not fitted on the current training split")
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        errors.append("invalid manifest.json: {}".format(error))

    try:
        references = json.loads((cache_root / "exposure_reference.json").read_text(encoding="utf-8"))
        if float(references["black_level"]) != float(black_level):
            errors.append("cached black level differs from requested value")
        if float(references["white_level"]) != float(white_level):
            errors.append("cached white level differs from requested value")
        if set(references["scenes"]) != pair_names:
            errors.append("exposure-reference scene coverage differs from data-root")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        errors.append("invalid exposure_reference.json: {}".format(error))

    if not (cache_root / "noise_model.json").is_file():
        errors.append("noise_model.json is missing")
    for pair in pairs:
        expected_shape = (4, pair.packed_height, pair.packed_width)
        for exposure in ("under", "over"):
            path = cache_root / pair.scene / (exposure + ".npy")
            try:
                packed = np.load(path, mmap_mode="r", allow_pickle=False)
                if packed.shape != expected_shape or packed.dtype != np.uint16:
                    errors.append("invalid packed cache: {}".format(path))
            except (OSError, ValueError) as error:
                errors.append("unreadable packed cache {}: {}".format(path, error))
    return errors


def load_cached_packed(path: Path, expected_shape: tuple[int, int, int]) -> np.ndarray:
    """Load and validate a complete cache file without filesystem mmap."""
    packed = np.load(path, allow_pickle=False)
    if not isinstance(packed, np.ndarray) or packed.shape != expected_shape or packed.dtype != np.uint16:
        raise ValueError(
            "invalid packed cache {}: expected uint16 {}, got {} {}".format(
                path, expected_shape, getattr(packed, "dtype", None), getattr(packed, "shape", None)
            )
        )
    return packed


def save_packed_atomic(path: Path, packed: np.ndarray) -> None:
    """Replace a cache only after the complete NPY has reached disk."""
    temporary = path.with_name(".{}.{}.tmp.npy".format(path.stem, os.getpid()))
    try:
        np.save(temporary, packed, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    args = parse_args()
    if not args.white_level > args.black_level:
        raise ValueError("white-level must be greater than black-level")
    pairs = discover_pairs(args.data_root)
    cache_root = Path(args.cache_root).expanduser().resolve()
    if args.check_only:
        errors = validate_cache(pairs, cache_root, args.black_level, args.white_level)
        if errors:
            print("MEF cache refresh required:")
            for error in errors[:20]:
                print("- " + error)
            if len(errors) > 20:
                print("- ... {} additional errors".format(len(errors) - 20))
            raise SystemExit(1)
        print("MEF cache is complete for {} scenes and the current split".format(len(pairs)))
        return
    cache_root.mkdir(parents=True, exist_ok=True)
    exposure_references = {}
    for pair in tqdm(pairs, desc="Packing RAW scenes"):
        scene_root = cache_root / pair.scene
        scene_root.mkdir(parents=True, exist_ok=True)
        packed_images = {}
        expected_shape = (4, pair.packed_height, pair.packed_width)
        for exposure, source in (("under", pair.under_path), ("over", pair.over_path)):
            target = scene_root / (exposure + ".npy")
            packed = None
            if target.exists() and not args.overwrite:
                try:
                    packed = load_cached_packed(target, expected_shape)
                except (OSError, ValueError) as error:
                    tqdm.write("Rebuilding unusable cache {}: {}".format(target, error))
            if packed is None:
                packed = pack_bayer(read_raw_mosaic(source), pair.bayer_pattern)
                if packed.dtype != np.uint16:
                    packed = packed.astype(np.uint16, copy=False)
                if packed.shape != expected_shape:
                    raise ValueError("Packed source {} has unexpected shape {}".format(source, packed.shape))
                save_packed_atomic(target, packed)
            packed_images[exposure] = packed
        scale = args.white_level - args.black_level
        under_linear = np.clip(
            (np.asarray(packed_images["under"], dtype=np.float32) - args.black_level) / scale, 0.0, 1.0
        )
        over_linear = np.clip(
            (np.asarray(packed_images["over"], dtype=np.float32) - args.black_level) / scale, 0.0, 1.0
        )
        reference = estimate_exposure_reference(
            under_linear,
            over_linear,
            min_luminance=args.exposure_min_luminance,
            max_luminance=args.exposure_max_luminance,
            trim_fraction=args.exposure_trim_fraction,
            min_valid_pixels=args.exposure_min_valid_pixels,
        )
        exposure_references[pair.scene] = reference.to_dict()
        if reference.estimation_mode != "strict_overlap":
            tqdm.write(
                "Warning: {} used relaxed exposure overlap [{:.4f}, {:.4f}] "
                "after strict overlap was empty or insufficient.".format(
                    pair.scene, reference.min_luminance, reference.max_luminance
                )
            )
    manifest = {
        "schema_version": 2,
        "source_root": str(Path(args.data_root).expanduser().resolve()),
        "cache_format": "canonical [R, Gr, B, Gb] uint16 arrays",
        "pairs": [pair.to_dict() for pair in pairs],
    }
    (cache_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    reference_manifest = {
        "schema_version": 1,
        "reference_domain": "per-scene geometric mean of under and over exposure",
        "black_level": args.black_level,
        "white_level": args.white_level,
        "scenes": exposure_references,
    }
    (cache_root / "exposure_reference.json").write_text(
        json.dumps(reference_manifest, indent=2) + "\n", encoding="utf-8"
    )
    if not args.skip_noise_fit:
        split = split_pairs_from_json(pairs, PACKAGED_SPLIT_JSON)
        split_source = str(PACKAGED_SPLIT_JSON)
        noise_model = fit_noise_model(
            split["train"], cache_root, args.black_level, args.white_level,
            samples_per_scene=args.noise_samples_per_scene,
            flat_fraction=args.noise_flat_fraction,
            bins=args.noise_bins,
            seed=args.noise_seed,
        )
        write_noise_model(cache_root, noise_model)
        manifest["noise_model"] = {
            "path": "noise_model.json",
            "fit_split": "train",
            "split_source": split_source,
            "training_scenes": len(split["train"]),
            "training_scene_fingerprint": training_split_fingerprint(
                [pair.scene for pair in split["train"]]
            ),
        }
        (cache_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print("Fitted four-channel noise model using {} training scenes".format(len(split["train"])))
    print("Prepared {} paired scenes at {}".format(len(pairs), cache_root))


if __name__ == "__main__":
    main()
