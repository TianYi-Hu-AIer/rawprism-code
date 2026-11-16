"""Apply U2Fusion-style hard source chroma to existing FinalFusion MEF RAW."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .data import load_json
from .mef.support.data import split_pairs_from_json
from .mef.support.exposure import estimate_exposure_reference
from .mef.support.hard_source_chroma import (
    HardSourceChromaConfig, hard_source_chroma_correction, metadata_noise_profile, metadata_path,
)
from .mef.support.raw import (
    denormalize_raw, discover_pairs, normalize_raw, pack_bayer, read_raw_mosaic, unpack_bayer, write_raw_tiff,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, help="Existing direct-network FinalFusion MEF output root.")
    parser.add_argument("--output-root", required=True, help="New corrected output root; never use input-root.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--metadata-root", required=True)
    parser.add_argument("--extra-metadata-root", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root, output_root = Path(args.input_root), Path(args.output_root)
    if input_root.resolve() == output_root.resolve():
        raise ValueError("--output-root must differ from --input-root to preserve A/B comparison")
    config = load_json(args.config)
    task = config.get("task_config", {}).get("mef", config)
    black, white = float(task["black_level"]), float(task["white_level"])
    pairs = split_pairs_from_json(discover_pairs(task["data_root"]), args.split_json)[args.split]
    roots = [args.metadata_root, *args.extra_metadata_root]
    hard = HardSourceChromaConfig()
    completed, skipped = [], []
    for pair in tqdm(pairs, desc="Hard-source chroma"):
        source = input_root / pair.scene / "fused_packed_float32.npy"
        destination = output_root / pair.scene
        packed_path, report_path = destination / "fused_packed_float32.npy", destination / "report.json"
        if not source.is_file():
            raise FileNotFoundError("Missing direct network output: {}".format(source))
        if not args.force and packed_path.is_file() and report_path.is_file():
            skipped.append(pair.scene)
            continue
        under = pack_bayer(normalize_raw(read_raw_mosaic(pair.under_path), black, white), pair.bayer_pattern)
        over = pack_bayer(normalize_raw(read_raw_mosaic(pair.over_path), black, white), pair.bayer_pattern)
        reference = estimate_exposure_reference(under, over)
        under_meta, over_meta = metadata_path(roots, pair.scene, "under"), metadata_path(roots, pair.scene, "over")
        corrected, components = hard_source_chroma_correction(
            np.load(source, allow_pickle=False), under, over, reference.under_to_reference,
            reference.over_to_reference, metadata_noise_profile(under_meta), metadata_noise_profile(over_meta), hard,
        )
        destination.mkdir(parents=True, exist_ok=True)
        np.save(packed_path, corrected)
        write_raw_tiff(destination / "{}_fused_{}_16bit.tiff".format(pair.scene, pair.bayer_pattern), denormalize_raw(unpack_bayer(corrected, pair.bayer_pattern), black, white))
        report_path.write_text(json.dumps({
            "scene": pair.scene,
            "input_network_raw": str(source.resolve()),
            "postprocess": "hard complete-source chromaticity with network brightness",
            "config": hard.to_dict(),
            "metadata_under": str(under_meta.resolve()), "metadata_over": str(over_meta.resolve()),
            "selected_over_fraction": float(components["selected_over"].mean()),
            "forced_under_from_saturated_over_fraction": float(components["forced_under_from_saturated_over"].mean()),
            "both_unreliable_fraction": float(components["both_unreliable"].mean()),
            "brightness_preservation_max_error": float(components["brightness_absolute_error"].max()),
        }, indent=2) + "\n", encoding="utf-8")
        completed.append(pair.scene)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "hard_source_chroma_manifest.json").write_text(json.dumps({"completed": completed, "skipped": skipped}, indent=2) + "\n", encoding="utf-8")
    print("Finished: completed={}, skipped={}".format(len(completed), len(skipped)))


if __name__ == "__main__":
    main()
