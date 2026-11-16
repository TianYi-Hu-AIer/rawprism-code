"""Batch MEF inference with one checkpoint/model load."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .data import load_json
from .infer import _infer_mef_pair, load_model
from .mef.support.raw import discover_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--split-json",
        default=str(Path(__file__).with_name("mef_scene_split.json")),
    )
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
    parser.add_argument("--scenes", nargs="*", default=None, help="Explicit scene names; overrides --split.")
    parser.add_argument("--device", default="")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--halo", type=int, default=32)
    parser.add_argument("--metadata-root", default=None, help="Numeric metadata root for hard source chroma.")
    parser.add_argument("--extra-metadata-root", action="append", default=[], help="Additional metadata root.")
    parser.add_argument("--no-hard-source-chroma", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def selected_scenes(args: argparse.Namespace, available: dict[str, Any]) -> list[str]:
    if args.scenes:
        names = list(dict.fromkeys(args.scenes))
    elif args.split == "all":
        names = sorted(available)
    else:
        payload = load_json(args.split_json)
        names = list(payload["splits"][args.split])
    missing = [name for name in names if name not in available]
    if missing:
        raise KeyError("Selected MEF scenes were not found: {}".format(", ".join(missing)))
    return names


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model, checkpoint = load_model(args.checkpoint, "mef", device)
    config = load_json(args.config) if args.config else checkpoint.get("config", {})
    task_config = config.get("task_config", {}).get("mef", config)
    metadata_roots = [] if args.no_hard_source_chroma else [args.metadata_root or task_config.get("metadata_root"), *args.extra_metadata_root]
    if not args.no_hard_source_chroma and not metadata_roots[0]:
        raise ValueError("MEF hard source chroma requires --metadata-root or config metadata_root")
    pairs = {pair.scene: pair for pair in discover_pairs(task_config["data_root"])}
    scenes = selected_scenes(args, pairs)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "split": "explicit" if args.scenes else args.split,
        "completed": [],
        "skipped": [],
        "failed": {},
    }
    for index, scene_name in enumerate(scenes, start=1):
        output_dir = output_root / scene_name
        fused_path = output_dir / "fused_packed_float32.npy"
        report_path = output_dir / "report.json"
        if not args.force and fused_path.is_file() and report_path.is_file():
            manifest["skipped"].append(scene_name)
            print("[{}/{}] {}: skipped".format(index, len(scenes), scene_name), flush=True)
            continue
        try:
            _infer_mef_pair(
                model, pairs[scene_name], output_dir, task_config,
                device, args.tile_size, args.halo, metadata_roots,
            )
            manifest["completed"].append(scene_name)
            print("[{}/{}] {}: completed".format(index, len(scenes), scene_name), flush=True)
        except Exception as error:
            manifest["failed"][scene_name] = "{}: {}".format(type(error).__name__, error)
            print("[{}/{}] {}: FAILED: {}".format(index, len(scenes), scene_name, error), flush=True)
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()
    manifest_path = output_root / "batch_inference_report.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        "Finished: completed={}, skipped={}, failed={}".format(
            len(manifest["completed"]), len(manifest["skipped"]), len(manifest["failed"])
        ),
        flush=True,
    )
    if manifest["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
