"""Task-routed inference for a FinalFusion checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import load_json
from .ivif.data import _pack_rggb, _read_ivif_pair
from .model import ARCHITECTURE_ID, TASKS, UnifiedFinalNet
from .rawprism import ARCHITECTURE_ID as RAWPRISM_ARCH, RawPrism
from .rawprism_v5 import ARCHITECTURE_ID as RAWPRISM_V5_ARCH, RawPrismV5
from .rawprism_v6 import ARCHITECTURE_ID as RAWPRISM_V6_ARCH, RawPrismV6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run IVIF, MEF, or MFF through one unified checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--input", required=True, help="IVIF scene dir, MEF scene name, or MFF registered NPZ.")
    parser.add_argument("--output", required=True, help="Output directory for IVIF/MEF or output NPZ for MFF.")
    parser.add_argument("--config", default=None, help="Training-compatible JSON for task metadata/settings.")
    parser.add_argument("--device", default="")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--halo", type=int, default=32)
    parser.add_argument("--metadata-root", default=None, help="MEF numeric metadata root for hard source chroma.")
    parser.add_argument("--extra-metadata-root", action="append", default=[], help="Additional MEF metadata root.")
    parser.add_argument("--no-hard-source-chroma", action="store_true", help="Write direct network MEF RAW4 instead.")
    return parser.parse_args()


def load_model(path: str, task: str, device: torch.device) -> tuple[UnifiedFinalNet, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    architecture = checkpoint["architecture_config"]
    tasks = tuple(architecture["tasks"])
    if task not in tasks:
        raise ValueError("Checkpoint does not contain task {!r}; available tasks are {}".format(task, tasks))
    width = int(checkpoint.get("config", {}).get("width", 44))
    arch = checkpoint.get("architecture")
    if arch == RAWPRISM_ARCH:
        backbone = str(architecture.get("backbone", "dense"))
        use_priors = bool(str(architecture.get("fusion", "")).startswith("physics-prior"))
        model = RawPrism(
            tasks=tasks,
            width=width,
            backbone=backbone,
            use_priors=use_priors,
            backbone_kwargs=checkpoint.get("config", {}).get("backbone_kwargs"),
        ).to(device)
    elif arch == RAWPRISM_V5_ARCH:
        model = RawPrismV5(
            tasks=tasks,
            use_semantic=bool(checkpoint.get("config", {}).get("use_semantic", False)),
            backbone_kwargs=checkpoint.get("config", {}).get("backbone_kwargs"),
        ).to(device)
    elif arch == RAWPRISM_V6_ARCH:
        model = RawPrismV6(
            tasks=tasks,
            use_semantic=bool(checkpoint.get("config", {}).get("use_semantic", False)),
            use_rft=bool(architecture.get("use_rft", True)),
            use_priors=bool(architecture.get("use_priors", True)),
            backbone_kwargs=checkpoint.get("config", {}).get("backbone_kwargs"),
            visible_residual=bool(checkpoint.get("config", {}).get("visible_residual",
                                  architecture.get("visible_residual", False))),
            visible_residual_scale=float(checkpoint.get("config", {}).get("visible_residual_scale",
                                        architecture.get("visible_residual_scale", 0.5))),
        ).to(device)
    elif arch == ARCHITECTURE_ID:
        model = UnifiedFinalNet(tasks=tasks, width=width).to(device)
    else:
        raise ValueError("Unknown checkpoint architecture {!r}".format(arch))
    model.load_state_dict(checkpoint["model"])
    return model.eval(), checkpoint


@torch.no_grad()
def tiled_forward(
    model: UnifiedFinalNet,
    task: str,
    source1: torch.Tensor,
    source2: torch.Tensor,
    metadata: dict[str, Any],
    tile_size: int,
    halo: int,
) -> torch.Tensor:
    if source1.shape != source2.shape or source1.ndim != 4 or source1.shape[:2] != (1, 4):
        raise ValueError("Tiled inference expects one matching RAW4 pair [1,4,H,W]")
    if tile_size <= 0 or halo < 12:
        raise ValueError("tile-size must be positive and halo must be at least 12 packed pixels")
    # Crop to a multiple of 8 so the U-Net down/up-sampling keeps its skip
    # connections aligned for every tile edge (otherwise the last tile can
    # differ by one pixel and the ``x + x2`` residual fails).
    height = source1.shape[-2] // 8 * 8
    width = source1.shape[-1] // 8 * 8
    source1 = source1[..., :height, :width]
    source2 = source2[..., :height, :width]
    fused = torch.empty_like(source1)
    for top in range(0, height, tile_size):
        bottom = min(top + tile_size, height)
        input_top, input_bottom = max(0, top - halo), min(height, bottom + halo)
        for left in range(0, width, tile_size):
            right = min(left + tile_size, width)
            input_left, input_right = max(0, left - halo), min(width, right + halo)
            first_tile = source1[..., input_top:input_bottom, input_left:input_right]
            second_tile = source2[..., input_top:input_bottom, input_left:input_right]
            tile = model(task, first_tile, second_tile, **metadata)["fused_raw4"]
            fused[..., top:bottom, left:right] = tile[
                ..., top - input_top:bottom - input_top, left - input_left:right - input_left
            ]
    return fused


def _ivif_input(scene: Path, config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    visible, infrared = _read_ivif_pair(
        scene, float(config.get("black_level", 16)), float(config.get("white_level", 4095))
    )
    visible_raw4, infrared_raw4 = _pack_rggb(visible), _pack_rggb(infrared)
    gains = infrared_raw4.mean(axis=(1, 2)) / (visible_raw4.mean(axis=(1, 2)) + 1.0e-6)
    visible_raw4 = np.clip(visible_raw4 * gains[:, None, None], 0.0, 1.0).astype(np.float32)
    return visible_raw4, infrared_raw4


def _unpack_rggb(raw4: np.ndarray) -> np.ndarray:
    height, width = raw4.shape[1:]
    mosaic = np.empty((height * 2, width * 2), dtype=np.float32)
    mosaic[0::2, 0::2], mosaic[0::2, 1::2] = raw4[0], raw4[1]
    mosaic[1::2, 0::2], mosaic[1::2, 1::2] = raw4[2], raw4[3]
    return mosaic


def infer_ivif(
    model: UnifiedFinalNet,
    scene: Path,
    output: Path,
    config: dict[str, Any],
    device: torch.device,
    tile_size: int,
    halo: int,
) -> None:
    first, second = _ivif_input(scene, config)
    first_tensor = torch.from_numpy(first).unsqueeze(0).to(device)
    second_tensor = torch.from_numpy(second).unsqueeze(0).to(device)
    fused = tiled_forward(model, "ivif", first_tensor, second_tensor, {}, tile_size, halo)[0].cpu().numpy()
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / (scene.name + "_fused_raw4.npy"), fused)
    np.save(output / (scene.name + "_fused_bayer.npy"), _unpack_rggb(fused))


def infer_mff(
    model: UnifiedFinalNet,
    pair_path: Path,
    output: Path,
    device: torch.device,
    tile_size: int,
    halo: int,
) -> None:
    from .mff.support.raw_dataset import canonical_plane_indices, validate_cfa_pattern

    with np.load(str(pair_path), allow_pickle=False) as archive:
        needed = {"reference_packed", "moving_registered_packed", "cfa_pattern"}
        if needed - set(archive.files):
            raise KeyError("MFF NPZ is missing {}".format(sorted(needed - set(archive.files))))
        payload = {key: archive[key] for key in archive.files}
    reference = np.asarray(payload["reference_packed"], dtype=np.float32)
    moving = np.asarray(payload["moving_registered_packed"], dtype=np.float32)
    pattern = validate_cfa_pattern(payload["cfa_pattern"], str(pair_path))
    indices = canonical_plane_indices(pattern)
    first = torch.from_numpy(np.clip(reference[list(indices)], 0.0, 1.0)).unsqueeze(0).to(device)
    second = torch.from_numpy(np.clip(moving[list(indices)], 0.0, 1.0)).unsqueeze(0).to(device)
    fused_canonical = tiled_forward(model, "mff", first, second, {}, tile_size, halo)[0].cpu().numpy()
    fused = np.empty_like(reference, dtype=np.float32)
    h, w = fused_canonical.shape[-2:]
    cropped = np.empty_like(reference[:, : h, : w], dtype=np.float32)
    cropped[list(indices)] = fused_canonical
    fused[:, : h, : w] = cropped
    if h < reference.shape[-2]:
        fused[:, h:, :w] = fused[:, max(h - 1, 0):max(h - 1, 0) + 1, :w]
    if w < reference.shape[-1]:
        fused[:, :, w:] = fused[:, :, max(w - 1, 0):max(w - 1, 0) + 1]
    payload["fused_packed"] = fused
    payload["finalfusion_architecture"] = np.asarray(ARCHITECTURE_ID)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(output), **payload)
    output.with_name(output.stem + "_metadata.json").write_text(
        json.dumps({
            "task": "mff",
            "input": str(pair_path.resolve()),
            "output": str(output.resolve()),
            "architecture": ARCHITECTURE_ID,
            "cfa_pattern": list(pattern),
            "tile_size": tile_size,
            "halo": halo,
        }, indent=2) + "\n",
        encoding="utf-8",
    )


def infer_mef(
    model: UnifiedFinalNet,
    scene_name: str,
    output: Path,
    config: dict[str, Any],
    device: torch.device,
    tile_size: int,
    halo: int,
    metadata_roots: list[str | Path] | None = None,
) -> None:
    from .mef.support.raw import discover_pairs

    pairs = {pair.scene: pair for pair in discover_pairs(config["data_root"])}
    if scene_name not in pairs:
        raise KeyError("MEF scene {!r} was not found".format(scene_name))
    _infer_mef_pair(model, pairs[scene_name], output, config, device, tile_size, halo, metadata_roots)


def _infer_mef_pair(
    model: UnifiedFinalNet,
    pair: Any,
    output: Path,
    config: dict[str, Any],
    device: torch.device,
    tile_size: int,
    halo: int,
    metadata_roots: list[str | Path] | None = None,
) -> None:
    """Infer one already-discovered MEF pair; used by single and batch entry points."""
    from .mef.support.exposure import estimate_exposure_reference
    from .mef.support.raw import (
        denormalize_raw,
        normalize_raw,
        pack_bayer,
        read_raw_mosaic,
        unpack_bayer,
        write_raw_tiff,
    )

    scene_name = pair.scene
    black, white = float(config["black_level"]), float(config["white_level"])
    under = normalize_raw(read_raw_mosaic(pair.under_path), black, white)
    over = normalize_raw(read_raw_mosaic(pair.over_path), black, white)
    under_packed = pack_bayer(under, pair.bayer_pattern)
    over_packed = pack_bayer(over, pair.bayer_pattern)
    reference = estimate_exposure_reference(under_packed, over_packed)
    first = torch.from_numpy(under_packed).unsqueeze(0).to(device)
    second = torch.from_numpy(over_packed).unsqueeze(0).to(device)
    metadata = {"exposure_ratio": torch.tensor([reference.over_to_under_ratio], device=device)}
    network = tiled_forward(model, "mef", first, second, metadata, tile_size, halo)[0].cpu().numpy()
    hard_chroma_report: dict[str, Any] | None = None
    if metadata_roots:
        from .mef.support.hard_source_chroma import (
            HardSourceChromaConfig, hard_source_chroma_correction, metadata_noise_profile, metadata_path,
        )

        config_hard = HardSourceChromaConfig()
        under_metadata = metadata_path(metadata_roots, scene_name, "under")
        over_metadata = metadata_path(metadata_roots, scene_name, "over")
        final, components = hard_source_chroma_correction(
            network, under_packed, over_packed, reference.under_to_reference, reference.over_to_reference,
            metadata_noise_profile(under_metadata), metadata_noise_profile(over_metadata), config_hard,
        )
        hard_chroma_report = {
            "mode": "network brightness plus hard complete-source chromaticity",
            "config": config_hard.to_dict(),
            "metadata_under": str(under_metadata.resolve()),
            "metadata_over": str(over_metadata.resolve()),
            "selected_over_fraction": float(components["selected_over"].mean()),
            "forced_under_from_saturated_over_fraction": float(components["forced_under_from_saturated_over"].mean()),
            "both_unreliable_fraction": float(components["both_unreliable"].mean()),
            "brightness_preservation_max_error": float(components["brightness_absolute_error"].max()),
        }
    else:
        final = network
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "fused_packed_float32.npy", final)
    mosaic = unpack_bayer(final, pair.bayer_pattern)
    tiff_name = "{}_fused_{}_16bit.tiff".format(scene_name, pair.bayer_pattern)
    write_raw_tiff(output / tiff_name, denormalize_raw(mosaic, black, white))
    report = {
        "task": "mef",
        "scene": scene_name,
        "architecture": ARCHITECTURE_ID,
        "under_input": str(Path(pair.under_path).resolve()),
        "over_input": str(Path(pair.over_path).resolve()),
        "bayer_pattern": pair.bayer_pattern,
        "black_level": black,
        "white_level": white,
        "exposure_reference": reference.to_dict(),
        "postprocess": hard_chroma_report or "none; fused_packed_float32.npy is the direct soft-clamped network RAW4 output",
        "tile_size": tile_size,
        "halo": halo,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model, checkpoint = load_model(args.checkpoint, args.task, device)
    config = load_json(args.config) if args.config else checkpoint.get("config", {})
    task_config = config.get("task_config", {}).get(args.task, config)
    input_path, output_path = Path(args.input), Path(args.output)
    if args.task == "ivif":
        infer_ivif(model, input_path, output_path, task_config, device, args.tile_size, args.halo)
    elif args.task == "mff":
        infer_mff(model, input_path, output_path, device, args.tile_size, args.halo)
    else:
        roots = [] if args.no_hard_source_chroma else [args.metadata_root or task_config.get("metadata_root"), *args.extra_metadata_root]
        if not args.no_hard_source_chroma and not roots[0]:
            raise ValueError("MEF hard source chroma requires --metadata-root or config metadata_root")
        infer_mef(model, args.input, output_path, task_config, device, args.tile_size, args.halo, roots)
    print("Completed {} inference with {}".format(args.task, output_path.resolve()))


if __name__ == "__main__":
    main()
