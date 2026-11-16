"""Run one RAWPrism checkpoint over the declared test splits.

For every scene it stores
  <out>/<task>/<scene>.png          the protocol renderer (declared: libraw template
                                    for MEF, camera-matched render_raw4 for MFF/IVIF)
  <out>/<task>/<scene>_raw4.npy     the fused packed RAW4 (for renderer attribution)
  <out>/<task>/<scene>_rgb.png      only for v11: the learned RGB head output
  <out>/<task>/shapes.json          fused/source shapes for the audit trail

Usage:
  python infer_remote.py --checkpoint runs/v10/best.pt --config configs/v10.json --out preds/v10
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("RAWPRISM_ROOT", CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / "src"))

from finalfusion.infer import load_model  # noqa: E402
from finalfusion.isp_rawpy import CameraMeta, render_mosaic_via_dng, render_raw4  # noqa: E402
from finalfusion.mef.support.exposure import estimate_exposure_reference  # noqa: E402
from finalfusion.mef.support.raw import (  # noqa: E402
    denormalize_raw,
    normalize_raw,
    pack_bayer,
    read_pair_manifest,
    read_raw_mosaic,
    unpack_bayer,
)
from finalfusion.mff.support.raw_dataset import canonical_plane_indices, validate_cfa_pattern  # noqa: E402

def _ivif_meta() -> CameraMeta:
    """Neutral sRGB ISP used for the IVIF fused output and vis_rgb_rt."""
    return CameraMeta(
        black=16.0,
        white=4095.0,
        camera_wb=(1.0, 1.0, 1.0),
        color_matrix=np.eye(3, dtype=np.float32),
        pattern="RGGB",
        raw_pattern=np.array([[0, 1], [1, 2]], dtype=np.int64),
        gamma=1.0,
    )


IVIF_META = _ivif_meta()


def read_rgb(path: Path) -> np.ndarray:
    buffer = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image[..., ::-1]


def save_rgb(rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = (np.clip(rgb, 0.0, 1.0)[..., ::-1] * 255.0).round().astype(np.uint8)
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("encode failed {}".format(path))
    path.write_bytes(buffer.tobytes())


def _gray_world(rgb: np.ndarray) -> np.ndarray:
    means = rgb.reshape(-1, 3).mean(axis=0) + 1e-6
    return np.clip(rgb / means.reshape(1, 1, 3) * means.mean(), 0.0, 1.0)


def _percentile_stretch(rgb: np.ndarray, low: float = 0.5, high: float = 99.5) -> np.ndarray:
    lum = rgb.mean(axis=2)
    lo, hi = np.percentile(lum, [low, high])
    if hi - lo < 1e-5:
        return rgb
    return np.clip((rgb - lo) / (hi - lo), 0.0, 1.0)


def render_ivif_display(raw4: np.ndarray) -> np.ndarray:
    """Declared IVIF ISP: neutral sRGB + gray-world + 0.5/99.5 percentile."""
    return _percentile_stretch(_gray_world(render_raw4(raw4, IVIF_META)))


def load_any(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    arch = checkpoint.get("architecture", "")
    if arch.startswith("rawprism-v11"):
        from rawprism_v11 import RawPrismV11

        config = checkpoint.get("config", {})
        model = RawPrismV11(
            tasks=tuple(checkpoint["architecture_config"]["tasks"]),
            use_semantic=bool(config.get("use_semantic", False)),
            use_rft=bool(checkpoint["architecture_config"].get("use_rft", True)),
            use_priors=bool(checkpoint["architecture_config"].get("use_priors", False)),
            backbone_kwargs=config.get("backbone_kwargs"),
            gate_mode=checkpoint["architecture_config"].get("gate_mode", "source_softmax"),
        )
        model.load_state_dict(checkpoint["model"])
        return model.to(device).eval(), checkpoint, True
    model, checkpoint = load_model(str(checkpoint_path), "ivif", device)
    return model.eval(), checkpoint, False


@torch.no_grad()
def tiled_all(model, task, s1, s2, metadata, tile, halo, with_rgb):
    """Return (fused_raw4, fused_rgb or None, (h, w)) at the *source* resolution.

    The U-Net halves the field twice, so a tile edge must be a multiple of 8;
    instead of cropping the image (which silently shrinks the field of view and
    produced the 600x800 vs 592x800 mismatch in the released pipeline) the input
    is edge-padded to the next multiple of 8 and the output is cropped back.
    """
    height, width = s1.shape[-2], s1.shape[-1]
    pad_h = (-height) % 8
    pad_w = (-width) % 8
    if pad_h or pad_w:
        s1 = torch.nn.functional.pad(s1, (0, pad_w, 0, pad_h), mode="replicate")
        s2 = torch.nn.functional.pad(s2, (0, pad_w, 0, pad_h), mode="replicate")
    padded_h, padded_w = s1.shape[-2], s1.shape[-1]
    raw_out = torch.empty_like(s1)
    rgb_out = torch.empty((1, 3, padded_h * 2, padded_w * 2), device=s1.device, dtype=s1.dtype) if with_rgb else None
    for top in range(0, padded_h, tile):
        bottom = min(top + tile, padded_h)
        input_top, input_bottom = max(0, top - halo), min(padded_h, bottom + halo)
        for left in range(0, padded_w, tile):
            right = min(left + tile, padded_w)
            input_left, input_right = max(0, left - halo), min(padded_w, right + halo)
            out = model(task, s1[..., input_top:input_bottom, input_left:input_right],
                        s2[..., input_top:input_bottom, input_left:input_right], **metadata)
            raw_out[..., top:bottom, left:right] = out["fused_raw4"][
                ..., top - input_top:bottom - input_top, left - input_left:right - input_left
            ]
            if with_rgb:
                rgb_out[..., top * 2:bottom * 2, left * 2:right * 2] = out["fused_rgb"][
                    ..., (top - input_top) * 2:(bottom - input_top) * 2, (left - input_left) * 2:(right - input_left) * 2
                ]
    raw_out = raw_out[..., :height, :width]
    if rgb_out is not None:
        rgb_out = rgb_out[..., : height * 2, : width * 2]
    return raw_out, rgb_out, (height, width)


def task_scenes(task: str, config: dict, limit: int = 0) -> list[str]:
    if task == "ivif":
        root = ROOT / "data" / "ivif" / "test"
        scenes = sorted(p.name for p in root.iterdir() if p.is_dir())
    else:
        split = json.loads((ROOT / config["task_config"][task]["split_json"]).read_text(encoding="utf-8"))
        scenes = sorted(split["splits"]["test"])
    return scenes[:limit] if limit else scenes


def infer_ivif(model, config, device, out_dir, tile, halo, with_rgb, shapes, limit=0):
    task_cfg = config["task_config"]["ivif"]
    black, white = float(task_cfg["black_level"]), float(task_cfg["white_level"])
    from finalfusion.ivif.data import _pack_rggb, _read_ivif_pair

    for scene in task_scenes("ivif", config, limit):
        root = ROOT / "data" / "ivif" / "test" / scene
        visible, infrared = _read_ivif_pair(root, black, white)
        visible_raw4, infrared_raw4 = _pack_rggb(visible), _pack_rggb(infrared)
        gains = infrared_raw4.mean(axis=(1, 2)) / (visible_raw4.mean(axis=(1, 2)) + 1.0e-6)
        visible_raw4 = np.clip(visible_raw4 * gains[:, None, None], 0.0, 1.0).astype(np.float32)
        s1 = torch.from_numpy(visible_raw4).unsqueeze(0).to(device)
        s2 = torch.from_numpy(infrared_raw4).unsqueeze(0).to(device)
        fused, rgb, size = tiled_all(model, "ivif", s1, s2, {}, tile, halo, with_rgb)
        fused_np = fused[0].float().cpu().numpy()
        np.save(out_dir / "ivif" / (scene + "_raw4.npy"), fused_np)
        save_rgb(render_ivif_display(fused_np), out_dir / "ivif" / (scene + ".png"))
        if with_rgb:
            save_rgb(rgb[0].float().cpu().numpy().transpose(1, 2, 0), out_dir / "ivif" / (scene + "_rgb.png"))
        shapes[scene] = {"fused": list(size), "source": list(visible.shape[:2])}


def infer_mef(model, config, device, out_dir, tile, halo, with_rgb, shapes, limit=0):
    task_cfg = config["task_config"]["mef"]
    black, white = float(task_cfg["black_level"]), float(task_cfg["white_level"])
    pairs = {p.scene: p for p in read_pair_manifest(ROOT / "cache" / "mef" / "manifest.json")}
    for scene in task_scenes("mef", config, limit):
        pair = pairs[scene]
        under = normalize_raw(read_raw_mosaic(pair.under_path), black, white)
        over = normalize_raw(read_raw_mosaic(pair.over_path), black, white)
        under_packed = pack_bayer(under, pair.bayer_pattern)
        over_packed = pack_bayer(over, pair.bayer_pattern)
        reference = estimate_exposure_reference(under_packed, over_packed)
        metadata = {"exposure_ratio": torch.tensor([reference.over_to_under_ratio], device=device)}
        s1 = torch.from_numpy(under_packed).unsqueeze(0).to(device)
        s2 = torch.from_numpy(over_packed).unsqueeze(0).to(device)
        fused, rgb, size = tiled_all(model, "mef", s1, s2, metadata, tile, halo, with_rgb)
        fused_np = fused[0].float().cpu().numpy()
        np.save(out_dir / "mef" / (scene + "_raw4.npy"), fused_np)
        mosaic = unpack_bayer(fused_np, pair.bayer_pattern)
        mosaic_u16 = denormalize_raw(mosaic, black, white)
        save_rgb(render_mosaic_via_dng(mosaic_u16, pair.under_path), out_dir / "mef" / (scene + ".png"))
        if with_rgb:
            save_rgb(rgb[0].float().cpu().numpy().transpose(1, 2, 0), out_dir / "mef" / (scene + "_rgb.png"))
        shapes[scene] = {"fused": list(size), "source": [under.shape[0], under.shape[1]]}


def infer_mff(model, config, device, out_dir, tile, halo, with_rgb, shapes, limit=0):
    camera = CameraMeta.from_dict(json.loads((ROOT / "cache" / "camera_meta.json").read_text(encoding="utf-8")))
    for scene in task_scenes("mff", config, limit):
        npz = ROOT / "data" / "mff" / scene / "registered_raw_pair.npz"
        with np.load(str(npz), allow_pickle=False) as archive:
            reference = np.asarray(archive["reference_packed"], dtype=np.float32)
            moving = np.asarray(archive["moving_registered_packed"], dtype=np.float32)
            pattern = validate_cfa_pattern(archive["cfa_pattern"], str(npz))
        indices = canonical_plane_indices(pattern)
        first = np.clip(reference[list(indices)], 0.0, 1.0)
        second = np.clip(moving[list(indices)], 0.0, 1.0)
        h = min(first.shape[1], second.shape[1]) // 2 * 2
        w = min(first.shape[2], second.shape[2]) // 2 * 2
        first, second = first[:, :h, :w], second[:, :h, :w]
        s1 = torch.from_numpy(first).unsqueeze(0).to(device)
        s2 = torch.from_numpy(second).unsqueeze(0).to(device)
        fused, rgb, size = tiled_all(model, "mff", s1, s2, {}, tile, halo, with_rgb)
        fused_np = fused[0].float().cpu().numpy()
        np.save(out_dir / "mff" / (scene + "_raw4.npy"), fused_np)
        save_rgb(render_raw4(fused_np, camera), out_dir / "mff" / (scene + ".png"))
        if with_rgb:
            save_rgb(rgb[0].float().cpu().numpy().transpose(1, 2, 0), out_dir / "mff" / (scene + "_rgb.png"))
        shapes[scene] = {"fused": list(size), "source": [h * 2, w * 2]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tasks", nargs="*", default=["ivif", "mef", "mff"])
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--halo", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0, help="only the first N scenes per task (smoke tests)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint, with_rgb = load_any(Path(args.checkpoint), device)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    out_dir = Path(args.out)
    for task in args.tasks:
        (out_dir / task).mkdir(parents=True, exist_ok=True)
    shapes: dict = {}
    for task in args.tasks:
        print("[infer]", task, flush=True)
        {"ivif": infer_ivif, "mef": infer_mef, "mff": infer_mff}[task](
            model, config, device, out_dir, args.tile, args.halo, with_rgb, shapes, args.limit
        )
        (out_dir / task / "shapes.json").write_text(json.dumps(shapes, indent=2), encoding="utf-8")
    print("INFER_DONE")


if __name__ == "__main__":
    main()
