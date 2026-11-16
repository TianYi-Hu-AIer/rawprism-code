"""Count true parameters of UniFusion (DinoFusion) with shared DINOv3 deduplicated."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path("/workspace/rawprism")
sys.path.insert(0, str(ROOT / "repos/UniFusion"))
sys.path.insert(0, str(ROOT / "repos/UniFusion/dinov3"))

from models.network_dinov3 import DinoFusion  # noqa: E402


def main() -> None:
    device = torch.device("cuda")
    repo_dir = str(ROOT / "repos/UniFusion/dinov3")
    dino_ckpt = str(ROOT / "repos/UniFusion/dinov3_vits16_pretrain_lvd1689m-08c60483.pth")
    extractor = torch.hub.load(repo_dir, "dinov3_vits16", source="local", weights=dino_ckpt).to(device)
    net = DinoFusion(
        upscale=1, in_chans=1, img_size=128, window_size=8, img_range=1.0,
        embed_dim=64, num_heads=[6, 6, 6, 6], mlp_ratio=2,
        upsampler=None, resi_connection="1conv", extractor=extractor,
    ).to(device)
    ckpt_path = ROOT / "repos/UniFusion/Model/Infrared_Visible_Fusion/G.pth"
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = ckpt.get("params", ckpt.get("state_dict", ckpt))
    missing, unexpected = net.load_state_dict(state, strict=False)
    params = sum(p.numel() for p in net.parameters())
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    state_tensors = sum(v.numel() for v in net.state_dict().values())
    ckpt_tensors = sum(v.numel() for v in state.values() if torch.is_tensor(v))
    print("params", params, "trainable", trainable)
    print("state_tensors", state_tensors, "ckpt_tensors", ckpt_tensors)
    print("missing", len(missing), "unexpected", len(unexpected))
    print("unique named_parameters", len(list(net.named_parameters())))


if __name__ == "__main__":
    main()
