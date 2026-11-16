"""Count true parameters (named_parameters) of SwinFusion checkpoints."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path("/workspace/rawprism")
sys.path.insert(0, str(ROOT / "repos/SwinFusion-master"))

from models.network_swinfusion1 import SwinFusion  # noqa: E402

CKPTS = {
    "ivif": ROOT / "repos/SwinFusion-master/Model/Infrared_Visible_Fusion/Infrared_Visible_Fusion/models/10000_E.pth",
    "mef": ROOT / "repos/SwinFusion-master/Model/Multi_Exposure_Fusion/Multi_Exposure_Fusion/models/10000_E.pth",
    "mff": ROOT / "repos/SwinFusion-master/Model/Multi_Focus_Fusion/Multi_Focus_Fusion/models/10000_E.pth",
}


def count(model: torch.nn.Module) -> tuple[int, int, int]:
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    state = sum(v.numel() for v in model.state_dict().values())
    return params, trainable, state


def main() -> None:
    for task, path in CKPTS.items():
        if not path.is_file():
            print(task, "missing", path)
            continue
        model = SwinFusion(upscale=1, in_chans=1, img_size=128, window_size=8, img_range=1.,
                           depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                           mlp_ratio=2, upsampler=None, resi_connection="1conv")
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        state = ckpt.get("params", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        p, t, s = count(model)
        print(task, "params", p, "trainable", t, "state_tensors", s,
              "missing", len(missing), "unexpected", len(unexpected))


if __name__ == "__main__":
    main()
