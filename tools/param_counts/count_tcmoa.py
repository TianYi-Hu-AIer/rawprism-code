"""Count true parameters of TC-MoA Base checkpoint."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml

ROOT = Path("/workspace/rawprism")
sys.path.insert(0, str(ROOT / "repos/TC-MoA-main"))

import model.ViT_MAE as VIT_MAE  # noqa: E402


def main() -> None:
    cfg = yaml.safe_load((ROOT / "repos/TC-MoA-main/config/predict.yaml").read_text())
    cfg["device"] = "cpu"
    model = getattr(VIT_MAE, cfg["model_type"])(cfg)
    ckpt = torch.load(str(ROOT / "repos/TC-MoA-main/checkpoint/checkpoint-0-TC_MoA_Base_test.pth"),
                      map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    state_tensors = sum(v.numel() for v in model.state_dict().values())
    ckpt_tensors = sum(v.numel() for v in state.values() if torch.is_tensor(v))
    print("params", params, "trainable", trainable)
    print("state_tensors", state_tensors, "ckpt_tensors", ckpt_tensors)
    print("missing", len(missing), "unexpected", len(unexpected))
    print("unique named_parameters", len(list(model.named_parameters())))


if __name__ == "__main__":
    main()
