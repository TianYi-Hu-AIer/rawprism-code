"""Data-free smoke test for the RAWPrism-Large (Restormer) backbone."""

from __future__ import annotations

import io

import torch

from .rawprism import ARCHITECTURE_ID, RawPrism


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    torch.manual_seed(7)
    model = RawPrism(backbone="restormer", use_priors=False)
    require(set(model.tasks) == {"ivif", "mef", "mff"}, "all task adapters must exist")
    require(model.backbone_name == "restormer", "large model must use the restormer backbone")
    require(sum(1 for _ in model.named_modules() if _[0] == "backbone") == 1,
            "there must be one shared backbone")

    first = torch.rand(1, 4, 128, 128) * 0.8 + 0.05
    second = torch.rand(1, 4, 128, 128) * 0.8 + 0.05
    metadata = {
        "ivif": {},
        "mef": {"exposure_ratio": torch.tensor([4.0])},
        "mff": {},
    }
    for task in model.tasks:
        output = model(task, first, second, **metadata[task])
        require(output["fused_raw4"].shape == first.shape, "{} output contract failed".format(task))
        require(bool(torch.isfinite(output["fused_raw4"]).all()), "{} output is non-finite".format(task))
        require(float(output["fused_raw4"].min()) >= 0.0, "{} output below zero".format(task))
        require(float(output["fused_raw4"].max()) <= 1.0, "{} output above one".format(task))
        require("reliability_priors" not in output, "{} should not emit physics priors".format(task))
        require(output["network_input"].shape[1] == 12, "{} must use 12-channel input".format(task))

        model.zero_grad(set_to_none=True)
        output["fused_raw4"].sum().backward()
        backbone_grad = sum(float(p.grad.abs().sum()) for p in model.backbone.parameters() if p.grad is not None)
        require(backbone_grad > 0.0, "{} did not update the shared backbone".format(task))

    ivif_output = model("ivif", first, second)["fused_raw4"]
    mff_output = model("mff", first, second)["fused_raw4"]
    require(not torch.allclose(ivif_output, mff_output),
            "task conditioning must produce different outputs for different tasks")

    buffer = io.BytesIO()
    checkpoint = {
        "architecture": ARCHITECTURE_ID,
        "architecture_config": model.architecture_config(),
        "config": {"width": 44},
        "model": model.state_dict(),
    }
    torch.save(checkpoint, buffer)
    buffer.seek(0)
    restored_checkpoint = torch.load(buffer, weights_only=False)
    restored = RawPrism(
        tasks=tuple(restored_checkpoint["architecture_config"]["tasks"]),
        backbone="restormer",
        use_priors=False,
    )
    restored.load_state_dict(restored_checkpoint["model"])
    with torch.no_grad():
        expected = model("ivif", first, second)["fused_raw4"]
        actual = restored("ivif", first, second)["fused_raw4"]
    require(torch.equal(expected, actual), "checkpoint round trip changed model output")
    print("RAWPrism-Large smoke test passed; parameters={}".format(sum(p.numel() for p in model.parameters())))


if __name__ == "__main__":
    main()
