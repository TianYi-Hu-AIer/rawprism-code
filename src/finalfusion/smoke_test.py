"""Data-free forward, loss, backward, and checkpoint tests."""

from __future__ import annotations

import io

import torch

from .losses import IvifTaskLoss, MefTaskLoss, MffTaskLoss
from .model import ARCHITECTURE_ID, CfaSynthesis, UnifiedFinalNet, soft_clamp_unit


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    torch.manual_seed(7)
    model = UnifiedFinalNet()
    require(set(model.adapters) == {"ivif", "mef", "mff"}, "all task adapters must exist")
    require(sum(1 for _ in model.named_modules() if _[0] == "backbone") == 1, "there must be one shared backbone")

    first = torch.rand(1, 4, 48, 48) * 0.8 + 0.05
    second = torch.rand(1, 4, 48, 48) * 0.8 + 0.05
    synthesis = CfaSynthesis()
    for task in model.tasks:
        spatial_first = first[:, (0, 1, 3, 2)] if task == "mef" else first
        latent, _ = model.adapters[task].analysis(spatial_first)
        restored = synthesis(latent)
        error = float((restored - spatial_first).abs().max().detach())
        require(error < 1.0e-5, "{} latent round trip failed: {}".format(task, error))

    ivif_output = model("ivif", first, second)
    require(torch.count_nonzero(ivif_output["context"]["source2_latent"][:, 4:]) == 0, "IR chroma must be zero")
    mef_output = model("mef", first, second, exposure_ratio=torch.tensor([4.0]))
    mff_output = model("mff", first, second)
    for task, output in (("ivif", ivif_output), ("mef", mef_output), ("mff", mff_output)):
        require(output["network_input"].shape == (1, 12, 48, 48), "{} input contract failed".format(task))
        require(output["fused_raw4"].shape == first.shape, "{} output contract failed".format(task))
        require(bool(torch.isfinite(output["fused_raw4"]).all()), "{} output is non-finite".format(task))
        require(float(output["fused_raw4"].min().detach()) >= 0.0, "{} output below zero".format(task))
        require(float(output["fused_raw4"].max().detach()) <= 1.0, "{} output above one".format(task))
        expected_clamp = soft_clamp_unit(output["fused_raw_preclamp"], beta=50.0)
        require(torch.equal(output["fused_raw4"], expected_clamp), "{} did not use the common soft clamp".format(task))

    from .mef.support.noise import NoiseModel

    noise = NoiseModel(
        under_a=[0.01] * 4,
        under_b=[0.001] * 4,
        over_a=[0.005] * 4,
        over_b=[0.0005] * 4,
        method="synthetic-smoke-test",
        samples_per_scene=1,
        flat_fraction=1.0,
        training_scenes=["synthetic"],
    )
    criteria = {
        "ivif": IvifTaskLoss(),
        "mef": MefTaskLoss(noise, loss_context=0),
        "mff": MffTaskLoss(),
    }
    constant = torch.full_like(first, 0.4)
    flat_mff = criteria["mff"].criterion(constant, constant, constant)
    require(torch.equal(flat_mff["weight1"], torch.full_like(flat_mff["weight1"], 0.5)),
            "flat MFF regions must use equal source weights")
    require(torch.equal(flat_mff["weight1"] + flat_mff["weight2"], torch.ones_like(flat_mff["weight1"])),
            "MFF focus weights must sum exactly to one")
    metadata = {
        "ivif": {},
        "mef": {"exposure_ratio": torch.tensor([4.0])},
        "mff": {},
    }
    for task in model.tasks:
        model.zero_grad(set_to_none=True)
        output = model(task, first, second, **metadata[task])
        loss, stats = criteria[task](output, first, second, metadata[task])
        require(bool(torch.isfinite(loss)), "{} loss is non-finite".format(task))
        require(all(bool(torch.isfinite(value)) for value in stats.values()),
                "{} loss statistics contain non-finite values".format(task))
        loss.backward()
        backbone_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.backbone.parameters()
            if parameter.grad is not None
        )
        require(backbone_gradient > 0.0, "{} did not update the shared backbone".format(task))

    from .infer import tiled_forward

    model.eval()
    with torch.no_grad():
        full = model("mff", first, second)["fused_raw4"]
        # halo must cover the 3-level U-Net receptive field (halo 12 leaves
        # a boundary residual of ~3e-6; 16 keeps the tiled/full error under
        # the tolerance with genuine partial tile overlap).
        tiled = tiled_forward(model, "mff", first, second, {}, tile_size=24, halo=16)
    tile_error = float((full - tiled).abs().max())
    require(tile_error < 2.0e-6, "tiled inference differs from full inference: {}".format(tile_error))

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
    restored_model = UnifiedFinalNet(tasks=tuple(restored_checkpoint["architecture_config"]["tasks"]))
    restored_model.load_state_dict(restored_checkpoint["model"])
    with torch.no_grad():
        expected = model("ivif", first, second)["fused_raw4"]
        actual = restored_model("ivif", first, second)["fused_raw4"]
    require(torch.equal(expected, actual), "checkpoint round trip changed model output")
    print("FinalFusion smoke test passed; parameters={}".format(sum(p.numel() for p in model.parameters())))


if __name__ == "__main__":
    main()
