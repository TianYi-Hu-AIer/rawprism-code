"""RAWPrism v5 -- reliability-field unified RAW fusion.

Unlike v1/v2 (which concatenated the two latents and let a black-box backbone
decide the fusion), v5 makes the fusion *explicit*:

    u = w1*z1 + w2*z2 + g(w1*z1, w2*z2)

where ``w1, w2`` are per-pixel reliability maps estimated by a
:class:`ReliabilityFieldTransformer` from *physical* RAW cues (SNR / saturation /
focus / spectral saliency) rather than from network latents.  A differentiable
renderer then turns the fused linear RAW into tone-mapped sRGB, fixing the
"fused output is too dark" root cause of the v1/v2 failures.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .common import TaskOutputHead
from .ivif.adapter import IvifInputAdapter
from .mef.adapter import MefInputAdapter
from .mff.adapter import MffInputAdapter
from .restormer_backbone import RestormerUNet
from .v5_components import (
    DifferentiableRawRenderer,
    PhysicsAnchoredFusion,
    ReliabilityFieldTransformer,
    physical_reliability_priors,
)


ARCHITECTURE_ID = "rawprism-v5-reliability-field"
TASKS = ("ivif", "mef", "mff")


class RawPrismV5(nn.Module):
    """One shared, task-conditioned reliability-field RAW fusion model."""

    def __init__(
        self,
        tasks: tuple[str, ...] | list[str] = TASKS,
        token_dim: int = 16,
        cond_dim: int = 32,
        use_semantic: bool = False,
        backbone_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        selected = tuple(dict.fromkeys(tasks))
        unknown = set(selected) - set(TASKS)
        if not selected or unknown:
            raise ValueError("tasks must be a non-empty subset of {}".format(TASKS))
        self.tasks = selected
        self.token_dim = int(token_dim)
        self.cond_dim = int(cond_dim)
        self.task_to_id = {task: index for index, task in enumerate(TASKS)}
        self.task_embed = nn.Embedding(len(TASKS), self.token_dim)
        self.cond_mlp = nn.Sequential(
            nn.Linear(self.token_dim, self.cond_dim),
            nn.GELU(),
            nn.Linear(self.cond_dim, self.cond_dim),
        )
        adapter_factories = {"ivif": IvifInputAdapter, "mef": MefInputAdapter, "mff": MffInputAdapter}
        head_factories = {
            "ivif": TaskOutputHead,
            "mef": lambda: TaskOutputHead(mef_external_order=True),
            "mff": TaskOutputHead,
        }
        self.adapters = nn.ModuleDict({task: adapter_factories[task]() for task in selected})
        self.rft = ReliabilityFieldTransformer(
            prior_channels=8, latent_channels=6, cond_dim=self.cond_dim, use_semantic=bool(use_semantic)
        )
        self.fusion = PhysicsAnchoredFusion(latent_channels=6)
        restormer_kwargs = {"in_channels": 6, "out_channels": 6, "cond_dim": self.cond_dim}
        restormer_kwargs.update(backbone_kwargs or {})
        self.backbone = RestormerUNet(**restormer_kwargs)
        self.heads = nn.ModuleDict({task: head_factories[task]() for task in selected})
        self.renderer = DifferentiableRawRenderer(learnable_tone=True)

    def _condition(self, task: str, device: torch.device) -> torch.Tensor:
        index = torch.tensor(self.task_to_id[task], device=device)
        return self.cond_mlp(self.task_embed(index))

    def _noise(self, task: str, device: torch.device, batch: int) -> dict[str, torch.Tensor] | None:
        if task != "mef":
            return None
        # Fixed shot-noise proxy; the calibrated per-exposure noise model is used
        # inside the MEF loss, this only seeds the model-side reliability prior.
        shape = (batch, 4, 1, 1)
        return {
            "a1": torch.full(shape, 0.5, device=device),
            "b1": torch.full(shape, 1e-4, device=device),
            "a2": torch.full(shape, 0.5, device=device),
            "b2": torch.full(shape, 1e-4, device=device),
        }

    def forward(self, task: str, source1: torch.Tensor, source2: torch.Tensor, **metadata: Any) -> dict[str, Any]:
        if task not in self.adapters:
            raise ValueError("Task {!r} is not enabled; available tasks are {}".format(task, self.tasks))
        condition = self._condition(task, source1.device)
        _, context = self.adapters[task](source1, source2, dict(metadata))
        latent1 = context["source1_latent"]
        latent2 = context["source2_latent"]
        raw1 = context["source1_model_raw4"]
        raw2 = context["source2_model_raw4"]
        noise = self._noise(task, source1.device, source1.shape[0])
        priors = physical_reliability_priors(raw1, raw2, task, noise=noise)
        weight1, weight2 = self.rft(priors, latent1, latent2, condition)
        fused_latent = self.fusion(latent1, latent2, weight1, weight2)
        shared_latent = self.backbone(fused_latent, condition)
        output = self.heads[task](shared_latent)
        output["fused_srgb"] = self.renderer(output["fused_raw4"])
        output["reliability"] = {
            "weight_source1": weight1,
            "weight_source2": weight2,
            "reliability_priors": priors,
        }
        output["network_input"] = fused_latent
        output["context"] = context
        output["task"] = task
        return output

    def architecture_config(self) -> dict[str, Any]:
        return {
            "id": ARCHITECTURE_ID,
            "tasks": list(self.tasks),
            "token_dim": self.token_dim,
            "cond_dim": self.cond_dim,
            "canonical_spatial_order": ["R", "Gh", "Gv", "B"],
            "fusion": "physics-anchored-reliability-field",
            "backbone": "restormer",
        }


__all__ = ["ARCHITECTURE_ID", "TASKS", "RawPrismV5"]

