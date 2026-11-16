"""RAWPrism v6 -- reliability-gated concatenation fusion.

v5 fused the two latents with a *weighted average* ``u = w1*z1 + w2*z2``, which
is physically correct for same-modality radiance fusion (MEF/MFF) but destroys
the complementary information for cross-modality fusion (IVIF: visible texture
+ thermal saliency).  v6 instead keeps the two sources separate and lets the
shared decoder combine them:

    u = concat(w1*z1, w2*z2)          # 12 channels, both sources preserved

The reliability maps ``w1, w2`` (from physical RAW cues, via the RFT) still gate
each source, but the decoder now sees both gated sources and learns the
cross-modality combination.  The differentiable renderer is unchanged.
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
    ReliabilityFieldTransformer,
    physical_reliability_priors,
)


ARCHITECTURE_ID = "rawprism-v6-reliability-gated-concat"
TASKS = ("ivif", "mef", "mff")


class RawPrismV6(nn.Module):
    """One shared, task-conditioned reliability-gated concatenation fusion model."""

    def __init__(
        self,
        tasks: tuple[str, ...] | list[str] = TASKS,
        token_dim: int = 16,
        cond_dim: int = 32,
        use_semantic: bool = False,
        use_rft: bool = True,
        use_priors: bool = True,
        backbone_kwargs: dict[str, Any] | None = None,
        visible_residual: bool = False,
        visible_residual_scale: float = 0.5,
    ) -> None:
        super().__init__()
        selected = tuple(dict.fromkeys(tasks))
        unknown = set(selected) - set(TASKS)
        if not selected or unknown:
            raise ValueError("tasks must be a non-empty subset of {}".format(TASKS))
        self.tasks = selected
        self.token_dim = int(token_dim)
        self.cond_dim = int(cond_dim)
        self.use_rft = bool(use_rft)
        self.use_priors = bool(use_priors)
        self.visible_residual = bool(visible_residual)
        self.visible_residual_scale = float(visible_residual_scale)
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
        if self.use_rft:
            self.rft = ReliabilityFieldTransformer(
                prior_channels=8, latent_channels=6, cond_dim=self.cond_dim, use_semantic=bool(use_semantic)
            )
        else:
            self.rft = None
        restormer_kwargs = {"in_channels": 12, "out_channels": 6, "cond_dim": self.cond_dim}
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
        if self.rft is not None:
            if self.use_priors:
                priors = physical_reliability_priors(raw1, raw2, task, noise=noise)
            else:
                priors = torch.zeros((source1.shape[0], 8, *source1.shape[2:]), device=source1.device)
            weight1, weight2 = self.rft(priors, latent1, latent2, condition)
            gated = torch.cat((weight1 * latent1, weight2 * latent2), dim=1)  # 12 channels
            reliability = {
                "weight_source1": weight1,
                "weight_source2": weight2,
                "reliability_priors": priors,
            }
        else:
            gated = torch.cat((latent1, latent2), dim=1)  # unweighted concat (no reliability field)
            reliability = None
        shared_latent = self.backbone(gated, condition)
        output = self.heads[task](shared_latent)
        if task == "ivif" and self.visible_residual:
            # Visible-anchored fusion: keep a fixed fraction of the visible RAW4
            # in the output, so fine-tuning only learns the IR/detail residual.
            alpha = min(max(self.visible_residual_scale, 0.0), 1.0)
            output["fused_raw4"] = torch.clamp(
                (1.0 - alpha) * raw1 + alpha * output["fused_raw4"], 0.0, 1.0
            )
        output["fused_srgb"] = self.renderer(output["fused_raw4"])
        if task == "ivif":
            # Render the visible source for the semantic-alignment loss (DINOv3).
            output["visible_srgb"] = self.renderer(raw1)
            output["ir_rgb"] = raw2.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        if reliability is not None:
            output["reliability"] = reliability
        output["network_input"] = gated
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
            "fusion": "reliability-gated-concatenation",
            "use_rft": self.use_rft,
            "use_priors": self.use_priors,
            "backbone": "restormer",
            "visible_residual": self.visible_residual,
            "visible_residual_scale": self.visible_residual_scale,
        }


__all__ = ["ARCHITECTURE_ID", "TASKS", "RawPrismV6"]

