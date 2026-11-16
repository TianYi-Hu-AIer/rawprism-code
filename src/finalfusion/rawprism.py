"""RAWPrism: task-conditioned, reliability-aware unified RAW-domain fusion.

RAWPrism upgrades the original ``UnifiedFinalNet`` in one specific way: instead
of feeding the two source latents as a raw 12-channel concatenation into the
shared U2Fusion backbone, it first fuses them with a *task-conditioned
reliability aggregator* and then processes the fused 6-channel latent with a
*FiLM-conditioned shared backbone*.  The aggregator learns per-pixel, per-source
reliability weights conditioned on the active fusion task and on cheap
data-driven physical cues (local detail, saturation/validity, and local energy).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .common import TaskOutputHead, soft_clamp_unit
from .ivif.adapter import IvifInputAdapter
from .mef.adapter import MefInputAdapter
from .mff.adapter import MffInputAdapter
from .loss_utils import gradient_magnitude
from .restormer_backbone import RestormerUNet


ARCHITECTURE_ID = "rawprism-task-conditioned-reliability-v1"
TASKS = ("ivif", "mef", "mff")


class FiLM(nn.Module):
    """Feature-wise linear modulation from a global condition vector."""

    def __init__(self, cond_dim: int, channels: int) -> None:
        super().__init__()
        self.scale = nn.Linear(cond_dim, channels)
        self.shift = nn.Linear(cond_dim, channels)

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        scale = self.scale(condition)
        shift = self.shift(condition)
        scale = scale.view(*scale.shape, 1, 1) + 1.0
        shift = shift.view(*shift.shape, 1, 1)
        return value * scale + shift


class FiLMReflectConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int, activate: bool = True,
                 use_film: bool = True) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3)
        self.activate = bool(activate)
        self.use_film = bool(use_film)
        self.film = FiLM(cond_dim, out_channels) if self.use_film else None

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        value = self.conv(F.pad(value, (1, 1, 1, 1), mode="reflect"))
        value = F.leaky_relu(value, negative_slope=0.2, inplace=True) if self.activate else value
        return self.film(value, condition) if self.use_film else value


class FiLMDenseLayer(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int, cond_dim: int, use_film: bool = True) -> None:
        super().__init__()
        self.layer = FiLMReflectConv(in_channels, growth_rate, cond_dim, use_film=use_film)

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.layer(value, condition), value), dim=1)


class FiLMConditionedBackbone(nn.Module):
    """Shared radiance encoder/decoder with a task condition injected via FiLM."""

    def __init__(self, width: int = 44, cond_dim: int = 32, use_film: bool = True,
                 in_channels: int = 20) -> None:
        super().__init__()
        self.stem = FiLMReflectConv(in_channels, width, cond_dim, use_film=use_film)
        channels = width
        self.dense = nn.ModuleList()
        for _ in range(5):
            self.dense.append(FiLMDenseLayer(channels, width, cond_dim, use_film=use_film))
            channels += width
        self.decoder = nn.ModuleList([
            FiLMReflectConv(channels, 128, cond_dim, use_film=use_film),
            FiLMReflectConv(128, 64, cond_dim, use_film=use_film),
            FiLMReflectConv(64, 32, cond_dim, use_film=use_film),
            FiLMReflectConv(32, 6, cond_dim, activate=False, use_film=use_film),
        ])
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu")
                nn.init.zeros_(module.bias)

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        value = self.stem(value, condition)
        for layer in self.dense:
            value = layer(value, condition)
        for layer in self.decoder:
            value = layer(value, condition)
        return value


def compute_physics_priors(context: dict[str, Any]) -> torch.Tensor:
    """Task-aware physical reliability cues injected as extra channels.

    Per source: [detail, saturation, energy, task_physics].  ``task_physics`` is
    an exposure-based Poisson SNR proxy for MEF (under/over), and zero for
    MFF/IVIF where sharpness/saliency are already captured by ``detail``.
    """

    def base(latent: torch.Tensor, raw4: torch.Tensor):
        detail = gradient_magnitude(latent.mean(dim=1, keepdim=True))
        detail = detail / (detail.mean(dim=(1, 2, 3), keepdim=True) + 1.0e-6)
        saturation = ((raw4 < 0.02) | (raw4 > 0.98)).float().mean(dim=1, keepdim=True)
        energy = raw4.mean(dim=1, keepdim=True)
        return detail, saturation, energy

    first = base(context["source1_latent"], context["source1_model_raw4"])
    second = base(context["source2_latent"], context["source2_model_raw4"])
    ratio = context.get("exposure_ratio")
    if ratio is not None:
        ratio = ratio.view(-1, 1, 1, 1) if ratio.ndim else ratio
        snr1 = first[2] / (torch.sqrt(ratio) + 1.0e-6)
        snr2 = second[2] * torch.sqrt(ratio)
    else:
        snr1 = torch.zeros_like(first[2])
        snr2 = torch.zeros_like(second[2])
    return torch.cat((first[0], first[1], first[2], snr1,
                      second[0], second[1], second[2], snr2), dim=1)


class TaskConditionedReliabilityAggregator(nn.Module):
    """Predict per-pixel, per-source reliability weights conditioned on the task.

    The weights are applied as a reliability-weighted concatenation in the
    caller, so the shared backbone still receives the full 12-channel source
    information instead of a compressed 6-channel fusion.
    """

    def __init__(self, latent_channels: int = 6, prior_channels: int = 6,
                 cond_dim: int = 32, hidden: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(2 * latent_channels + prior_channels, hidden, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.film = FiLM(cond_dim, hidden)
        self.to_logits = nn.Conv2d(hidden, 2, 1)
        nn.init.zeros_(self.to_logits.weight)
        nn.init.zeros_(self.to_logits.bias)

    def forward(self, condition: torch.Tensor, latent1: torch.Tensor, latent2: torch.Tensor,
                priors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        value = F.gelu(self.conv1(torch.cat((latent1, latent2, priors), dim=1)))
        value = F.gelu(self.film(self.conv2(value), condition))
        logits = self.to_logits(value)
        weights = torch.softmax(logits, dim=1)
        weight1, weight2 = weights[:, 0:1], weights[:, 1:2]
        return weight1, weight2, {"weight_source1": weight1, "weight_source2": weight2,
                                  "reliability_priors": priors}


class RawPrism(nn.Module):
    """One shared, task-conditioned RAW fusion model for IVIF, MEF, and MFF."""

    def __init__(self, tasks: tuple[str, ...] | list[str] = TASKS, width: int = 44,
                 token_dim: int = 16, cond_dim: int = 32, use_film: bool = True,
                 backbone: str = "dense", use_priors: bool = True,
                 backbone_kwargs: dict[str, Any] | None = None) -> None:
        super().__init__()
        selected = tuple(dict.fromkeys(tasks))
        unknown = set(selected) - set(TASKS)
        if not selected or unknown:
            raise ValueError("tasks must be a non-empty subset of {}".format(TASKS))
        self.tasks = selected
        self.token_dim = int(token_dim)
        self.cond_dim = int(cond_dim)
        self.use_film = bool(use_film)
        if backbone not in ("dense", "restormer"):
            raise ValueError("backbone must be 'dense' or 'restormer'")
        self.backbone_name = backbone
        self.use_priors = bool(use_priors)
        self.backbone_kwargs = dict(backbone_kwargs or {})
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
        self.aggregator = (
            TaskConditionedReliabilityAggregator(
                latent_channels=6, prior_channels=8, cond_dim=self.cond_dim, hidden=32,
            ) if self.use_priors else None
        )
        backbone_in_channels = 12
        if self.backbone_name == "restormer":
            restormer_kwargs = {
                "in_channels": backbone_in_channels,
                "out_channels": 6,
                "cond_dim": self.cond_dim,
            }
            restormer_kwargs.update(self.backbone_kwargs)
            self.backbone = RestormerUNet(**restormer_kwargs)
        else:
            self.backbone = FiLMConditionedBackbone(
                width=width, cond_dim=self.cond_dim, use_film=self.use_film,
                in_channels=backbone_in_channels,
            )
        self.heads = nn.ModuleDict({task: head_factories[task]() for task in selected})

    def _condition(self, task: str, device: torch.device) -> torch.Tensor:
        index = torch.tensor(self.task_to_id[task], device=device)
        return self.cond_mlp(self.task_embed(index))

    def forward(self, task: str, source1: torch.Tensor, source2: torch.Tensor, **metadata: Any) -> dict[str, Any]:
        if task not in self.adapters:
            raise ValueError("Task {!r} is not enabled; available tasks are {}".format(task, self.tasks))
        condition = self._condition(task, source1.device)
        network_input, context = self.adapters[task](source1, source2, dict(metadata))
        latent1 = context["source1_latent"]
        latent2 = context["source2_latent"]
        reliability = None
        if self.aggregator is not None:
            priors = compute_physics_priors(context)
            weight1, weight2, agg = self.aggregator(condition, latent1, latent2, priors)
            network_input = torch.cat((weight1 * latent1, weight2 * latent2), dim=1)
            reliability = {
                "weight_source1": weight1,
                "weight_source2": weight2,
                "reliability_priors": priors,
            }
        else:
            network_input = torch.cat((latent1, latent2), dim=1)
        shared_latent = self.backbone(network_input, condition)
        output = self.heads[task](shared_latent)
        extra_output = {
            "network_input": network_input,
            "context": context,
            "task": task,
        }
        if reliability is not None:
            extra_output["reliability"] = reliability
        output.update(extra_output)
        return output

    def architecture_config(self) -> dict[str, Any]:
        return {
            "id": ARCHITECTURE_ID,
            "tasks": list(self.tasks),
            "token_dim": self.token_dim,
            "cond_dim": self.cond_dim,
            "canonical_spatial_order": ["R", "Gh", "Gv", "B"],
            "fusion": "physics-prior-augmented-concatenation" if self.use_priors else "canonical-concatenation",
            "backbone": self.backbone_name,
            "output_constraint": {task: "soft-beta50" for task in self.tasks},
        }


FinalPrism = RawPrism


__all__ = ["ARCHITECTURE_ID", "TASKS", "RawPrism", "FinalPrism", "FiLM",
           "FiLMConditionedBackbone", "TaskConditionedReliabilityAggregator",
           "compute_physics_priors", "soft_clamp_unit"]
