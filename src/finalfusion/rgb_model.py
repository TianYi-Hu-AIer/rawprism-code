"""RGB-domain controlled variant of RAWPrism.

This model reuses the *exact* same ``RestormerUNet`` backbone and task
conditioning as ``RawPrism``, but swaps the RAW/CFA input adapters and the
CFA-synthesis head for plain RGB equivalents.  The only differences are the
3-channel RGB input representation, the 3-channel RGB output head, and the
absence of CFA-specific loss terms.  This is what enables the controlled
RAW-vs-RGB, same-backbone ablation.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .restormer_backbone import RestormerUNet


TASKS = ("ivif", "mef", "mff")
ARCHITECTURE_ID = "rawprism-rgb-controlled"


class RGBInputAdapter(nn.Module):
    """Map one 3-channel RGB source to a 6-channel canonical latent."""

    def __init__(self, in_channels: int = 3, out_channels: int = 6) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class RGBOutputHead(nn.Module):
    """Map the 6-channel shared latent back to a 3-channel RGB image."""

    def __init__(self, in_channels: int = 6, out_channels: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(32, out_channels, 3, 1, 1),
        )
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(value))


class RawPrismRGB(nn.Module):
    """One shared task-conditioned RGB fusion model for IVIF, MEF, and MFF."""

    def __init__(
        self,
        tasks: tuple[str, ...] | list[str] = TASKS,
        token_dim: int = 16,
        cond_dim: int = 32,
        backbone_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        selected = tuple(dict.fromkeys(tasks))
        if not selected or set(selected) - set(TASKS):
            raise ValueError("tasks must be a non-empty subset of {}".format(TASKS))
        self.tasks = selected
        self.cond_dim = int(cond_dim)
        self.task_to_id = {task: index for index, task in enumerate(TASKS)}
        self.task_embed = nn.Embedding(len(TASKS), token_dim)
        self.cond_mlp = nn.Sequential(
            nn.Linear(token_dim, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.adapters = nn.ModuleDict({task: RGBInputAdapter() for task in selected})
        kwargs = {"in_channels": 12, "out_channels": 6, "cond_dim": cond_dim}
        kwargs.update(backbone_kwargs or {})
        self.backbone = RestormerUNet(**kwargs)
        self.heads = nn.ModuleDict({task: RGBOutputHead() for task in selected})

    def _condition(self, task: str, device: torch.device) -> torch.Tensor:
        index = torch.tensor(self.task_to_id[task], device=device)
        return self.cond_mlp(self.task_embed(index))

    def forward(self, task: str, source1: torch.Tensor, source2: torch.Tensor) -> dict[str, torch.Tensor]:
        if task not in self.adapters:
            raise ValueError("Task {!r} is not enabled".format(task))
        condition = self._condition(task, source1.device)
        latent1 = self.adapters[task](source1)
        latent2 = self.adapters[task](source2)
        shared = self.backbone(torch.cat((latent1, latent2), dim=1), condition)
        return {"fused_rgb": self.heads[task](shared), "shared_latent": shared}

    def architecture_config(self) -> dict[str, Any]:
        return {
            "id": ARCHITECTURE_ID,
            "tasks": list(self.tasks),
            "cond_dim": self.cond_dim,
            "backbone": "restormer",
            "input": "rgb",
        }


__all__ = ["TASKS", "ARCHITECTURE_ID", "RGBInputAdapter", "RGBOutputHead", "RawPrismRGB"]
