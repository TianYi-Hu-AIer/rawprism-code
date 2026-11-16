"""Task router assembling shared components and task-specific adapters."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .common import CfaAnalysis, CfaSynthesis, DirectionalGreenPredictor, ResidualProjection, TaskOutputHead, soft_clamp_unit
from .common import SharedU2FusionBackbone
from .ivif.adapter import IvifInputAdapter
from .mef.adapter import MefInputAdapter
from .mff.adapter import MffInputAdapter


ARCHITECTURE_ID = "finalfusion-multitask-latent6-v1"
TASKS = ("ivif", "mef", "mff")


class UnifiedFinalNet(nn.Module):
    """One checkpoint with task-routed adapters/heads and one shared backbone."""

    def __init__(self, tasks: tuple[str, ...] | list[str] = TASKS, width: int = 44) -> None:
        super().__init__()
        selected = tuple(dict.fromkeys(tasks))
        unknown = set(selected) - set(TASKS)
        if not selected or unknown:
            raise ValueError("tasks must be a non-empty subset of {}".format(TASKS))
        adapter_factories = {"ivif": IvifInputAdapter, "mef": MefInputAdapter, "mff": MffInputAdapter}
        head_factories = {"ivif": TaskOutputHead, "mef": lambda: TaskOutputHead(mef_external_order=True), "mff": TaskOutputHead}
        self.tasks = selected
        self.adapters = nn.ModuleDict({task: adapter_factories[task]() for task in selected})
        self.backbone = SharedU2FusionBackbone(width=width)
        self.heads = nn.ModuleDict({task: head_factories[task]() for task in selected})

    def forward(self, task: str, source1: torch.Tensor, source2: torch.Tensor, **metadata: Any) -> dict[str, Any]:
        if task not in self.adapters:
            raise ValueError("Task {!r} is not enabled; available tasks are {}".format(task, self.tasks))
        network_input, context = self.adapters[task](source1, source2, metadata)
        output = self.heads[task](self.backbone(network_input))
        output.update({"network_input": network_input, "context": context, "task": task})
        return output

    def architecture_config(self) -> dict[str, Any]:
        selector_modes = {"ivif": "shared", "mef": "shared", "mff": "separate"}
        return {"id": ARCHITECTURE_ID, "tasks": list(self.tasks),
                "shared_backbone": "joint12-u2fusion-44-dense5-latent6",
                "canonical_spatial_order": ["R", "Gh", "Gv", "B"],
                "selector_mode": {task: selector_modes[task] for task in self.tasks},
                "output_constraint": {task: "soft-beta50" for task in self.tasks}}


FinalFusion = UnifiedFinalNet

__all__ = ["ARCHITECTURE_ID", "TASKS", "FinalFusion", "UnifiedFinalNet", "CfaAnalysis", "CfaSynthesis",
           "DirectionalGreenPredictor", "ResidualProjection", "soft_clamp_unit"]
