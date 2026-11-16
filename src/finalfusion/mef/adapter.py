"""MEF-specific exposure normalization and CFA order adapter."""

from typing import Any

import torch

from ..common import TaskInputAdapter


class MefInputAdapter(TaskInputAdapter):
    _EXTERNAL_TO_SPATIAL = (0, 1, 3, 2)

    def __init__(self) -> None:
        super().__init__(shared_selector=True)

    @staticmethod
    def _ratio_tensor(ratio: Any, reference: torch.Tensor) -> torch.Tensor:
        value = torch.as_tensor(ratio, device=reference.device, dtype=reference.dtype)
        if value.ndim == 0:
            value = value.expand(reference.shape[0])
        if value.ndim == 1 and value.shape[0] == reference.shape[0]:
            value = value.view(-1, 1, 1, 1)
        if value.shape != (reference.shape[0], 1, 1, 1) or torch.any(value <= 0):
            raise ValueError("MEF exposure_ratio must be positive with one value per batch item")
        return value

    def prepare_sources(self, source1: torch.Tensor, source2: torch.Tensor, metadata: dict[str, Any]):
        if "exposure_ratio" not in metadata:
            raise KeyError("MEF requires metadata['exposure_ratio']")
        ratio = self._ratio_tensor(metadata["exposure_ratio"], source1)
        scale = torch.sqrt(ratio)
        under_reference, over_reference = source1 * scale, source2 / scale
        index = torch.as_tensor(self._EXTERNAL_TO_SPATIAL, device=source1.device)
        context = dict(metadata)
        context.update({"under_native": source1, "over_native": source2,
                        "under_reference": under_reference, "over_reference": over_reference,
                        "exposure_ratio": ratio})
        return under_reference.index_select(1, index), over_reference.index_select(1, index), context
