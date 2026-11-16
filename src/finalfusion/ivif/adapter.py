"""IVIF-specific RAW adapter."""

import torch

from ..common import TaskInputAdapter


class IvifInputAdapter(TaskInputAdapter):
    def __init__(self) -> None:
        super().__init__(shared_selector=True)

    def alter_latents(self, first: torch.Tensor, second: torch.Tensor):
        achromatic_ir = torch.cat((second[:, :4], torch.zeros_like(second[:, 4:])), dim=1)
        return first, achromatic_ir, {"infrared_latent_achromatic": achromatic_ir}
