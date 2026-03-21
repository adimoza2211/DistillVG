from __future__ import annotations

import torch
from torch import nn


class GroundingDINOTeacher(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.eval()

    @torch.no_grad()
    def forward(self, images: torch.Tensor, texts: list[str]) -> dict[str, torch.Tensor]:
        raise NotImplementedError("Implement teacher inference wrapper only.")
