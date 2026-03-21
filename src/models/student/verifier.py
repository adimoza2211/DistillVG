from __future__ import annotations

import torch
from torch import nn


class VerifierHead(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, candidate_features: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([candidate_features, text_embedding], dim=-1)
        return self.scorer(fused).squeeze(-1)
