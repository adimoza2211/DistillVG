from __future__ import annotations

import torch
from torch import nn


class FusionTransformer(nn.Module):
    def __init__(self, hidden_dim: int, target_tokens: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.target_tokens = target_tokens

    def forward(self, visual_tokens: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
        text_context = text_tokens.mean(dim=1, keepdim=True)
        fused = self.norm(visual_tokens + text_context)
        fused = fused + self.mlp(fused)
        if fused.shape[1] > self.target_tokens:
            fused = fused[:, : self.target_tokens, :]
        return fused
