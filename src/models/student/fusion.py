from __future__ import annotations

import torch
from torch import nn


def _select_evenly_spaced_tokens(tokens: torch.Tensor, target_tokens: int) -> torch.Tensor:
    batch_size, source_tokens, hidden_dim = tokens.shape
    if target_tokens <= 0:
        raise ValueError("target_tokens must be > 0")
    if source_tokens == target_tokens:
        return tokens
    if source_tokens > target_tokens:
        indices = torch.linspace(
            0,
            source_tokens - 1,
            steps=target_tokens,
            device=tokens.device,
        ).round().long()
        gather_index = indices.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, hidden_dim)
        return torch.gather(tokens, dim=1, index=gather_index)

    padding = tokens.new_zeros(batch_size, target_tokens - source_tokens, hidden_dim)
    return torch.cat([tokens, padding], dim=1)


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

    def forward(
        self,
        visual_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if text_mask is None:
            text_context = text_tokens.mean(dim=1, keepdim=True)
        else:
            weights = text_mask.unsqueeze(-1).to(text_tokens.dtype)
            denominator = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
            text_context = (text_tokens * weights).sum(dim=1, keepdim=True) / denominator

        fused = self.norm(visual_tokens + text_context)
        fused = fused + self.mlp(fused)
        if fused.shape[1] != self.target_tokens:
            fused = _select_evenly_spaced_tokens(fused, self.target_tokens)
        return fused
