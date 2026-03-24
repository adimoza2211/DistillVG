from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _factorize_target_tokens(target_tokens: int) -> tuple[int, int]:
    if target_tokens <= 0:
        raise ValueError("target_tokens must be > 0")

    best_h = 1
    best_w = target_tokens
    best_gap = abs(best_w - best_h)
    for height in range(1, int(target_tokens**0.5) + 1):
        if target_tokens % height != 0:
            continue
        width = target_tokens // height
        gap = abs(width - height)
        if gap < best_gap:
            best_h, best_w, best_gap = height, width, gap
    return best_h, best_w


def _resize_spatial_tokens(tokens: torch.Tensor, target_tokens: int) -> torch.Tensor:
    batch_size, source_tokens, hidden_dim = tokens.shape
    if source_tokens == target_tokens:
        return tokens

    source_h, source_w = _factorize_target_tokens(source_tokens)

    target_h, target_w = _factorize_target_tokens(target_tokens)
    feature_map = tokens.transpose(1, 2).reshape(batch_size, hidden_dim, source_h, source_w)
    resized = F.adaptive_avg_pool2d(feature_map, output_size=(target_h, target_w))
    return resized.flatten(2).transpose(1, 2)


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
            fused = _resize_spatial_tokens(fused, self.target_tokens)
        return fused
