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
    def __init__(
        self,
        hidden_dim: int,
        target_tokens: int,
        num_layers: int,
        attention_heads: int,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be > 0")
        if attention_heads <= 0:
            raise ValueError("attention_heads must be > 0")
        if hidden_dim % attention_heads != 0:
            raise ValueError("hidden_dim must be divisible by attention_heads")

        self.input_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=attention_heads,
                    dropout=0.0,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.self_attention = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=attention_heads,
                    dropout=0.0,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_after_cross = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.norm_after_self = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.norm_after_mlp = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.mlp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                )
                for _ in range(num_layers)
            ]
        )
        self.target_tokens = target_tokens

    def forward(
        self,
        visual_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if text_mask is None:
            text_mask = torch.ones(
                text_tokens.shape[0],
                text_tokens.shape[1],
                dtype=torch.bool,
                device=text_tokens.device,
            )
        else:
            text_mask = text_mask.to(dtype=torch.bool)

        empty_rows = ~text_mask.any(dim=1)
        if empty_rows.any():
            text_mask = text_mask.clone()
            text_mask[empty_rows, 0] = True

        weights = text_mask.unsqueeze(-1).to(text_tokens.dtype)
        denominator = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        text_context = (text_tokens * weights).sum(dim=1, keepdim=True) / denominator

        fused = self.input_norm(visual_tokens + text_context)
        key_padding_mask = ~text_mask
        for layer_idx in range(len(self.cross_attention)):
            cross_output, _ = self.cross_attention[layer_idx](
                query=fused,
                key=text_tokens,
                value=text_tokens,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            fused = self.norm_after_cross[layer_idx](fused + cross_output)

            self_output, _ = self.self_attention[layer_idx](
                query=fused,
                key=fused,
                value=fused,
                need_weights=False,
            )
            fused = self.norm_after_self[layer_idx](fused + self_output)

            mlp_output = self.mlp[layer_idx](fused)
            fused = self.norm_after_mlp[layer_idx](fused + mlp_output)

        if fused.shape[1] != self.target_tokens:
            fused = _resize_spatial_tokens(fused, self.target_tokens)
        return fused
