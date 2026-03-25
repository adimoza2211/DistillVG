from __future__ import annotations

import torch
from torch import nn


class TextGuidedQueryGen(nn.Module):
    """Generate decoder queries conditioned on text input.

    Combines word-level cross-attention (queries attend to individual text tokens)
    and sentence-level gating (pooled text modulates queries) following
    SimVG (NeurIPS 2024) and RefFormer (NeurIPS 2024).

    Args:
        hidden_dim: Dimensionality of query/text features.
        num_queries: Number of object queries to generate (1 for REC).
        num_heads: Number of attention heads for word-level cross-attention.
    """

    def __init__(self, hidden_dim: int = 384, num_queries: int = 1, num_heads: int = 8) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim

        # Learnable query prototypes.
        self.query_prototypes = nn.Parameter(torch.randn(num_queries, hidden_dim) * 0.02)

        # Word-level: queries cross-attend to text tokens.
        self.word_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.word_norm = nn.LayerNorm(hidden_dim)

        # Sentence-level: project pooled text to modulate queries.
        self.sent_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.sent_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Generate text-conditioned queries.

        Args:
            text_tokens: [B, seq_len, hidden_dim] — text embeddings.
            text_mask: [B, seq_len] — boolean mask (True = real token).

        Returns:
            queries: [B, num_queries, hidden_dim] — text-conditioned decoder queries.
        """
        batch_size = text_tokens.shape[0]

        # Expand prototypes to batch.
        proto = self.query_prototypes.unsqueeze(0).expand(batch_size, -1, -1)

        # Word-level cross-attention: queries attend to text tokens.
        key_padding_mask = ~text_mask.bool() if text_mask.dtype != torch.bool else ~text_mask
        attended, _ = self.word_attn(
            query=proto,
            key=text_tokens,
            value=text_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        queries = self.word_norm(proto + attended)

        # Sentence-level gating: pooled text modulates queries.
        weights = text_mask.unsqueeze(-1).to(text_tokens.dtype)
        denominator = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled_text = (text_tokens * weights).sum(dim=1) / denominator.squeeze(1)
        sent_gate = self.sent_proj(pooled_text).unsqueeze(1)
        queries = self.sent_norm(queries + sent_gate)

        return queries
