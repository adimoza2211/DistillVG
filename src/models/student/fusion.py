from __future__ import annotations

import torch
from torch import nn


class _EncoderLayer(nn.Module):
    """Pre-norm encoder layer: visual tokens cross-attend to text, then self-attend."""

    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        # Cross-attention: visual queries, text keys/values.
        self.norm_cross = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        # Self-attention on visual tokens.
        self.norm_self = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        # FFN.
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        visual: torch.Tensor,
        text: torch.Tensor,
        text_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Cross-attention (pre-norm).
        v = self.norm_cross(visual)
        cross_out, _ = self.cross_attn(
            query=v, key=text, value=text,
            key_padding_mask=text_key_padding_mask,
            need_weights=False,
        )
        visual = visual + cross_out

        # Self-attention (pre-norm).
        v = self.norm_self(visual)
        self_out, _ = self.self_attn(
            query=v, key=v, value=v,
            need_weights=False,
        )
        visual = visual + self_out

        # FFN (pre-norm).
        visual = visual + self.ffn(self.norm_ffn(visual))
        return visual


class _DecoderLayer(nn.Module):
    """Pre-norm decoder layer: queries self-attend, then cross-attend to memory."""

    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        # Self-attention on queries.
        self.norm_self = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        # Cross-attention: queries attend to encoder memory.
        self.norm_cross = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        # FFN.
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # Self-attention (pre-norm).
        q = self.norm_self(queries)
        self_out, _ = self.self_attn(
            query=q, key=q, value=q,
            need_weights=False,
        )
        queries = queries + self_out

        # Cross-attention to encoder memory (pre-norm).
        q = self.norm_cross(queries)
        cross_out, _ = self.cross_attn(
            query=q, key=memory, value=memory,
            need_weights=False,
        )
        queries = queries + cross_out

        # FFN (pre-norm).
        queries = queries + self.ffn(self.norm_ffn(queries))
        return queries


class FusionEncoderDecoder(nn.Module):
    """Encoder-decoder fusion transformer for visual grounding.

    Encoder: Visual tokens cross-attend to text tokens, producing
    text-conditioned visual features (memory).

    Decoder: Object queries (from TQG module) cross-attend to encoder
    memory, self-attend, then FFN. Standard DETR decoder pattern.

    Args:
        hidden_dim: Model dimensionality.
        num_encoder_layers: Number of encoder layers.
        num_decoder_layers: Number of decoder layers.
        num_heads: Number of attention heads.
        ffn_dim: FFN intermediate dimensionality.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        hidden_dim: int = 384,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 3,
        num_heads: int = 8,
        ffn_dim: int = 1536,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")

        self.encoder_layers = nn.ModuleList([
            _EncoderLayer(hidden_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.encoder_norm = nn.LayerNorm(hidden_dim)

        self.decoder_layers = nn.ModuleList([
            _DecoderLayer(hidden_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_decoder_layers)
        ])
        self.decoder_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        queries: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run encoder-decoder fusion.

        Args:
            visual_tokens: [B, N_vis, hidden_dim] — multi-scale visual tokens.
            text_tokens: [B, N_text, hidden_dim] — text token embeddings.
            text_mask: [B, N_text] — boolean mask (True = real token).
            queries: [B, N_queries, hidden_dim] — text-guided object queries.

        Returns:
            decoded: [B, N_queries, hidden_dim] — decoded object representations.
            memory: [B, N_vis, hidden_dim] — encoder output (text-conditioned visual features).
        """
        text_key_padding_mask = ~text_mask.bool() if text_mask.dtype != torch.bool else ~text_mask

        # Encoder: visual ← cross-attend → text.
        memory = visual_tokens
        for layer in self.encoder_layers:
            memory = layer(memory, text_tokens, text_key_padding_mask)
        memory = self.encoder_norm(memory)

        # Decoder: queries ← cross-attend → memory.
        decoded = queries
        for layer in self.decoder_layers:
            decoded = layer(decoded, memory)
        decoded = self.decoder_norm(decoded)

        return decoded, memory
