from __future__ import annotations

import torch
from torch import nn


class StudentTextEncoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, token_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        tokens = self.embedding(token_ids)
        tokens = self.proj(tokens)
        mask = attention_mask.unsqueeze(-1).to(tokens.dtype)
        return tokens * mask
