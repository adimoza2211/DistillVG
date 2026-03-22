from __future__ import annotations

import torch
from torch import nn


class VerifierHead(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int = 1) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.num_classes),
        )

    def forward(self, candidate_features: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:
        if candidate_features.dim() == 3 and text_embedding.dim() == 2:
            text_embedding = text_embedding.unsqueeze(1).expand(-1, candidate_features.shape[1], -1)
        fused = torch.cat([candidate_features, text_embedding], dim=-1)
        logits = self.scorer(fused)
        if self.num_classes == 1:
            return logits.squeeze(-1)
        return logits

    @staticmethod
    def score(logits: torch.Tensor) -> torch.Tensor:
        if logits.dim() == 2:
            return logits
        if logits.shape[-1] != 4:
            raise ValueError("Ordinal score expects 4-class logits in the last dimension.")
        probs = torch.softmax(logits, dim=-1)
        ordinal_weights = torch.tensor([3.0, 2.0, 1.0, 0.0], device=probs.device, dtype=probs.dtype)
        return (probs * ordinal_weights).sum(dim=-1)
