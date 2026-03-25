from __future__ import annotations

import torch
from torch import nn


class AlignmentHead(nn.Module):
    """Text-visual alignment scoring head.

    Takes decoded object features and text features, produces a scalar
    alignment score that can be distilled from a teacher verifier's signal.

    Replaces the old VerifierHead that ranked proposals via ordinal scoring.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        decoded_features: torch.Tensor,
        text_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Compute alignment score.

        Args:
            decoded_features: [B, hidden_dim] — decoded object representation.
            text_embedding: [B, hidden_dim] — pooled text embedding.

        Returns:
            scores: [B] — alignment score per sample.
        """
        fused = torch.cat([decoded_features, text_embedding], dim=-1)
        return self.scorer(fused).squeeze(-1)


# Backward compatibility alias.
VerifierHead = AlignmentHead
