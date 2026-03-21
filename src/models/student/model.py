from __future__ import annotations

import torch
from torch import nn

from src.models.student.backbone import StudentBackbone
from src.models.student.fusion import FusionTransformer
from src.models.student.text_encoder import StudentTextEncoder
from src.models.student.verifier import VerifierHead


class StudentModel(nn.Module):
    def __init__(self, hidden_dim: int, vocab_size: int, tob_tokens: int) -> None:
        super().__init__()
        self.backbone = StudentBackbone(hidden_dim=hidden_dim)
        self.text_encoder = StudentTextEncoder(vocab_size=vocab_size, hidden_dim=hidden_dim)
        self.fusion = FusionTransformer(hidden_dim=hidden_dim, target_tokens=tob_tokens)
        self.bbox_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 4),
        )
        self.verifier = VerifierHead(hidden_dim=hidden_dim)

    def forward(self, images: torch.Tensor, token_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        visual_tokens = self.backbone(images)
        text_tokens = self.text_encoder(token_ids, attention_mask)
        fused_tokens = self.fusion(visual_tokens, text_tokens)

        pooled_visual = fused_tokens.mean(dim=1)
        pooled_text = text_tokens.mean(dim=1)

        bbox = self.bbox_head(pooled_visual)
        verifier_logit = self.verifier(pooled_visual, pooled_text)

        return {
            "bbox": bbox,
            "verifier_logit": verifier_logit,
            "fused_tokens": fused_tokens,
            "text_tokens": text_tokens,
        }
