from __future__ import annotations

import torch
from torch import nn

from src.models.student.backbone import StudentBackbone
from src.models.student.fusion import FusionTransformer
from src.models.student.proposals import YOLO26ProposalGenerator
from src.models.student.text_encoder import StudentTextEncoder
from src.models.student.verifier import VerifierHead


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


class StudentModel(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        tob_tokens: int,
        proposal_count: int,
        use_yolo26_proposals: bool,
        yolo26_model_cfg: str,
        yolo26_weights_path: str,
        yolo26_conf_threshold: float,
        yolo26_iou_threshold: float,
    ) -> None:
        super().__init__()
        self.backbone = StudentBackbone(hidden_dim=hidden_dim)
        self.text_encoder = StudentTextEncoder(vocab_size=vocab_size, hidden_dim=hidden_dim)
        self.fusion = FusionTransformer(hidden_dim=hidden_dim, target_tokens=tob_tokens)
        self.proposal_count = int(proposal_count)
        self.use_yolo26_proposals = bool(use_yolo26_proposals)
        self.bbox_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 4),
        )
        self.proposal_bbox_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 4),
        )
        self.verifier = VerifierHead(hidden_dim=hidden_dim, num_classes=4)
        self.yolo26: YOLO26ProposalGenerator | None = None
        if self.use_yolo26_proposals:
            self.yolo26 = YOLO26ProposalGenerator.from_config(
                max_proposals=self.proposal_count,
                conf_threshold=float(yolo26_conf_threshold),
                iou_threshold=float(yolo26_iou_threshold),
                model_cfg=yolo26_model_cfg,
                weights_path=yolo26_weights_path,
            )

    def forward(self, images: torch.Tensor, token_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        visual_tokens = self.backbone(images)
        text_tokens = self.text_encoder(token_ids, attention_mask)
        fused_tokens = self.fusion(visual_tokens, text_tokens, attention_mask)

        pooled_visual = fused_tokens.mean(dim=1)
        text_weights = attention_mask.unsqueeze(-1).to(text_tokens.dtype)
        pooled_text = (text_tokens * text_weights).sum(dim=1) / text_weights.sum(dim=1).clamp_min(1.0)

        pooled_proposal_features = _select_evenly_spaced_tokens(fused_tokens, self.proposal_count)

        if self.yolo26 is None:
            proposal_boxes = self.proposal_bbox_head(pooled_proposal_features)
        else:
            proposal_boxes = self.yolo26.generate(images)

        proposal_verifier_logits = self.verifier(pooled_proposal_features, pooled_text)
        proposal_ranking_scores = VerifierHead.score(proposal_verifier_logits)

        top_indices = proposal_ranking_scores.argmax(dim=1, keepdim=True)
        gathered_boxes = torch.gather(
            proposal_boxes,
            dim=1,
            index=top_indices.unsqueeze(-1).expand(-1, -1, proposal_boxes.shape[-1]),
        )
        bbox = gathered_boxes.squeeze(1)

        verifier_logit = torch.gather(proposal_ranking_scores, dim=1, index=top_indices).squeeze(1)

        bbox = (bbox + self.bbox_head(pooled_visual)).clamp(0.0, 1.0)

        return {
            "bbox": bbox,
            "verifier_logit": verifier_logit,
            "proposal_boxes": proposal_boxes,
            "proposal_verifier_logits": proposal_verifier_logits,
            "fused_tokens": fused_tokens,
            "text_tokens": text_tokens,
        }
