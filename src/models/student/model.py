from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.models.student.backbone import StudentBackbone
from src.models.student.fusion import FusionTransformer
from src.models.student.proposals import YOLO26ProposalGenerator
from src.models.student.text_encoder import StudentTextEncoder
from src.models.student.verifier import VerifierHead


class StudentModel(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        fusion_layers: int,
        attention_heads: int,
        roi_tokens: int,
        tob_tokens: int,
        proposal_count: int,
        use_yolo26_proposals: bool,
        yolo26_model_cfg: str,
        yolo26_weights_path: str,
        yolo26_conf_threshold: float,
        yolo26_iou_threshold: float,
        text_encoder_model_name: str = "mobileclip_s1",
        text_encoder_pretrained: str | None = None,
    ) -> None:
        super().__init__()
        # Text encoder: MobileCLIP-S1 (pretrained CLIP language model).
        self.text_encoder = StudentTextEncoder(
            model_name=text_encoder_model_name,
            pretrained=text_encoder_pretrained,
        )

        self.roi_tokens = int(roi_tokens)
        self.proposal_count = int(proposal_count)
        self.use_yolo26_proposals = bool(use_yolo26_proposals)

        # YOLO26 proposal generator (frozen, for proposals + FPN features).
        self.yolo26: YOLO26ProposalGenerator | None = None
        if self.use_yolo26_proposals:
            self.yolo26 = YOLO26ProposalGenerator.from_config(
                max_proposals=self.proposal_count,
                conf_threshold=float(yolo26_conf_threshold),
                iou_threshold=float(yolo26_iou_threshold),
                model_cfg=yolo26_model_cfg,
                weights_path=yolo26_weights_path,
            )

        # Visual backbone: extracts FPN features via RoI-Align from YOLO.
        yolo_model = self.yolo26.predictor if self.yolo26 is not None else None
        self.backbone = StudentBackbone(hidden_dim=hidden_dim, yolo_model=yolo_model)

        # Fusion transformer: cross-modal mixing of visual + text tokens.
        self.fusion = FusionTransformer(
            hidden_dim=hidden_dim,
            target_tokens=tob_tokens,
            num_layers=fusion_layers,
            attention_heads=attention_heads,
        )

        # Heads.
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

    def forward(
        self,
        images: torch.Tensor,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        precomputed_proposals: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # 1. Text encoding (MobileCLIP-S1).
        text_tokens = self.text_encoder(token_ids, attention_mask)

        # 2. Proposal generation.
        if precomputed_proposals is not None:
            proposal_boxes = precomputed_proposals
        elif self.yolo26 is not None:
            proposal_boxes = self.yolo26.generate(images)
        else:
            # Fallback: learned proposals from backbone features.
            visual_tokens = self.backbone(images)
            proposal_boxes = self.proposal_bbox_head(visual_tokens)

        # 3. Visual feature extraction (RoI-Align from YOLO FPN).
        if self.yolo26 is not None:
            roi_tokens = self.backbone(images, proposal_boxes)
        else:
            roi_tokens = self.backbone(images)

        # 4. Resize visual tokens to match fusion transformer's expected count.
        if roi_tokens.shape[1] != self.roi_tokens:
            roi_tokens = self._resize_spatial_tokens(roi_tokens, self.roi_tokens)

        # 5. Fusion: cross-modal mixing.
        fused_tokens = self.fusion(roi_tokens, text_tokens, attention_mask)

        # 6. Pooled features for heads.
        pooled_visual = fused_tokens.mean(dim=1)
        text_weights = attention_mask.unsqueeze(-1).to(text_tokens.dtype)
        pooled_text = (text_tokens * text_weights).sum(dim=1) / text_weights.sum(dim=1).clamp_min(1.0)

        # 7. Resize fused tokens for proposal-level scoring.
        pooled_proposal_features = self._resize_spatial_tokens(fused_tokens, self.proposal_count)

        # 8. Verifier head: rank proposals by language-visual compatibility.
        proposal_verifier_logits = self.verifier(pooled_proposal_features, pooled_text)
        proposal_ranking_scores = VerifierHead.score(proposal_verifier_logits)

        # 9. Select top proposal, refine box.
        top_indices = proposal_ranking_scores.argmax(dim=1, keepdim=True)
        gathered_boxes = torch.gather(
            proposal_boxes,
            dim=1,
            index=top_indices.unsqueeze(-1).expand(-1, -1, proposal_boxes.shape[-1]),
        )
        bbox = gathered_boxes.squeeze(1)
        verifier_logit = torch.gather(proposal_ranking_scores, dim=1, index=top_indices).squeeze(1)

        # 10. Box refinement residual.
        bbox = (bbox + self.bbox_head(pooled_visual)).clamp(0.0, 1.0)

        return {
            "bbox": bbox,
            "verifier_logit": verifier_logit,
            "proposal_boxes": proposal_boxes,
            "proposal_verifier_logits": proposal_verifier_logits,
            "fused_tokens": fused_tokens,
            "text_tokens": text_tokens,
        }

    @staticmethod
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

    @staticmethod
    def _resize_spatial_tokens(tokens: torch.Tensor, target_tokens: int) -> torch.Tensor:
        batch_size, source_tokens, hidden_dim = tokens.shape
        if source_tokens == target_tokens:
            return tokens

        source_h, source_w = StudentModel._factorize_target_tokens(source_tokens)
        target_h, target_w = StudentModel._factorize_target_tokens(target_tokens)
        feature_map = tokens.transpose(1, 2).reshape(batch_size, hidden_dim, source_h, source_w)

        if target_tokens > source_tokens:
            resized = F.interpolate(feature_map, size=(target_h, target_w), mode="bilinear", align_corners=False)
        else:
            resized = F.adaptive_avg_pool2d(feature_map, output_size=(target_h, target_w))

        return resized.flatten(2).transpose(1, 2)
