from __future__ import annotations

import torch
from torch import nn

from src.models.student.backbone import StudentBackbone
from src.models.student.fusion import FusionEncoderDecoder
from src.models.student.query_gen import TextGuidedQueryGen
from src.models.student.proposals import YOLO26ProposalGenerator
from src.models.student.text_encoder import StudentTextEncoder


def _masked_mean(
    tokens: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute masked mean pooling of tokens.

    Args:
        tokens: [B, seq_len, D]
        mask: [B, seq_len] — 1 for real tokens, 0 for padding.
    """
    weights = mask.unsqueeze(-1).to(tokens.dtype)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    return (tokens * weights).sum(dim=1) / denominator


class StudentModel(nn.Module):
    """SimVG-style dual-branch visual grounding model.

    Architecture:
        1. Frozen YOLO26 backbone → multi-scale FPN feature tokenization
        2. Frozen MobileCLIP-S1 → text token embeddings
        3. Text-guided query generation (word+sentence level)
        4. Encoder-decoder fusion (4 encoder + 3 decoder layers)
        5. Dual-branch box regression:
           - Decoder branch: decoded queries → 3-layer MLP → box (training)
           - MLP branch: gated memory pooling → 2-layer MLP → box (inference)
        6. Text-visual alignment head (for verifier knowledge distillation)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        attention_heads: int,
        num_decoder_queries: int = 1,
        visual_spatial_sizes: tuple[tuple[int, int], ...] = ((10, 10), (7, 7), (5, 5)),
        ffn_dim: int | None = None,
        dropout: float = 0.1,
        use_yolo26: bool = True,
        yolo26_model_cfg: str = "",
        yolo26_weights_path: str = "",
        yolo26_conf_threshold: float = 0.001,
        yolo26_iou_threshold: float = 0.85,
        text_encoder_model_name: str = "mobileclip_s1",
        text_encoder_pretrained: str | None = None,
    ) -> None:
        super().__init__()
        if ffn_dim is None:
            ffn_dim = hidden_dim * 4

        # Text encoder: MobileCLIP-S1 (pretrained CLIP language model).
        self.text_encoder = StudentTextEncoder(
            hidden_dim=hidden_dim,
            model_name=text_encoder_model_name,
            pretrained=text_encoder_pretrained,
        )

        # YOLO26 backbone (frozen, for FPN feature extraction only).
        self.yolo26: YOLO26ProposalGenerator | None = None
        if use_yolo26:
            self.yolo26 = YOLO26ProposalGenerator.from_config(
                max_proposals=1,  # Not used for proposals, only need the model
                conf_threshold=yolo26_conf_threshold,
                iou_threshold=yolo26_iou_threshold,
                model_cfg=yolo26_model_cfg,
                weights_path=yolo26_weights_path,
            )

        # Visual backbone: multi-scale FPN feature tokenization.
        yolo_model = self.yolo26.predictor if self.yolo26 is not None else None
        self.backbone = StudentBackbone(
            hidden_dim=hidden_dim,
            yolo_model=yolo_model,
            spatial_sizes=visual_spatial_sizes,
        )

        # Text-guided query generation.
        self.query_gen = TextGuidedQueryGen(
            hidden_dim=hidden_dim,
            num_queries=num_decoder_queries,
            num_heads=attention_heads,
        )

        # Encoder-decoder fusion transformer.
        self.fusion = FusionEncoderDecoder(
            hidden_dim=hidden_dim,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            num_heads=attention_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )

        # --- Heads ---

        # Decoder branch: box regression from decoded queries (training path).
        self.decoder_bbox_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 4),
        )

        # MLP branch: lightweight fast inference path.
        self.text_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.mlp_bbox_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 4),
        )

        # Text-visual alignment head (for verifier KD).
        self.alignment_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        images: torch.Tensor,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            images: [B, 3, H, W] — input images.
            token_ids: [B, seq_len] — tokenized text.
            attention_mask: [B, seq_len] — text attention mask.

        Returns:
            dict with keys:
                decoder_bbox: [B, 4] — decoder branch box prediction (training).
                mlp_bbox: [B, 4] — MLP branch box prediction (fast inference).
                bbox: [B, 4] — primary box prediction (decoder during training).
                alignment_score: [B] — text-visual alignment score.
                fused_tokens: [B, N_vis, hidden_dim] — encoder memory.
                text_tokens: [B, seq_len, hidden_dim] — text embeddings.
        """
        # 1. Text encoding (MobileCLIP-S1).
        text_tokens = self.text_encoder(token_ids, attention_mask)

        # 2. Multi-scale visual tokenization (frozen YOLO26 FPN).
        visual_tokens = self.backbone(images)

        # 3. Text-guided query generation.
        text_mask = attention_mask.bool()
        queries = self.query_gen(text_tokens, text_mask)

        # 4. Encoder-decoder fusion.
        decoded, memory = self.fusion(visual_tokens, text_tokens, text_mask, queries)

        # 5. Decoder branch: direct box regression.
        decoder_bbox = self.decoder_bbox_head(decoded[:, 0]).sigmoid()

        # 6. MLP branch: gated pooling → fast box regression.
        text_pool = _masked_mean(text_tokens, attention_mask)
        gate = self.text_gate(text_pool).unsqueeze(1)  # [B, 1, D]
        gated_memory = (memory * gate).mean(dim=1)  # [B, D]
        mlp_bbox = self.mlp_bbox_head(gated_memory).sigmoid()

        # 7. Text-visual alignment score (for verifier KD).
        align_feat = torch.cat([decoded[:, 0], text_pool], dim=-1)
        alignment_score = self.alignment_head(align_feat).squeeze(-1)

        return {
            "decoder_bbox": decoder_bbox,
            "mlp_bbox": mlp_bbox,
            "bbox": decoder_bbox,  # Backward compat: primary prediction
            "alignment_score": alignment_score,
            "fused_tokens": memory,
            "text_tokens": text_tokens,
        }
