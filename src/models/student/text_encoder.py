from __future__ import annotations

import torch
from torch import nn

import mobileclip


class StudentTextEncoder(nn.Module):
    """MobileCLIP-S1 text encoder wrapper.

    Encodes grounding phrases into token-level and pooled embeddings with
    pretrained CLIP language understanding. Output dim = 512.
    """

    def __init__(
        self,
        hidden_dim: int,
        model_name: str = "mobileclip_s1",
        pretrained: str | None = None,
    ) -> None:
        super().__init__()
        # Load full CLIP model just to extract the text encoder.
        clip_model, _, _ = mobileclip.create_model_and_transforms(
            model_name,
            pretrained=pretrained if pretrained else None,
        )
        self.text_encoder = clip_model.text_encoder
        self.projection_dim = int(clip_model.projection_dim)

        # Store the tokenizer variant name for later use.
        self._model_name = model_name

        # Delete the image encoder to save memory — we only need text.
        del clip_model.image_encoder
        del clip_model

        # Project native CLIP features (e.g. 512) to student's hidden dim (e.g. 64).
        self.proj = nn.Linear(self.projection_dim, hidden_dim)

    @property
    def output_dim(self) -> int:
        return self.projection_dim

    def forward(self, token_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode token IDs into contextual embeddings.

        Args:
            token_ids: [B, seq_len] tensor of token IDs.
            attention_mask: [B, seq_len] tensor (1 for real token, 0 for pad).

        Returns:
            [B, seq_len, hidden_dim] tensor of text embeddings.
        """
        # MobileCLIP native forward can return all unprojected sequence embeddings
        # directly from the transformer layers (before pool/project)
        text_features: torch.Tensor = self.text_encoder(
            text_tokens=token_ids,
            return_all_tokens=True
        )  # [B, seq_len, 512]

        return self.proj(text_features)

    @staticmethod
    def get_tokenizer(model_name: str = "mobileclip_s1"):
        """Return the CLIP tokenizer function.

        Usage:
            tokenizer = StudentTextEncoder.get_tokenizer()
            token_ids = tokenizer(["a person wearing red"])  # [1, 77]
        """
        return mobileclip.get_tokenizer(model_name)
