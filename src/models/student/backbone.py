from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def _sinusoidal_pos_encoding_2d(h: int, w: int, d_model: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Generate 2D sinusoidal positional encoding.

    Returns: [1, h*w, d_model]
    """
    assert d_model % 4 == 0, "d_model must be divisible by 4 for 2D pos encoding"
    half = d_model // 2
    quarter = d_model // 4

    y_pos = torch.arange(h, device=device, dtype=dtype).unsqueeze(1).expand(h, w)
    x_pos = torch.arange(w, device=device, dtype=dtype).unsqueeze(0).expand(h, w)

    div_term = torch.exp(torch.arange(0, quarter, device=device, dtype=dtype) * -(math.log(10000.0) / quarter))

    pe = torch.zeros(h, w, d_model, device=device, dtype=dtype)
    pe[:, :, 0:quarter] = torch.sin(x_pos.unsqueeze(-1) * div_term)
    pe[:, :, quarter:half] = torch.cos(x_pos.unsqueeze(-1) * div_term)
    pe[:, :, half:half + quarter] = torch.sin(y_pos.unsqueeze(-1) * div_term)
    pe[:, :, half + quarter:d_model] = torch.cos(y_pos.unsqueeze(-1) * div_term)

    return pe.reshape(1, h * w, d_model)


class StudentBackbone(nn.Module):
    """Multi-scale FPN feature tokenization from a frozen YOLO26 backbone.

    Replaces per-proposal RoI-Align with direct flattening of P3/P4/P5 
    feature maps into visual tokens. Each FPN level is adaptively pooled
    to a fixed spatial size, projected to hidden_dim, and augmented with
    learned scale embeddings and 2D sinusoidal positional encoding.

    The YOLO backbone is always frozen — only projections are trainable.
    """

    def __init__(
        self,
        hidden_dim: int,
        yolo_model=None,
        spatial_sizes: tuple[tuple[int, int], ...] = ((10, 10), (7, 7), (5, 5)),
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.spatial_sizes = spatial_sizes

        # Hide the ultralytics.YOLO wrapper from nn.Module since it
        # strongly overrides `.train()` and causes crashes.
        self._yolo_container = [yolo_model] if yolo_model is not None else []

        # Expose the internal PyTorch model so `.to(device)` moves weights.
        self.yolo_pt_model = yolo_model.model if yolo_model is not None else None

        # FPN channel dims are unknown until first forward pass.
        self._fpn_dims: list[int] | None = None
        self.projs: nn.ModuleList | None = None

        # Learned scale embeddings to distinguish FPN levels.
        num_levels = len(spatial_sizes)
        self.scale_embed = nn.Embedding(num_levels, hidden_dim)

        # Precompute total token count for downstream use.
        self.total_tokens = sum(h * w for h, w in spatial_sizes)

        self._initialized = False

    def _lazy_init(self, fpn_dims: list[int], device: torch.device) -> None:
        """Initialize projection layers once FPN channel dims are known."""
        self._fpn_dims = fpn_dims
        num_levels = min(len(fpn_dims), len(self.spatial_sizes))
        self.projs = nn.ModuleList([
            nn.Conv2d(fpn_dims[i], self.hidden_dim, kernel_size=1, bias=True).to(device)
            for i in range(num_levels)
        ])
        self._initialized = True

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extract multi-scale visual tokens from FPN features.

        Args:
            images: [B, 3, H, W] — input images.

        Returns:
            visual_tokens: [B, total_tokens, hidden_dim] — multi-scale visual tokens.
        """
        if not self._yolo_container:
            raise RuntimeError("YOLO backbone is required but was not provided.")

        fpn_features = self._extract_fpn_features(images)

        if not self._initialized:
            fpn_dims = [feat.shape[1] for feat in fpn_features]
            self._lazy_init(fpn_dims, images.device)

        num_levels = min(len(fpn_features), len(self.spatial_sizes))
        all_tokens: list[torch.Tensor] = []

        for level_idx in range(num_levels):
            feat_map = fpn_features[level_idx]
            target_h, target_w = self.spatial_sizes[level_idx]

            # Adaptively pool to fixed spatial size.
            pooled = F.adaptive_avg_pool2d(feat_map, (target_h, target_w))

            # Project channels to hidden_dim: [B, hidden_dim, H, W].
            projected = self.projs[level_idx](pooled)

            # Add scale embedding (broadcast over spatial dims).
            scale_emb = self.scale_embed(
                torch.tensor(level_idx, device=projected.device)
            )  # [hidden_dim]
            projected = projected + scale_emb.view(1, -1, 1, 1)

            # Flatten to tokens: [B, H*W, hidden_dim].
            tokens = projected.flatten(2).transpose(1, 2)

            # Add 2D sinusoidal positional encoding.
            pos_enc = _sinusoidal_pos_encoding_2d(
                target_h, target_w, self.hidden_dim,
                device=tokens.device, dtype=tokens.dtype,
            )
            tokens = tokens + pos_enc

            all_tokens.append(tokens)

        return torch.cat(all_tokens, dim=1)  # [B, total_tokens, hidden_dim]

    def _extract_fpn_features(self, images: torch.Tensor) -> list[torch.Tensor]:
        """Extract FPN feature maps from the YOLO backbone.

        Returns feature maps from finest (P3, stride 8) to coarsest (P5, stride 32).
        """
        if self.yolo_pt_model is None:
            raise RuntimeError(
                "StudentBackbone requires a YOLO model for FPN feature extraction. "
                "Pass yolo_model to __init__."
            )

        torch_model = self.yolo_pt_model.model

        with torch.no_grad():
            x = images

            if hasattr(torch_model, '__iter__'):
                save_outputs = {}
                for i, layer in enumerate(torch_model):
                    if hasattr(layer, 'f') and isinstance(layer.f, list):
                        inputs = [save_outputs.get(j, x) if j != -1 else x for j in layer.f]
                        x = layer(inputs)
                    elif hasattr(layer, 'f') and isinstance(layer.f, int) and layer.f != -1:
                        x = layer(save_outputs[layer.f])
                    else:
                        x = layer(x)
                    save_outputs[i] = x

                spatial_features = []
                for i in sorted(save_outputs.keys()):
                    feat = save_outputs[i]
                    if isinstance(feat, torch.Tensor) and feat.dim() == 4 and feat.shape[1] > 3:
                        spatial_features.append(feat)

                if len(spatial_features) >= 3:
                    return spatial_features[-3:]
                elif spatial_features:
                    return spatial_features
                else:
                    raise RuntimeError("Could not extract FPN features from YOLO model.")
            else:
                raise RuntimeError("Unsupported YOLO model structure for FPN extraction.")
