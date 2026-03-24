from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import roi_align


class StudentBackbone(nn.Module):
    """Extracts per-proposal visual features from a frozen YOLO FPN backbone.

    Instead of a toy CNN, this uses the same YOLO model that generates proposals
    to extract P3/P4/P5 FPN feature maps, then applies RoI-Align to produce one
    feature vector per proposal box.

    The YOLO backbone is always frozen — only the projection layer is trainable.
    """

    def __init__(self, hidden_dim: int, yolo_model=None) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self._yolo = yolo_model  # Shared with YOLO26ProposalGenerator.
        self._fpn_dim: int | None = None
        self.proj: nn.Linear | None = None
        self.roi_output_size = 7

        # Projection will be lazily initialized on first forward pass
        # once we know the FPN channel dimension.
        self._initialized = False

    def _lazy_init(self, fpn_dim: int, device: torch.device) -> None:
        """Initialize projection layer once FPN dim is known."""
        self._fpn_dim = fpn_dim
        # roi_align output: [K, C, 7, 7] → mean pool → [K, C] → project → [K, hidden_dim]
        self.proj = nn.Linear(fpn_dim, self.hidden_dim).to(device)
        self._initialized = True

    def forward(self, images: torch.Tensor, proposal_boxes: torch.Tensor | None = None) -> torch.Tensor:
        """Extract visual tokens from FPN features using RoI-Align.

        Args:
            images: [B, 3, H, W] — input images.
            proposal_boxes: [B, K, 4] — normalized xyxy boxes in [0, 1].
                If None, returns a global average pooled feature (fallback).

        Returns:
            roi_tokens: [B, K, hidden_dim] — one feature vector per proposal.
        """
        batch_size = images.shape[0]
        _, _, img_h, img_w = images.shape

        if self._yolo is None:
            raise RuntimeError("YOLO backbone is required but was not provided.")

        # Extract FPN features from YOLO backbone.
        fpn_features = self._extract_fpn_features(images)

        if not self._initialized:
            # Determine FPN channel dimension from the deepest feature map.
            sample_feat = fpn_features[-1] if fpn_features else fpn_features[0]
            self._lazy_init(sample_feat.shape[1], images.device)

        if proposal_boxes is None:
            # Fallback: global pooling of deepest FPN feature.
            feat = fpn_features[-1]
            pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)
            return self.proj(pooled).unsqueeze(1)

        num_proposals = proposal_boxes.shape[1]

        # Convert normalized xyxy boxes to absolute pixel coordinates
        # and format as (batch_idx, x1, y1, x2, y2) for roi_align.
        boxes_abs = proposal_boxes.clone()
        boxes_abs[..., 0] *= img_w
        boxes_abs[..., 1] *= img_h
        boxes_abs[..., 2] *= img_w
        boxes_abs[..., 3] *= img_h

        # Build roi list: [total_rois, 5] where first column is batch index.
        roi_list = []
        for b in range(batch_size):
            batch_indices = torch.full(
                (num_proposals, 1), b, dtype=boxes_abs.dtype, device=boxes_abs.device
            )
            roi_list.append(torch.cat([batch_indices, boxes_abs[b]], dim=1))
        rois = torch.cat(roi_list, dim=0)  # [B*K, 5]

        # Compute proposal areas for FPN level assignment.
        widths = (rois[:, 3] - rois[:, 1]).clamp_min(1.0)
        heights = (rois[:, 4] - rois[:, 2]).clamp_min(1.0)
        areas = widths * heights

        # Assign each proposal to the best FPN level based on area.
        # P3 (finest): small, P4: medium, P5 (coarsest): large.
        num_levels = len(fpn_features)
        if num_levels >= 3:
            thresholds = [32.0 * 32.0, 96.0 * 96.0]  # pixels^2
            level_assignments = torch.zeros(rois.shape[0], dtype=torch.long, device=rois.device)
            level_assignments[areas > thresholds[0]] = 1
            level_assignments[areas > thresholds[1]] = 2
        elif num_levels == 2:
            level_assignments = torch.zeros(rois.shape[0], dtype=torch.long, device=rois.device)
            level_assignments[areas > 64.0 * 64.0] = 1
        else:
            level_assignments = torch.zeros(rois.shape[0], dtype=torch.long, device=rois.device)

        # RoI-Align from appropriate FPN level for each proposal.
        roi_features_list = []
        for level_idx in range(num_levels):
            level_mask = (level_assignments == level_idx)
            if not level_mask.any():
                continue

            level_rois = rois[level_mask]
            feat_map = fpn_features[level_idx]
            spatial_scale = feat_map.shape[2] / img_h

            aligned = roi_align(
                feat_map,
                level_rois,
                output_size=self.roi_output_size,
                spatial_scale=spatial_scale,
                aligned=True,
            )
            # Mean pool the 7x7 spatial grid → [N, C]
            pooled = aligned.mean(dim=(2, 3))
            roi_features_list.append((level_mask, pooled))

        # Reassemble in original order.
        total_rois = rois.shape[0]
        fpn_dim = fpn_features[0].shape[1]
        all_features = torch.zeros(total_rois, fpn_dim, device=images.device, dtype=images.dtype)
        for mask, feats in roi_features_list:
            all_features[mask] = feats.to(all_features.dtype)

        # Project to hidden_dim.
        projected = self.proj(all_features)  # [B*K, hidden_dim]
        return projected.reshape(batch_size, num_proposals, self.hidden_dim)

    def _extract_fpn_features(self, images: torch.Tensor) -> list[torch.Tensor]:
        """Extract FPN feature maps from the YOLO backbone.

        Returns feature maps from finest (P3, stride 8) to coarsest (P5, stride 32).
        """
        if self._yolo is None:
            raise RuntimeError(
                "StudentBackbone requires a YOLO model for FPN feature extraction. "
                "Pass yolo_model to __init__."
            )

        yolo_model = self._yolo
        # Access the underlying PyTorch model from Ultralytics.
        if hasattr(yolo_model, 'predictor') and hasattr(yolo_model.predictor, 'model'):
            torch_model = yolo_model.predictor.model.model
        elif hasattr(yolo_model, 'model'):
            if hasattr(yolo_model.model, 'model'):
                torch_model = yolo_model.model.model
            else:
                torch_model = yolo_model.model
        else:
            torch_model = yolo_model

        # Run through YOLO backbone + neck to get FPN outputs.
        # Ultralytics models have a sequential `model` attribute.
        # The backbone outputs are at specific layer indices.
        with torch.no_grad():
            features = []
            x = images

            # If the model is a nn.Sequential-like (Ultralytics DetectionModel),
            # iterate through layers and capture FPN outputs.
            if hasattr(torch_model, '__iter__'):
                save_outputs = {}
                for i, layer in enumerate(torch_model):
                    if hasattr(layer, 'f') and isinstance(layer.f, list):
                        # Layers with multiple inputs (concat, etc.)
                        inputs = [save_outputs.get(j, x) if j != -1 else x for j in layer.f]
                        x = layer(inputs)
                    elif hasattr(layer, 'f') and isinstance(layer.f, int) and layer.f != -1:
                        x = layer(save_outputs[layer.f])
                    else:
                        x = layer(x)
                    save_outputs[i] = x

                # Collect feature maps — look for the last 3 feature maps
                # that are 4D tensors (spatial feature maps).
                spatial_features = []
                for i in sorted(save_outputs.keys()):
                    feat = save_outputs[i]
                    if isinstance(feat, torch.Tensor) and feat.dim() == 4 and feat.shape[1] > 3:
                        spatial_features.append(feat)

                if len(spatial_features) >= 3:
                    # Take the last 3 spatial features (P3, P4, P5 from neck output).
                    features = spatial_features[-3:]
                elif spatial_features:
                    features = spatial_features
                else:
                    raise RuntimeError("Could not extract FPN features from YOLO model.")
            else:
                raise RuntimeError("Unsupported YOLO model structure for FPN extraction.")

        return features
