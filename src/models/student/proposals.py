from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import torch

os.environ.setdefault("YOLO_AUTOINSTALL", "false")

from ultralytics import YOLO


@dataclass(frozen=True)
class YOLO26ProposalConfig:
    model_cfg: str
    weights_path: str
    max_proposals: int
    conf_threshold: float
    iou_threshold: float


def _default_yolo26_cfg_path() -> str:
    import ultralytics

    root = Path(ultralytics.__file__).resolve().parent
    candidate = root / "cfg" / "models" / "26" / "yolo26.yaml"
    if not candidate.exists():
        raise FileNotFoundError(f"Unable to find yolo26.yaml in ultralytics package: {candidate}")
    return str(candidate)


def _normalize_xyxy(boxes: torch.Tensor, image_width: int, image_height: int) -> torch.Tensor:
    normalized = boxes.clone().float()
    normalized[:, 0] = normalized[:, 0] / max(image_width, 1)
    normalized[:, 1] = normalized[:, 1] / max(image_height, 1)
    normalized[:, 2] = normalized[:, 2] / max(image_width, 1)
    normalized[:, 3] = normalized[:, 3] / max(image_height, 1)
    return normalized.clamp_(0.0, 1.0)


class YOLO26ProposalGenerator:
    def __init__(self, config: YOLO26ProposalConfig) -> None:
        model_source = config.weights_path.strip() if config.weights_path.strip() else config.model_cfg
        self.predictor = YOLO(model_source)
        self.max_proposals = int(config.max_proposals)
        self.conf_threshold = float(config.conf_threshold)
        self.iou_threshold = float(config.iou_threshold)

    @classmethod
    def from_config(
        cls,
        *,
        max_proposals: int,
        conf_threshold: float,
        iou_threshold: float,
        model_cfg: str | None,
        weights_path: str,
    ) -> "YOLO26ProposalGenerator":
        resolved_cfg = model_cfg.strip() if model_cfg and model_cfg.strip() else _default_yolo26_cfg_path()
        config = YOLO26ProposalConfig(
            model_cfg=resolved_cfg,
            weights_path=weights_path,
            max_proposals=max_proposals,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )
        return cls(config)

    def generate(self, images: torch.Tensor) -> torch.Tensor:
        batch_size, _, image_height, image_width = images.shape
        with torch.no_grad():
            results = self.predictor.predict(
                source=images,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                max_det=self.max_proposals,
                verbose=False,
                imgsz=max(image_height, image_width),
                stream=False,
            )

        packed_boxes: list[torch.Tensor] = []
        for result in results:
            if result.boxes is None or result.boxes.xyxy.numel() == 0:
                boxes = torch.zeros(self.max_proposals, 4, dtype=images.dtype, device=images.device)
                packed_boxes.append(boxes)
                continue

            xyxy = result.boxes.xyxy.to(images.device)
            conf = result.boxes.conf.to(images.device)
            sort_idx = torch.argsort(conf, descending=True)
            xyxy = xyxy[sort_idx][: self.max_proposals]
            xyxy = _normalize_xyxy(xyxy, image_width=image_width, image_height=image_height)

            if xyxy.shape[0] < self.max_proposals:
                pad = xyxy[-1:].expand(self.max_proposals - xyxy.shape[0], -1)
                xyxy = torch.cat([xyxy, pad], dim=0)

            packed_boxes.append(xyxy)

        if len(packed_boxes) != batch_size:
            raise RuntimeError("YOLO26 predictor returned a different batch size than input images.")

        return torch.stack(packed_boxes, dim=0).to(dtype=images.dtype)
