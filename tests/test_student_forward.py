from __future__ import annotations

import torch

from src.models.student.model import StudentModel


def test_student_model_forward_shapes() -> None:
    model = StudentModel(
        hidden_dim=64,
        fusion_layers=2,
        attention_heads=4,
        vocab_size=128,
        roi_tokens=48,
        tob_tokens=32,
        proposal_count=40,
        use_yolo26_proposals=False,
        yolo26_model_cfg="",
        yolo26_weights_path="",
        yolo26_conf_threshold=0.001,
        yolo26_iou_threshold=0.95,
    )
    images = torch.randn(2, 3, 64, 64)
    token_ids = torch.randint(0, 128, (2, 12))
    attention_mask = torch.ones(2, 12, dtype=torch.long)

    outputs = model(images=images, token_ids=token_ids, attention_mask=attention_mask)

    assert outputs["bbox"].shape == (2, 4)
    assert outputs["verifier_logit"].shape == (2,)
    assert outputs["proposal_boxes"].shape == (2, 40, 4)
    assert outputs["proposal_boxes"].shape[2] == 4
    assert outputs["proposal_verifier_logits"].shape[:2] == outputs["proposal_boxes"].shape[:2]
    assert outputs["fused_tokens"].shape[1] <= 32


def test_student_model_applies_roi_token_budget_before_fusion() -> None:
    roi_tokens = 56
    model = StudentModel(
        hidden_dim=64,
        fusion_layers=2,
        attention_heads=4,
        vocab_size=128,
        roi_tokens=roi_tokens,
        tob_tokens=32,
        proposal_count=40,
        use_yolo26_proposals=False,
        yolo26_model_cfg="",
        yolo26_weights_path="",
        yolo26_conf_threshold=0.001,
        yolo26_iou_threshold=0.95,
    )
    images = torch.randn(1, 3, 64, 64)
    token_ids = torch.randint(0, 128, (1, 12))
    attention_mask = torch.ones(1, 12, dtype=torch.long)

    captured: dict[str, int] = {}

    def _fusion_pre_hook(_module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        visual_tokens = args[0]
        captured["fusion_input_tokens"] = int(visual_tokens.shape[1])

    handle = model.fusion.register_forward_pre_hook(_fusion_pre_hook)
    _ = model(images=images, token_ids=token_ids, attention_mask=attention_mask)
    handle.remove()

    assert captured["fusion_input_tokens"] == roi_tokens
