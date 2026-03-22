from __future__ import annotations

import torch

from src.models.student.model import StudentModel


def test_student_model_forward_shapes() -> None:
    model = StudentModel(
        hidden_dim=64,
        vocab_size=128,
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
