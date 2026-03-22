from __future__ import annotations

import torch

from src.losses.box_losses import box_loss
from src.losses.combined import GroundingLoss
from src.losses.consistency import consistency_loss
from src.losses.dwbd import DWBDLoss, compute_dwbd_alpha
from src.losses.stal_progloss import compute_progloss_scale, compute_stal_weight
from src.losses.verifier_kl import verifier_kl_loss
from src.models.student.verifier import VerifierHead


def test_verifier_kl_loss_is_finite() -> None:
    student_logits = torch.randn(2, 5, 4)
    teacher_soft = torch.softmax(torch.randn(2, 5, 4), dim=-1)
    mask = torch.ones(2, 5, dtype=torch.bool)
    loss = verifier_kl_loss(student_logits, teacher_soft, mask, temperature=2.0)
    assert torch.isfinite(loss)


def test_dwbd_alpha_decays_over_epochs() -> None:
    alpha_start = compute_dwbd_alpha(epoch_index=0, total_epochs=10, alpha_max=0.8, gamma=2.0)
    alpha_end = compute_dwbd_alpha(epoch_index=9, total_epochs=10, alpha_max=0.8, gamma=2.0)
    assert alpha_start > alpha_end


def test_dwbd_loss_is_finite() -> None:
    loss_fn = DWBDLoss()
    student_logits = torch.randn(2, 5, 4)
    teacher_soft = torch.softmax(torch.randn(2, 5, 4), dim=-1)
    gt_one_hot = torch.zeros(2, 5, 4)
    gt_one_hot[..., 0] = 1.0
    loss = loss_fn(student_logits, teacher_soft, gt_one_hot, alpha=0.5)
    assert torch.isfinite(loss)


def test_consistency_loss_is_zero_for_single_map() -> None:
    single = [torch.randn(2, 6, 150, 77)]
    loss = consistency_loss(single)
    assert torch.isfinite(loss)
    assert float(loss.item()) == 0.0


def test_box_loss_is_finite() -> None:
    pred = torch.randn(2, 4)
    gt = torch.rand(2, 4)
    loss = box_loss(pred, gt)
    assert torch.isfinite(loss)


def test_grounding_loss_and_verifier_ordinal_score() -> None:
    loss_fn = GroundingLoss(lambda1=0.1, lambda2=1.0, lambda3=5.0)
    student_logits = torch.randn(2, 5, 4)
    teacher_soft = torch.softmax(torch.randn(2, 5, 4), dim=-1)
    gt_one_hot = torch.zeros(2, 5, 4)
    gt_one_hot[..., 1] = 1.0
    mask = torch.ones(2, 5, dtype=torch.bool)
    pred_boxes = torch.randn(2, 4)
    gt_boxes = torch.rand(2, 4)
    out = loss_fn(
        student_logits=student_logits,
        teacher_soft=teacher_soft,
        gt_one_hot=gt_one_hot,
        alpha=0.7,
        verifier_mask=mask,
        pred_boxes=pred_boxes,
        gt_boxes=gt_boxes,
        attention_maps=[torch.randn(2, 6, 150, 77), torch.randn(2, 6, 150, 77)],
    )
    assert torch.isfinite(out.total)

    verifier = VerifierHead(hidden_dim=32, num_classes=4)
    features = torch.randn(2, 5, 32)
    text = torch.randn(2, 32)
    logits = verifier(features, text)
    score = VerifierHead.score(logits)
    assert logits.shape == (2, 5, 4)
    assert score.shape == (2, 5)


def test_stal_weight_boosts_smaller_targets() -> None:
    boxes = torch.tensor(
        [
            [0.0, 0.0, 0.08, 0.08],
            [0.1, 0.1, 0.9, 0.9],
        ],
        dtype=torch.float32,
    )
    weight = compute_stal_weight(
        boxes,
        area_threshold=0.05,
        max_boost=2.0,
        enabled=True,
    )
    assert float(weight.item()) > 1.0


def test_progloss_scale_progresses_linearly() -> None:
    start = compute_progloss_scale(
        epoch_index=0,
        total_epochs=10,
        start_scale=1.2,
        end_scale=0.8,
        power=1.0,
        enabled=True,
    )
    end = compute_progloss_scale(
        epoch_index=9,
        total_epochs=10,
        start_scale=1.2,
        end_scale=0.8,
        power=1.0,
        enabled=True,
    )
    assert start > end
