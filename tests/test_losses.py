from __future__ import annotations

import torch

from src.models.losses import compute_distillation_losses


def test_compute_distillation_losses_returns_finite_values() -> None:
    batch_size = 2
    token_count = 6
    proposal_count = 200
    pred_box = torch.randn(batch_size, 4)
    target_box = torch.rand(batch_size, 4)
    verifier_logit = torch.randn(batch_size)
    verifier_target = torch.ones(batch_size)
    student_tokens = torch.randn(batch_size, token_count, 32)
    student_text_tokens = torch.randn(batch_size, token_count, 32)
    teacher_logits = torch.randn(batch_size, proposal_count)
    teacher_proposal_feats = torch.randn(batch_size, proposal_count, 256)
    teacher_pooled_embed = torch.randn(batch_size, 512)
    teacher_cross_attn = torch.randn(batch_size, 2, 4, proposal_count, token_count)

    losses = compute_distillation_losses(
        pred_box=pred_box,
        target_box=target_box,
        verifier_logit=verifier_logit,
        verifier_target=verifier_target,
        student_tokens=student_tokens,
        student_text_tokens=student_text_tokens,
        teacher_logits=teacher_logits,
        teacher_proposal_feats=teacher_proposal_feats,
        teacher_pooled_embed=teacher_pooled_embed,
        teacher_cross_attn=teacher_cross_attn,
        kl_temperature=2.0,
    )

    assert torch.isfinite(losses.total)
    assert torch.isfinite(losses.kl)
    assert torch.isfinite(losses.giou)
    assert torch.isfinite(losses.mse)
    assert torch.isfinite(losses.attn)
