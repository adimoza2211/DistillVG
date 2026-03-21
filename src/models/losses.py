from __future__ import annotations

from dataclasses import dataclass

import torch
from torchvision.ops import generalized_box_iou


@dataclass
class LossOutput:
    total: torch.Tensor
    ce: torch.Tensor
    kl: torch.Tensor
    l1: torch.Tensor
    giou: torch.Tensor
    mse: torch.Tensor
    attn: torch.Tensor
    cst: torch.Tensor
    ver: torch.Tensor


def _pool_teacher_vector(vector: torch.Tensor, target_dim: int) -> torch.Tensor:
    if vector.shape[-1] == target_dim:
        return vector
    pooled = torch.nn.functional.adaptive_avg_pool1d(vector.unsqueeze(1), target_dim)
    return pooled.squeeze(1)


def _proposal_distribution(logits: torch.Tensor, target_len: int, temperature: float) -> torch.Tensor:
    pooled_logits = torch.nn.functional.adaptive_avg_pool1d(logits.unsqueeze(1), target_len).squeeze(1)
    return torch.softmax(pooled_logits / max(temperature, 1e-6), dim=-1)


def _token_saliency(tokens: torch.Tensor) -> torch.Tensor:
    saliency = tokens.norm(dim=-1)
    return torch.softmax(saliency, dim=-1)


def _box_loss(pred_box: torch.Tensor, target_box: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pred_box = pred_box.sigmoid()
    x1 = torch.minimum(pred_box[..., 0], pred_box[..., 2])
    y1 = torch.minimum(pred_box[..., 1], pred_box[..., 3])
    x2 = torch.maximum(pred_box[..., 0], pred_box[..., 2]).clamp_min(x1 + 1e-4)
    y2 = torch.maximum(pred_box[..., 1], pred_box[..., 3]).clamp_min(y1 + 1e-4)
    pred_box = torch.stack([x1, y1, x2, y2], dim=-1)
    l1 = torch.nn.functional.l1_loss(pred_box, target_box)
    giou = 1.0 - generalized_box_iou(pred_box, target_box).diagonal().mean()
    return l1, giou


def compute_distillation_losses(
    pred_box: torch.Tensor,
    target_box: torch.Tensor,
    verifier_logit: torch.Tensor,
    verifier_target: torch.Tensor,
    student_tokens: torch.Tensor,
    student_text_tokens: torch.Tensor,
    teacher_logits: torch.Tensor,
    teacher_proposal_feats: torch.Tensor,
    teacher_pooled_embed: torch.Tensor,
    teacher_cross_attn: torch.Tensor,
    augmented_student_tokens: torch.Tensor | None = None,
    augmented_student_text_tokens: torch.Tensor | None = None,
    lambda_kl: float = 1.0,
    lambda_l1: float = 1.0,
    lambda_giou: float = 1.0,
    lambda_mse: float = 1.0,
    lambda_attn: float = 1.0,
    lambda_cst: float = 1.0,
    lambda_ver: float = 1.0,
    kl_temperature: float = 1.0,
) -> LossOutput:
    l1, giou = _box_loss(pred_box, target_box)
    ver = torch.nn.functional.binary_cross_entropy_with_logits(verifier_logit, verifier_target)

    teacher_probs = _proposal_distribution(teacher_logits, student_tokens.shape[1], kl_temperature)
    student_probs = _token_saliency(student_tokens)
    kl = torch.nn.functional.kl_div(
        torch.log(student_probs.clamp_min(1e-8)),
        teacher_probs,
        reduction="batchmean",
    ) * (kl_temperature ** 2)

    student_embed = student_tokens.mean(dim=1)
    teacher_embed = _pool_teacher_vector(teacher_pooled_embed, student_embed.shape[-1])
    mse = torch.nn.functional.mse_loss(student_embed, teacher_embed)

    teacher_attn = teacher_cross_attn.mean(dim=(1, 2, 3))
    teacher_attn = torch.nn.functional.adaptive_avg_pool1d(teacher_attn.unsqueeze(1), student_tokens.shape[1]).squeeze(1)
    teacher_attn = teacher_attn / teacher_attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    attn = torch.nn.functional.mse_loss(student_probs, teacher_attn)

    if augmented_student_tokens is not None:
        augmented_probs = _token_saliency(augmented_student_tokens)
        cst = torch.nn.functional.mse_loss(student_probs, augmented_probs)
    else:
        cst = torch.zeros((), device=pred_box.device, dtype=pred_box.dtype)

    ce = ver

    total = (
        lambda_kl * kl
        + lambda_l1 * l1
        + lambda_giou * giou
        + lambda_mse * mse
        + lambda_attn * attn
        + lambda_cst * cst
        + lambda_ver * ver
    )

    return LossOutput(
        total=total,
        ce=ce,
        kl=kl,
        l1=l1,
        giou=giou,
        mse=mse,
        attn=attn,
        cst=cst,
        ver=ver,
    )
