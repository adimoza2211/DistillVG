from __future__ import annotations

import torch
import torch.nn.functional as F


def verifier_kl_loss(
    student_logits: torch.Tensor,
    teacher_soft_labels: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    student_log_probs = F.log_softmax(student_logits / max(temperature, 1e-6), dim=-1)
    teacher_probs = teacher_soft_labels.clamp(min=1e-8)
    kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(-1)
    weights = mask.float()
    normalizer = weights.sum().clamp_min(1.0)
    return (kl * weights).sum() / normalizer
