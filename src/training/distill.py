from __future__ import annotations

import torch

from src.models.losses import LossOutput, compute_distillation_losses
from src.models.verifier.prompting import build_verifier_queries
from src.models.verifier.runtime import BaseOnlineVerifier
from src.training.verifier_crops import crop_and_resize_from_boxes, normalize_predicted_boxes


def run_distillation_step(
    model: torch.nn.Module,
    online_verifier: BaseOnlineVerifier,
    batch: dict[str, torch.Tensor | list[str] | list[tuple[str, ...]]],
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    grad_accum_steps: int,
    loss_weights: dict[str, float],
    verifier_crop_size: int,
    use_augmented_verifier_queries: bool,
    verifier_query_selection: str,
) -> LossOutput:
    images = batch["images"].to(device, non_blocking=True)
    token_ids = batch["token_ids"].to(device, non_blocking=True)
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
    aug_token_ids = batch.get("aug_token_ids")
    aug_attention_mask = batch.get("aug_attention_mask")
    target_box = batch["target_box"].to(device, non_blocking=True)
    phrases = list(batch.get("phrases", []))
    augmented_phrases = batch.get("augmented_phrases")
    if augmented_phrases is None:
        augmented_phrase_sets: list[tuple[str, ...]] | None = None
    else:
        augmented_phrase_sets = list(augmented_phrases)
    teacher_logits = batch.get("logits")
    teacher_proposal_feats = batch.get("proposal_feats")
    teacher_pooled_embed = batch.get("pooled_embed")
    teacher_cross_attn = batch.get("cross_attn")

    if aug_token_ids is not None:
        aug_token_ids = aug_token_ids.to(device, non_blocking=True)
    if aug_attention_mask is not None:
        aug_attention_mask = aug_attention_mask.to(device, non_blocking=True)
    if teacher_logits is not None:
        teacher_logits = teacher_logits.to(device, non_blocking=True)
    else:
        teacher_logits = torch.zeros(images.shape[0], 200, device=device, dtype=images.dtype)
    if teacher_proposal_feats is not None:
        teacher_proposal_feats = teacher_proposal_feats.to(device, non_blocking=True)
    else:
        teacher_proposal_feats = torch.zeros(images.shape[0], 200, 256, device=device, dtype=images.dtype)
    if teacher_pooled_embed is not None:
        teacher_pooled_embed = teacher_pooled_embed.to(device, non_blocking=True)
    else:
        teacher_pooled_embed = torch.zeros(images.shape[0], 512, device=device, dtype=images.dtype)
    if teacher_cross_attn is not None:
        teacher_cross_attn = teacher_cross_attn.to(device, non_blocking=True)
    else:
        teacher_cross_attn = torch.zeros(images.shape[0], 2, 4, 200, token_ids.shape[1], device=device, dtype=images.dtype)

    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        outputs = model(images=images, token_ids=token_ids, attention_mask=attention_mask)
        proposal_boxes = normalize_predicted_boxes(outputs["bbox"].detach())
        crops = crop_and_resize_from_boxes(
            images=images.detach(),
            boxes_xyxy=proposal_boxes,
            output_size=verifier_crop_size,
        )

        with torch.no_grad():
            verifier_queries = build_verifier_queries(
                base_phrases=phrases,
                augmented_phrase_sets=augmented_phrase_sets,
                use_augmented=use_augmented_verifier_queries,
                selection_strategy=verifier_query_selection,
            )
            online_verifier_logits = online_verifier(
                crops=crops,
                queries=verifier_queries,
            )
        verifier_target = torch.sigmoid(online_verifier_logits).to(outputs["verifier_logit"].dtype)

        augmented_student_tokens = None
        if aug_token_ids is not None and aug_attention_mask is not None:
            batch_size, aug_count, seq_len = aug_token_ids.shape
            aug_images = images.unsqueeze(1).expand(-1, aug_count, -1, -1, -1).reshape(batch_size * aug_count, *images.shape[1:])
            aug_token_ids_flat = aug_token_ids.reshape(batch_size * aug_count, seq_len)
            aug_attention_mask_flat = aug_attention_mask.reshape(batch_size * aug_count, seq_len)
            aug_model_outputs = model(images=aug_images, token_ids=aug_token_ids_flat, attention_mask=aug_attention_mask_flat)
            augmented_student_tokens = aug_model_outputs["fused_tokens"].reshape(batch_size, aug_count, *aug_model_outputs["fused_tokens"].shape[1:]).mean(dim=1)

        losses = compute_distillation_losses(
            pred_box=outputs["bbox"],
            target_box=target_box,
            verifier_logit=outputs["verifier_logit"],
            verifier_target=verifier_target,
            student_tokens=outputs["fused_tokens"],
            student_text_tokens=outputs["text_tokens"],
            teacher_logits=teacher_logits,
            teacher_proposal_feats=teacher_proposal_feats,
            teacher_pooled_embed=teacher_pooled_embed,
            teacher_cross_attn=teacher_cross_attn,
            augmented_student_tokens=augmented_student_tokens,
            augmented_student_text_tokens=None,
            lambda_kl=float(loss_weights["lambda_kl"]),
            lambda_l1=float(loss_weights["lambda_l1"]),
            lambda_giou=float(loss_weights["lambda_giou"]),
            lambda_mse=float(loss_weights["lambda_mse"]),
            lambda_attn=float(loss_weights["lambda_attn"]),
            lambda_cst=float(loss_weights["lambda_cst"]),
            lambda_ver=float(loss_weights["lambda_ver"]),
            kl_temperature=float(loss_weights["kl_temperature"]),
        )
        scaled_total = losses.total / grad_accum_steps

    scaler.scale(scaled_total).backward()
    return losses
