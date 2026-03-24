from __future__ import annotations

from typing import Any

import torch


def collate_grounding_batch(batch: list[Any]) -> dict[str, torch.Tensor | list[str]]:
    result: dict[str, Any] = {
        "images": torch.stack([sample.image for sample in batch], dim=0),
        "token_ids": torch.stack([sample.token_ids for sample in batch], dim=0),
        "attention_mask": torch.stack([sample.attention_mask for sample in batch], dim=0),
        "target_box": torch.stack([sample.target_box for sample in batch], dim=0),
        "phrases": [sample.phrase for sample in batch],
        "sources": [sample.source for sample in batch],
    }

    # Stack precomputed tensors if all samples in the batch have them.
    if all(sample.precomputed_proposals is not None for sample in batch):
        result["precomputed_proposals"] = torch.stack(
            [sample.precomputed_proposals for sample in batch], dim=0
        )

    if all(sample.precomputed_verifier_targets is not None for sample in batch):
        result["precomputed_verifier_targets"] = torch.stack(
            [sample.precomputed_verifier_targets for sample in batch], dim=0
        )

    if all(sample.precomputed_verifier_mask is not None for sample in batch):
        result["precomputed_verifier_mask"] = torch.stack(
            [sample.precomputed_verifier_mask for sample in batch], dim=0
        )

    return result


collate_synthetic_batch = collate_grounding_batch
