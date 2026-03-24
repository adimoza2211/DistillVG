from __future__ import annotations

from typing import Any

import torch


def collate_grounding_batch(batch: list[Any]) -> dict[str, torch.Tensor | list[str]]:
    return {
        "images": torch.stack([sample.image for sample in batch], dim=0),
        "token_ids": torch.stack([sample.token_ids for sample in batch], dim=0),
        "attention_mask": torch.stack([sample.attention_mask for sample in batch], dim=0),
        "target_box": torch.stack([sample.target_box for sample in batch], dim=0),
        "phrases": [sample.phrase for sample in batch],
        "sources": [sample.source for sample in batch],
    }


collate_synthetic_batch = collate_grounding_batch
