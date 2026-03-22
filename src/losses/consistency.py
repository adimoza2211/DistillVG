from __future__ import annotations

import torch
import torch.nn.functional as F


def consistency_loss(attn_maps: list[torch.Tensor]) -> torch.Tensor:
    if len(attn_maps) <= 1:
        device = attn_maps[0].device if attn_maps else torch.device("cpu")
        return torch.zeros((), device=device)

    losses: list[torch.Tensor] = []
    flattened = [tensor.flatten(start_dim=1) for tensor in attn_maps]
    for i in range(len(flattened)):
        for j in range(i + 1, len(flattened)):
            cos = F.cosine_similarity(flattened[i], flattened[j], dim=-1)
            losses.append((1.0 - cos).mean())

    return torch.stack(losses).mean() if losses else torch.zeros((), device=flattened[0].device)
