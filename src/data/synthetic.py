from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass
class SyntheticSample:
    image: torch.Tensor
    token_ids: torch.Tensor
    attention_mask: torch.Tensor
    target_box: torch.Tensor
    verifier_target: torch.Tensor


class SyntheticGroundingDataset(Dataset[SyntheticSample]):
    def __init__(self, num_samples: int, image_size: int, seq_len: int, vocab_size: int) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.image_size = image_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, _: int) -> SyntheticSample:
        image = torch.randn(3, self.image_size, self.image_size)
        token_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        attention_mask = torch.ones(self.seq_len, dtype=torch.long)
        target_box = torch.rand(4)
        verifier_target = torch.randint(0, 2, (1,), dtype=torch.float32).squeeze(0)
        return SyntheticSample(
            image=image,
            token_ids=token_ids,
            attention_mask=attention_mask,
            target_box=target_box,
            verifier_target=verifier_target,
        )
