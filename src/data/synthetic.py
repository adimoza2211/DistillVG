from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from src.data.phrase_bank import PhraseRecord


@dataclass
class SyntheticSample:
    image: torch.Tensor
    token_ids: torch.Tensor
    attention_mask: torch.Tensor
    target_box: torch.Tensor
    phrase: str
    augmented_phrases: tuple[str, ...]


def _encode_phrase_to_ids(phrase: str, seq_len: int, vocab_size: int) -> torch.Tensor:
    token_ids = torch.zeros(seq_len, dtype=torch.long)
    words = phrase.lower().split()
    for index, word in enumerate(words[:seq_len]):
        token_ids[index] = abs(hash(word)) % max(vocab_size, 2)
    return token_ids


class SyntheticGroundingDataset(Dataset[SyntheticSample]):
    def __init__(
        self,
        num_samples: int,
        image_size: int,
        seq_len: int,
        vocab_size: int,
        phrase_bank: list[PhraseRecord] | None = None,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.image_size = image_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.phrase_bank = phrase_bank or []

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> SyntheticSample:
        if self.phrase_bank:
            phrase_record = self.phrase_bank[index % len(self.phrase_bank)]
            phrase = phrase_record.base_phrase
            augmented_phrases = phrase_record.augmented_phrases
            token_ids = _encode_phrase_to_ids(phrase=phrase, seq_len=self.seq_len, vocab_size=self.vocab_size)
            attention_mask = (token_ids != 0).long()
            if attention_mask.sum().item() == 0:
                attention_mask[0] = 1
        else:
            phrase = "target object"
            augmented_phrases = (phrase,)
            token_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
            attention_mask = torch.ones(self.seq_len, dtype=torch.long)

        image = torch.randn(3, self.image_size, self.image_size)
        target_box = torch.rand(4)
        return SyntheticSample(
            image=image,
            token_ids=token_ids,
            attention_mask=attention_mask,
            target_box=target_box,
            phrase=phrase,
            augmented_phrases=augmented_phrases,
        )
