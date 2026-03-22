from __future__ import annotations

from dataclasses import dataclass
from glob import glob
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


@dataclass(frozen=True)
class GroundingSample:
    image: torch.Tensor
    token_ids: torch.Tensor
    attention_mask: torch.Tensor
    target_box: torch.Tensor
    phrase: str
    augmented_phrases: tuple[str, ...]


@dataclass(frozen=True)
class GroundingRecord:
    image_name: str
    box_xywh: tuple[float, float, float, float]
    phrases: tuple[str, ...]
    source: str = "unknown"


def _encode_phrase_to_ids(phrase: str, seq_len: int, vocab_size: int) -> torch.Tensor:
    token_ids = torch.zeros(seq_len, dtype=torch.long)
    words = phrase.lower().split()
    for index, word in enumerate(words[:seq_len]):
        token_ids[index] = abs(hash(word)) % max(vocab_size, 2)
    return token_ids


def _normalize_phrase(text: str) -> str:
    return " ".join(text.strip().split())


def _record_from_tuple(item: tuple[object, ...]) -> GroundingRecord | None:
    if len(item) < 4:
        return None
    image_name = item[0]
    box_xywh = item[2]
    phrase_list = item[3]

    if not isinstance(image_name, str):
        return None
    if not isinstance(box_xywh, list) or len(box_xywh) != 4:
        return None
    if not isinstance(phrase_list, list) or not all(isinstance(value, str) for value in phrase_list):
        return None

    normalized_phrases = tuple(_normalize_phrase(value) for value in phrase_list if _normalize_phrase(value))
    if not normalized_phrases:
        return None

    return GroundingRecord(
        image_name=image_name,
        box_xywh=(float(box_xywh[0]), float(box_xywh[1]), float(box_xywh[2]), float(box_xywh[3])),
        phrases=normalized_phrases,
    )


def load_grounding_records(pattern: str) -> list[GroundingRecord]:
    records: list[GroundingRecord] = []
    for file_path in sorted(glob(pattern)):
        path = Path(file_path)
        source_name = path.parent.name or "unknown"
        try:
            payload = torch.load(path, weights_only=False)
        except Exception:
            continue

        if not isinstance(payload, list):
            continue

        for item in payload:
            if not isinstance(item, tuple):
                continue
            record = _record_from_tuple(item)
            if record is not None:
                records.append(
                    GroundingRecord(
                        image_name=record.image_name,
                        box_xywh=record.box_xywh,
                        phrases=record.phrases,
                        source=source_name,
                    )
                )

    return records


class AugmentedGroundingDataset(Dataset[GroundingSample]):
    def __init__(
        self,
        records: list[GroundingRecord],
        image_root: str,
        image_size: int,
        seq_len: int,
        vocab_size: int,
    ) -> None:
        super().__init__()
        if not records:
            raise ValueError("No grounding records were loaded. Cannot build dataset.")

        self.records = records
        self.image_root = Path(image_root)
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> GroundingSample:
        record = self.records[index]
        image_path = self.image_root / record.image_name
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        image_tensor = self.transform(image)

        x, y, w, h = record.box_xywh
        x1 = max(0.0, min(x, width - 1.0))
        y1 = max(0.0, min(y, height - 1.0))
        x2 = max(x1 + 1e-4, min(x + w, width))
        y2 = max(y1 + 1e-4, min(y + h, height))
        target_box = torch.tensor([x1 / width, y1 / height, x2 / width, y2 / height], dtype=torch.float32)

        phrase = record.phrases[0]
        token_ids = _encode_phrase_to_ids(phrase=phrase, seq_len=self.seq_len, vocab_size=self.vocab_size)
        attention_mask = (token_ids != 0).long()
        if int(attention_mask.sum().item()) == 0:
            attention_mask[0] = 1

        return GroundingSample(
            image=image_tensor,
            token_ids=token_ids,
            attention_mask=attention_mask,
            target_box=target_box,
            phrase=phrase,
            augmented_phrases=record.phrases,
        )
