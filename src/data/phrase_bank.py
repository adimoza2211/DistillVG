from __future__ import annotations

from dataclasses import dataclass
from glob import glob
from pathlib import Path

import torch


@dataclass(frozen=True)
class PhraseRecord:
    base_phrase: str
    augmented_phrases: tuple[str, ...]


def _normalize_phrase(text: str) -> str:
    return " ".join(text.strip().split())


def _extract_phrase_record(item: tuple[object, ...]) -> PhraseRecord | None:
    if len(item) >= 4 and isinstance(item[3], list) and all(isinstance(value, str) for value in item[3]):
        augmented = tuple(_normalize_phrase(value) for value in item[3] if _normalize_phrase(value))
        if augmented:
            return PhraseRecord(base_phrase=augmented[0], augmented_phrases=augmented)

    str_candidates = [element for element in item if isinstance(element, str) and element and not element.lower().endswith(".jpg")]
    if not str_candidates:
        return None
    phrase = _normalize_phrase(str_candidates[0])
    if not phrase:
        return None
    return PhraseRecord(base_phrase=phrase, augmented_phrases=(phrase,))


def load_phrase_bank_from_augmented_pth(pattern: str) -> list[PhraseRecord]:
    records: list[PhraseRecord] = []
    for file_path in sorted(glob(pattern)):
        path = Path(file_path)
        try:
            dataset = torch.load(path, weights_only=False)
        except Exception:
            continue

        if not isinstance(dataset, list):
            continue

        for item in dataset:
            if not isinstance(item, tuple):
                continue
            record = _extract_phrase_record(item)
            if record is not None:
                records.append(record)

    return records
