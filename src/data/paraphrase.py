from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class ParaphraseRecord:
    sample_id: str
    original: str
    paraphrases: list[str]
    mode: str
    model_name: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def generate_rule_based_paraphrases(expression: str) -> list[str]:
    base = expression.strip()
    p1 = f"the object described as: {base}".strip()
    p2 = f"locate: {base}".strip()
    if p1 == p2:
        p2 = f"find this target: {base}".strip()
    return [p1, p2]


def validate_record(record: ParaphraseRecord) -> None:
    if not record.sample_id:
        raise ValueError("sample_id must be non-empty")
    if not record.original.strip():
        raise ValueError("original must be non-empty")
    if len(record.paraphrases) != 2:
        raise ValueError("paraphrases must contain exactly 2 items")
    if any(not item.strip() for item in record.paraphrases):
        raise ValueError("paraphrases must be non-empty strings")
