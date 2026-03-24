from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from glob import glob
from pathlib import Path
import re

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
    source: str
    precomputed_proposals: torch.Tensor | None = None
    precomputed_verifier_targets: torch.Tensor | None = None
    precomputed_verifier_mask: torch.Tensor | None = None


@dataclass(frozen=True)
class GroundingRecord:
    image_name: str
    box_xywh: tuple[float, float, float, float]
    phrases: tuple[str, ...]
    source: str = "unknown"


def _ensure_image_filename(image_name: str) -> str:
    if image_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
        return image_name
    return f"{image_name}.jpg"


def _resolve_coco_name(image_name: str) -> str:
    if image_name.startswith("COCO_"):
        return image_name

    image_stem = image_name.rsplit(".", 1)[0]
    match = re.search(r"\d+", image_stem)
    if match is None:
        return _ensure_image_filename(image_name)

    numeric_id = int(match.group(0))
    return f"COCO_train2014_{numeric_id:012d}.jpg"


# TokenizeFn signature: (phrase: str) -> tuple[token_ids: Tensor, attention_mask: Tensor]
TokenizeFn = Callable[[str], tuple[torch.Tensor, torch.Tensor]]


def _normalize_phrase(text: str) -> str:
    return " ".join(text.strip().split())


def _record_from_tuple(item: tuple[object, ...]) -> list[GroundingRecord]:
    """Parse tuple and expand into one record per phrase (for augmented phrases)."""
    if len(item) < 4:
        return []
    image_name = item[0]
    box_xywh = item[2]
    phrase_field = item[3]

    if not isinstance(image_name, str):
        return []
    if not isinstance(box_xywh, list) or len(box_xywh) != 4:
        return []
    
    phrases: list[str] = []
    if isinstance(phrase_field, str):
        normalized = _normalize_phrase(phrase_field)
        if normalized:
            phrases.append(normalized)
    elif isinstance(phrase_field, list) and all(isinstance(value, str) for value in phrase_field):
        phrases = [_normalize_phrase(value) for value in phrase_field if _normalize_phrase(value)]
    else:
        return []

    if not phrases:
        return []

    records = [
        GroundingRecord(
            image_name=image_name,
            box_xywh=(float(box_xywh[0]), float(box_xywh[1]), float(box_xywh[2]), float(box_xywh[3])),
            phrases=(phrase,),  # Single phrase per record
        )
        for phrase in phrases
    ]
    return records


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
            expanded_records = _record_from_tuple(item)
            for record in expanded_records:
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
        resize_long_edge: int,
        padded_square_size: int,
        tokenize_fn: TokenizeFn,
        source_image_roots: dict[str, str] | None = None,
        source_image_styles: dict[str, str] | None = None,
        precomputed_cache: dict | None = None,
    ) -> None:
        super().__init__()
        if not records:
            raise ValueError("No grounding records were loaded. Cannot build dataset.")

        self.records = records
        self.image_root = Path(image_root)
        self.source_image_roots = {key: Path(value) for key, value in (source_image_roots or {}).items()}
        self.source_image_styles = {key: value.lower() for key, value in (source_image_styles or {}).items()}
        self.tokenize_fn = tokenize_fn
        self.resize_long_edge = int(resize_long_edge)
        self.padded_square_size = int(padded_square_size)
        self.to_tensor = transforms.ToTensor()

        # Precomputed data caches (populated by precompute_targets.py).
        self._proposals_cache: dict[str, torch.Tensor] = {}
        self._targets_cache: dict[tuple[str, str], torch.Tensor] = {}
        self._masks_cache: dict[tuple[str, str], torch.Tensor] = {}
        if precomputed_cache is not None:
            self._proposals_cache = precomputed_cache.get("proposals", {})
            self._targets_cache = precomputed_cache.get("targets", {})
            self._masks_cache = precomputed_cache.get("masks", {})

    def __len__(self) -> int:
        return len(self.records)

    def _resize_and_pad_image(self, image: Image.Image) -> tuple[torch.Tensor, float, int, int]:
        width, height = image.size
        long_edge = max(width, height)
        effective_long_edge = min(self.resize_long_edge, self.padded_square_size)
        scale = float(effective_long_edge) / float(max(long_edge, 1))

        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        resized = image.resize((resized_width, resized_height), Image.BILINEAR)

        canvas = Image.new("RGB", (self.padded_square_size, self.padded_square_size), color=(0, 0, 0))
        pad_x = max((self.padded_square_size - resized_width) // 2, 0)
        pad_y = max((self.padded_square_size - resized_height) // 2, 0)
        canvas.paste(resized, (pad_x, pad_y))

        return self.to_tensor(canvas), scale, pad_x, pad_y

    def _resolve_image_path(self, record: GroundingRecord) -> Path:
        root = self.source_image_roots.get(record.source, self.image_root)
        style = self.source_image_styles.get(record.source, "default")

        if style == "coco":
            image_name = _resolve_coco_name(record.image_name)
        elif style == "referit":
            image_name = _ensure_image_filename(record.image_name)
        else:
            image_name = _ensure_image_filename(record.image_name)

        return root / image_name

    def __getitem__(self, index: int) -> GroundingSample:
        record = self.records[index]
        image_path = self._resolve_image_path(record)
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        image_tensor, scale, pad_x, pad_y = self._resize_and_pad_image(image)

        x, y, w, h = record.box_xywh
        x1 = max(0.0, min(x, width - 1.0))
        y1 = max(0.0, min(y, height - 1.0))
        x2 = max(x1 + 1e-4, min(x + w, width))
        y2 = max(y1 + 1e-4, min(y + h, height))

        x1_canvas = x1 * scale + pad_x
        y1_canvas = y1 * scale + pad_y
        x2_canvas = x2 * scale + pad_x
        y2_canvas = y2 * scale + pad_y
        canvas_size = float(self.padded_square_size)
        target_box = torch.tensor(
            [x1_canvas / canvas_size, y1_canvas / canvas_size, x2_canvas / canvas_size, y2_canvas / canvas_size],
            dtype=torch.float32,
        )

        phrase = record.phrases[0]
        token_ids, attention_mask = self.tokenize_fn(phrase)

        # Lookup precomputed data if available.
        precomputed_proposals = self._proposals_cache.get(record.image_name)
        cache_key = (record.image_name, phrase)
        precomputed_targets = self._targets_cache.get(cache_key)
        precomputed_mask = self._masks_cache.get(cache_key)

        return GroundingSample(
            image=image_tensor,
            token_ids=token_ids,
            attention_mask=attention_mask,
            target_box=target_box,
            phrase=phrase,
            source=record.source,
            precomputed_proposals=precomputed_proposals,
            precomputed_verifier_targets=precomputed_targets,
            precomputed_verifier_mask=precomputed_mask,
        )
