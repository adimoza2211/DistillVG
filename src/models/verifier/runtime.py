from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

import torch
from omegaconf import DictConfig
from torch import nn
from torchvision.transforms.functional import to_pil_image


def _normalize_crop_for_processor(crop: torch.Tensor) -> torch.Tensor:
    crop = crop.detach().float().cpu()
    crop_min = float(crop.min().item())
    crop_max = float(crop.max().item())
    if crop_min < 0.0 or crop_max > 1.0:
        scale = max(crop_max - crop_min, 1e-6)
        crop = (crop - crop_min) / scale
    return crop.clamp(0.0, 1.0)


def _collect_single_token_ids(tokenizer: Any, candidates: list[str]) -> list[int]:
    token_ids: list[int] = []
    for candidate in candidates:
        encoded = tokenizer.encode(candidate, add_special_tokens=False)
        if len(encoded) == 1:
            token_ids.append(int(encoded[0]))
    unique = sorted(set(token_ids))
    if not unique:
        raise RuntimeError(f"None of the candidate labels are single-token for this tokenizer: {candidates}")
    return unique


def _group_score(next_token_logits: torch.Tensor, token_ids: list[int]) -> torch.Tensor:
    token_logits = next_token_logits[:, token_ids]
    return torch.logsumexp(token_logits, dim=-1)


class BaseOnlineVerifier(nn.Module, ABC):
    @abstractmethod
    def forward(self, crops: torch.Tensor, queries: list[str]) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def score_4class(self, images: torch.Tensor, queries: list[str]) -> torch.Tensor:
        raise NotImplementedError


class MockOnlineVerifier(BaseOnlineVerifier):
    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(17)
        image_proj = torch.randn(3, hidden_dim, generator=generator) / math.sqrt(3.0)
        text_proj = torch.randn(2, hidden_dim, generator=generator) / math.sqrt(2.0)
        self.register_buffer("image_proj", image_proj)
        self.register_buffer("text_proj", text_proj)

    def forward(self, crops: torch.Tensor, queries: list[str]) -> torch.Tensor:
        if len(queries) != crops.shape[0]:
            raise ValueError("Verifier query count must match crop batch size.")

        pooled_images = crops.mean(dim=(2, 3))
        query_lens = torch.tensor([len(query.split()) for query in queries], device=crops.device, dtype=crops.dtype)
        query_chars = torch.tensor([len(query) for query in queries], device=crops.device, dtype=crops.dtype)
        text_stats = torch.stack([query_lens, query_chars.sqrt()], dim=-1)

        image_features = pooled_images @ self.image_proj
        text_features = text_stats @ self.text_proj
        logits = (image_features * text_features).sum(dim=-1) / math.sqrt(float(image_features.shape[-1]))
        return logits

    def score_4class(self, images: torch.Tensor, queries: list[str]) -> torch.Tensor:
        binary = self.forward(images, queries)
        p = torch.sigmoid(binary)
        labels = torch.stack(
            [
                0.7 * p + 0.01 * (1.0 - p),
                0.2 * p + 0.04 * (1.0 - p),
                0.08 * p + 0.15 * (1.0 - p),
                0.02 * p + 0.80 * (1.0 - p),
            ],
            dim=-1,
        )
        return torch.log(labels.clamp_min(1e-6))


class InternVLOnlineVerifier(BaseOnlineVerifier):
    def __init__(self, model_id: str, device: torch.device) -> None:
        super().__init__()
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError(
                "InternVL verifier backend requires transformers and bitsandbytes. Install optional dependencies first."
            ) from exc

        if device.type != "cuda":
            raise RuntimeError("InternVL verifier backend currently requires a CUDA device for 4-bit loading.")

        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            quantization_config=quantization,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.tokenizer = getattr(self.processor, "tokenizer", None)
        if self.tokenizer is None:
            raise RuntimeError("InternVL processor did not expose a tokenizer required for True/False scoring.")

        self.true_token_ids = _collect_single_token_ids(self.tokenizer, ["True", " true", "Yes", " yes"])
        self.false_token_ids = _collect_single_token_ids(self.tokenizer, ["False", " false", "No", " no"])
        self.excellent_token_ids = _collect_single_token_ids(self.tokenizer, ["Excellent", " excellent"])
        self.good_token_ids = _collect_single_token_ids(self.tokenizer, ["Good", " good"])
        self.partial_token_ids = _collect_single_token_ids(self.tokenizer, ["Partial", " partial"])
        self.no_token_ids = _collect_single_token_ids(self.tokenizer, ["No", " no"])

        device_map = getattr(self.model, "hf_device_map", None)
        if isinstance(device_map, dict) and device_map:
            first_device = next(iter(device_map.values()))
            if isinstance(first_device, str):
                self.inference_device = torch.device(first_device)
            elif isinstance(first_device, int):
                self.inference_device = torch.device(f"cuda:{first_device}")
            else:
                self.inference_device = device
        else:
            self.inference_device = device

    def forward(self, crops: torch.Tensor, queries: list[str]) -> torch.Tensor:
        if len(queries) != crops.shape[0]:
            raise ValueError("Verifier query count must match crop batch size.")

        pil_crops = [to_pil_image(_normalize_crop_for_processor(crop)) for crop in crops]
        inputs = self.processor(
            images=pil_crops,
            text=queries,
            padding=True,
            return_tensors="pt",
        )
        moved_inputs = {
            key: value.to(self.inference_device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }

        outputs = self.model(**moved_inputs)
        next_token_logits = outputs.logits[:, -1, :].float()
        true_score = _group_score(next_token_logits, self.true_token_ids)
        false_score = _group_score(next_token_logits, self.false_token_ids)
        return true_score - false_score

    def score_4class(self, images: torch.Tensor, queries: list[str]) -> torch.Tensor:
        if len(queries) != images.shape[0]:
            raise ValueError("Verifier query count must match image batch size.")

        pil_images = [to_pil_image(_normalize_crop_for_processor(image)) for image in images]
        inputs = self.processor(
            images=pil_images,
            text=queries,
            padding=True,
            return_tensors="pt",
        )
        moved_inputs = {
            key: value.to(self.inference_device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }

        outputs = self.model(**moved_inputs)
        next_token_logits = outputs.logits[:, -1, :].float()
        class_logits = torch.stack(
            [
                _group_score(next_token_logits, self.excellent_token_ids),
                _group_score(next_token_logits, self.good_token_ids),
                _group_score(next_token_logits, self.partial_token_ids),
                _group_score(next_token_logits, self.no_token_ids),
            ],
            dim=-1,
        )
        return class_logits


def _freeze_verifier(verifier: BaseOnlineVerifier) -> BaseOnlineVerifier:
    verifier.eval()
    for parameter in verifier.parameters():
        parameter.requires_grad = False
    return verifier


def build_online_verifier(cfg: DictConfig, device: torch.device) -> BaseOnlineVerifier:
    backend = str(cfg.backend).lower()
    if backend == "mock":
        verifier = MockOnlineVerifier(hidden_dim=int(cfg.mock_hidden_dim))
    elif backend == "internvl":
        verifier = InternVLOnlineVerifier(model_id=str(cfg.model_id), device=device)
    else:
        raise ValueError(f"Unsupported verifier backend: {cfg.backend}")

    verifier = _freeze_verifier(verifier)
    return verifier.to(device)
