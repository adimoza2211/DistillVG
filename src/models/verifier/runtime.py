from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
from omegaconf import DictConfig
from torch import nn


class BaseOnlineVerifier(nn.Module, ABC):
    @abstractmethod
    def forward(self, crops: torch.Tensor, queries: list[str]) -> torch.Tensor:
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


class InternVLOnlineVerifier(BaseOnlineVerifier):
    def __init__(self, model_id: str, device: torch.device) -> None:
        super().__init__()
        try:
            from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig
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
        self.model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            quantization_config=quantization,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    def forward(self, crops: torch.Tensor, queries: list[str]) -> torch.Tensor:
        raise NotImplementedError(
            "InternVL online scoring bridge is scaffolded but not fully wired: True/False token logit extraction from verifier generation outputs is still pending."
        )


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
