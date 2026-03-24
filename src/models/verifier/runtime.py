from __future__ import annotations

import importlib.util
import math
from abc import ABC, abstractmethod
from typing import Any
import warnings

import torch
from omegaconf import DictConfig
from torch import nn
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms.functional import to_pil_image


def _normalize_crop_for_processor(crop: torch.Tensor) -> torch.Tensor:
    crop = crop.detach().float().cpu()
    crop_min = float(crop.min().item())
    crop_max = float(crop.max().item())
    if crop_min < 0.0 or crop_max > 1.0:
        scale = max(crop_max - crop_min, 1e-6)
        crop = (crop - crop_min) / scale
    return crop.clamp(0.0, 1.0)


def _internvl_binary_question(query: str) -> str:
    return (
        "<image>\n"
        "Decide whether this image region matches the text query.\n"
        f"Query: {query}\n"
        "Answer with exactly one word: True or False."
    )


def _internvl_4class_question(query: str) -> str:
    return (
        "<image>\n"
        "Rate how well this image region matches the text query.\n"
        f"Query: {query}\n"
        "Answer with exactly one word from: Excellent, Good, Partial, No."
    )


def _parse_binary_response(response: str) -> float:
    normalized = response.strip().lower()
    if normalized.startswith("true") or normalized.startswith("yes"):
        return 1.0
    if normalized.startswith("false") or normalized.startswith("no"):
        return -1.0
    return 0.0


def _parse_4class_response(response: str) -> int:
    normalized = response.strip().lower()
    if normalized.startswith("excellent"):
        return 0
    if normalized.startswith("good"):
        return 1
    if normalized.startswith("partial"):
        return 2
    if normalized.startswith("no"):
        return 3
    return 3


def _flash_attn_available() -> bool:
    return importlib.util.find_spec("flash_attn") is not None


def _resolve_use_flash_attn(preference: str | bool) -> bool:
    if isinstance(preference, bool):
        requested = preference
        is_explicit_true = preference
    else:
        normalized = preference.strip().lower()
        if normalized == "auto":
            requested = True
            is_explicit_true = False
        elif normalized in {"true", "1", "yes", "y", "on"}:
            requested = True
            is_explicit_true = True
        elif normalized in {"false", "0", "no", "n", "off"}:
            requested = False
            is_explicit_true = False
        else:
            raise ValueError(f"Unsupported verifier.use_flash_attn value: {preference}")

    if not requested:
        return False
    if _flash_attn_available():
        return True
    if is_explicit_true:
        raise RuntimeError(
            "verifier.use_flash_attn=true but flash_attn is not installed. "
            "Install flash_attn or set verifier.use_flash_attn=auto/false."
        )
    warnings.warn("flash_attn is not installed; falling back to non-FlashAttention verifier inference.")
    return False


class BaseOnlineVerifier(nn.Module, ABC):
    @abstractmethod
    def forward(self, crops: torch.Tensor, queries: list[str]) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def score_4class(self, images: torch.Tensor, queries: list[str]) -> torch.Tensor:
        raise NotImplementedError


class InternVLOnlineVerifier(BaseOnlineVerifier):
    def __init__(self, model_id: str, device: torch.device, use_flash_attn: str | bool = "auto") -> None:
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
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
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
        target_device_index = 0 if device.index is None else int(device.index)
        flash_attn_enabled = _resolve_use_flash_attn(use_flash_attn)
        self.model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            quantization_config=quantization,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=flash_attn_enabled,
            device_map={"": target_device_index},
        ).eval()

        model_class = self.model.__class__
        if not hasattr(model_class, "_distillvg_original_extract_feature"):
            setattr(model_class, "_distillvg_original_extract_feature", model_class.extract_feature)

            def _extract_feature_dtype_safe(self_model: Any, pixel_values: torch.Tensor) -> torch.Tensor:
                original_method = getattr(self_model.__class__, "_distillvg_original_extract_feature")
                embeddings = original_method(self_model, pixel_values)
                return embeddings.to(dtype=pixel_values.dtype)

            model_class.extract_feature = _extract_feature_dtype_safe

        self.image_transform = T.Compose(
            [
                T.Lambda(lambda image: image.convert("RGB") if image.mode != "RGB" else image),
                T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        generation_common: dict[str, Any] = {
            "do_sample": False,
        }
        if eos_token_id is not None:
            generation_common["eos_token_id"] = int(eos_token_id)
            generation_common["pad_token_id"] = int(eos_token_id)

        self.binary_generation_config: dict[str, Any] = {
            "max_new_tokens": 1,
            **generation_common,
        }
        self.class_generation_config: dict[str, Any] = {
            "max_new_tokens": 2,
            **generation_common,
        }

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

    def _prepare_pixel_values(self, images: torch.Tensor) -> tuple[torch.Tensor, list[int]]:
        transformed: list[torch.Tensor] = []
        for image in images:
            pil_image = to_pil_image(_normalize_crop_for_processor(image))
            transformed.append(self.image_transform(pil_image))
        pixel_values = torch.stack(transformed, dim=0)
        pixel_values = pixel_values.to(device=self.inference_device, dtype=torch.bfloat16, non_blocking=True)
        num_patches_list = [1] * int(images.shape[0])
        return pixel_values, num_patches_list

    def forward(self, crops: torch.Tensor, queries: list[str]) -> torch.Tensor:
        if len(queries) != crops.shape[0]:
            raise ValueError("Verifier query count must match crop batch size.")

        pixel_values, num_patches_list = self._prepare_pixel_values(crops)
        questions = [_internvl_binary_question(query) for query in queries]
        with torch.no_grad():
            responses = self.model.batch_chat(
                self.tokenizer,
                pixel_values,
                num_patches_list=num_patches_list,
                questions=questions,
                generation_config=self.binary_generation_config,
            )

        scores = torch.tensor([_parse_binary_response(response) for response in responses], device=crops.device)
        return scores

    def score_4class(self, images: torch.Tensor, queries: list[str]) -> torch.Tensor:
        if len(queries) != images.shape[0]:
            raise ValueError("Verifier query count must match image batch size.")

        pixel_values, num_patches_list = self._prepare_pixel_values(images)
        questions = [_internvl_4class_question(query) for query in queries]
        with torch.no_grad():
            responses = self.model.batch_chat(
                self.tokenizer,
                pixel_values,
                num_patches_list=num_patches_list,
                questions=questions,
                generation_config=self.class_generation_config,
            )

        logits = torch.full((len(responses), 4), fill_value=-4.0, device=images.device)
        for index, response in enumerate(responses):
            class_index = _parse_4class_response(response)
            logits[index, class_index] = 4.0
        return logits


def _freeze_verifier(verifier: BaseOnlineVerifier) -> BaseOnlineVerifier:
    verifier.eval()
    for parameter in verifier.parameters():
        parameter.requires_grad = False
    return verifier


def build_online_verifier(cfg: DictConfig, device: torch.device) -> BaseOnlineVerifier:
    backend = str(cfg.backend).lower()
    if backend == "internvl":
        verifier = InternVLOnlineVerifier(
            model_id=str(cfg.model_id),
            device=device,
            use_flash_attn=getattr(cfg, "use_flash_attn", "auto"),
        )
    else:
        raise ValueError(f"Unsupported verifier backend: {cfg.backend}. Only 'internvl' is supported.")

    verifier = _freeze_verifier(verifier)
    return verifier.to(device)
