import torch

from src.models.verifier.runtime import _parse_4class_response, _parse_binary_response, _resolve_use_flash_attn
from src.training.trainer import Trainer


def test_parse_binary_response_true_false() -> None:
    assert _parse_binary_response("True") == 1.0
    assert _parse_binary_response("yes, this matches") == 1.0
    assert _parse_binary_response("False") == -1.0
    assert _parse_binary_response("no") == -1.0


def test_parse_binary_response_unknown_defaults_neutral() -> None:
    assert _parse_binary_response("maybe") == 0.0


def test_parse_4class_response_labels() -> None:
    assert _parse_4class_response("Excellent") == 0
    assert _parse_4class_response("good") == 1
    assert _parse_4class_response("Partial match") == 2
    assert _parse_4class_response("No") == 3


def test_parse_4class_response_unknown_defaults_no() -> None:
    assert _parse_4class_response("uncertain") == 3


def test_flash_attn_preference_off() -> None:
    assert _resolve_use_flash_attn("off") is False


def test_trainer_amp_dtype_parsing() -> None:
    assert Trainer._resolve_amp_dtype("bf16") == torch.bfloat16
    assert Trainer._resolve_amp_dtype("fp16") == torch.float16
