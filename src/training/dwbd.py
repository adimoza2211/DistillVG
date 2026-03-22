from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DWBDConfig:
    enabled: bool
    verifier_start_scale: float
    verifier_end_scale: float
    box_start_scale: float
    box_end_scale: float


def _linear_interpolate(start: float, end: float, progress: float) -> float:
    return start + (end - start) * progress


def compute_dwbd_weights(
    *,
    epoch_index: int,
    total_epochs: int,
    base_lambda_l1: float,
    base_lambda_giou: float,
    base_lambda_ver: float,
    dwbd: DWBDConfig,
) -> dict[str, float]:
    if not dwbd.enabled:
        return {
            "lambda_l1": base_lambda_l1,
            "lambda_giou": base_lambda_giou,
            "lambda_ver": base_lambda_ver,
        }

    if total_epochs <= 1:
        progress = 1.0
    else:
        progress = max(0.0, min(1.0, epoch_index / float(total_epochs - 1)))

    verifier_scale = _linear_interpolate(dwbd.verifier_start_scale, dwbd.verifier_end_scale, progress)
    box_scale = _linear_interpolate(dwbd.box_start_scale, dwbd.box_end_scale, progress)

    return {
        "lambda_l1": base_lambda_l1 * box_scale,
        "lambda_giou": base_lambda_giou * box_scale,
        "lambda_ver": base_lambda_ver * verifier_scale,
    }
