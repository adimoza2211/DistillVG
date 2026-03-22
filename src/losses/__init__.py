from __future__ import annotations

from src.losses.box_losses import box_loss
from src.losses.combined import GroundingLoss
from src.losses.consistency import consistency_loss
from src.losses.dwbd import DWBDLoss, compute_dwbd_alpha
from src.losses.stal_progloss import compute_progloss_scale, compute_stal_weight
from src.losses.verifier_kl import verifier_kl_loss

__all__ = [
    "box_loss",
    "GroundingLoss",
    "consistency_loss",
    "DWBDLoss",
    "compute_dwbd_alpha",
    "compute_stal_weight",
    "compute_progloss_scale",
    "verifier_kl_loss",
]
