from __future__ import annotations

import torch

from src.training.dwbd import DWBDConfig, compute_dwbd_weights
from src.training.optimizers import MuSGD


def test_dwbd_increases_box_weight_and_decreases_verifier_weight_over_epochs() -> None:
    config = DWBDConfig(
        enabled=True,
        verifier_start_scale=1.5,
        verifier_end_scale=1.0,
        box_start_scale=0.75,
        box_end_scale=1.0,
    )
    start = compute_dwbd_weights(
        epoch_index=0,
        total_epochs=10,
        base_lambda_l1=5.0,
        base_lambda_giou=2.0,
        base_lambda_ver=1.0,
        dwbd=config,
    )
    end = compute_dwbd_weights(
        epoch_index=9,
        total_epochs=10,
        base_lambda_l1=5.0,
        base_lambda_giou=2.0,
        base_lambda_ver=1.0,
        dwbd=config,
    )
    assert start["lambda_ver"] > end["lambda_ver"]
    assert start["lambda_l1"] < end["lambda_l1"]
    assert start["lambda_giou"] < end["lambda_giou"]


def test_musgd_updates_parameter() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = MuSGD([parameter], lr=0.1, momentum=0.9, muon_beta=0.95)
    loss = parameter.pow(2).sum()
    loss.backward()
    before = parameter.detach().clone()
    optimizer.step()
    after = parameter.detach().clone()
    assert not torch.allclose(before, after)
