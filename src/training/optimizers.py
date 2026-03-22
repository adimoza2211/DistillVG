from __future__ import annotations

from typing import Iterable

import torch
from torch.optim import Optimizer


class MuSGD(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        muon_beta: float = 0.95,
        eps: float = 1e-8,
    ) -> None:
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "muon_beta": muon_beta,
            "eps": eps,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            momentum = float(group["momentum"])
            weight_decay = float(group["weight_decay"])
            muon_beta = float(group["muon_beta"])
            eps = float(group["eps"])

            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("MuSGD does not support sparse gradients")

                if weight_decay != 0.0:
                    grad = grad.add(param, alpha=weight_decay)

                state = self.state[param]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(param)
                    state["muon_norm_ema"] = torch.zeros((), device=param.device, dtype=param.dtype)

                momentum_buffer = state["momentum_buffer"]
                muon_norm_ema = state["muon_norm_ema"]

                momentum_buffer.mul_(momentum).add_(grad, alpha=1.0 - momentum)

                grad_norm = momentum_buffer.norm()
                muon_norm_ema.mul_(muon_beta).add_(grad_norm, alpha=1.0 - muon_beta)
                denom = muon_norm_ema.clamp_min(eps)
                stabilized_step = momentum_buffer / denom

                param.add_(stabilized_step, alpha=-lr)

        return loss
