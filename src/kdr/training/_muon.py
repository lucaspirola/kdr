"""Vendored Muon optimizer (single-device variant).

Source: https://github.com/KellerJordan/Muon/blob/master/muon.py
Commit: 6399c658d3c4a3356ba823fa6664b10e23871068 (2026-01-19)
License: MIT (Keller Jordan, 2024)

Reference: https://kellerjordan.github.io/posts/muon/

Vendored as ~100 LOC rather than added as an external dependency because:
- The reference implementation is small and stable.
- The PyPI package's API has churned around the AuxAdam variants.
- We want full audit of every line that touches grads in the training hot path.

Only the single-device variant is vendored; the DDP `all_gather` branches are
dropped because the kdr training loop runs single-process under accelerate
(and `muon_with_adamw` hard-errors under DeepSpeed-ZeRO-3 anyway — see
`build_optimizer`). Constructor exposes `ns_steps` and `nesterov` as kwargs
so the kdr `DistillationConfig` can plumb them through.

Muon is intended only for 2D hidden weight matrices. Embeddings, the LM head,
biases, RMSNorm scales, and routers should be optimized with AdamW; the
`classify_params` helper in `optim.py` enforces this split.

# REQ: LLR-0010
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import nn


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int) -> torch.Tensor:
    """Newton-Schulz iteration to compute the zeroth-power / orthogonalization of G.

    Quintic iteration with coefficients chosen to maximize the slope at zero;
    this trades exact UV^T (from the SVD G = USV^T) for a numerically stable
    bfloat16-friendly approximation US'V^T with S'_{ii} ~ Uniform(0.5, 1.5).
    Empirically this does not hurt model performance vs the exact UV^T.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X: torch.Tensor = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(
    grad: torch.Tensor,
    momentum: torch.Tensor,
    beta: float = 0.95,
    ns_steps: int = 5,
    nesterov: bool = True,
) -> torch.Tensor:
    """One Muon update step: momentum EMA → optional Nesterov → orthogonalize."""
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
    return update


class SingleDeviceMuon(torch.optim.Optimizer):
    """Muon - MomentUm Orthogonalized by Newton-schulz (single-device variant).

    Internally runs standard SGD-momentum, then replaces each 2D parameter's
    update with the nearest orthogonal matrix via Newton-Schulz iteration in
    bfloat16. See module docstring for usage constraints.

    Arguments:
        params: iterable of `nn.Parameter` to optimize. All must be 2D
            (the orthogonalization step requires `ndim >= 2`).
        lr: learning rate in units of spectral norm per update. Typical values
            are O(1e-3..1e-1) — much larger than AdamW's O(1e-5..1e-4).
        weight_decay: AdamW-style decoupled weight decay.
        momentum: EMA coefficient on the gradient (0.95 is the published default).
        nesterov: if True, apply a Nesterov-style lookahead step before
            orthogonalization (the published default).
        ns_steps: number of Newton-Schulz iterations; 5 is the published default.
    """

    def __init__(
        self,
        params: list[nn.Parameter],
        lr: float = 0.02,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
    ) -> None:
        defaults: dict[str, Any] = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(  # type: ignore[override]
        self, closure: Callable[[], float] | None = None
    ) -> float | None:
        loss: float | None = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                update = muon_update(
                    p.grad,
                    state["momentum_buffer"],
                    beta=group["momentum"],
                    ns_steps=group["ns_steps"],
                    nesterov=group["nesterov"],
                )
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])

        return loss
