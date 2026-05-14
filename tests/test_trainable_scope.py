"""Tests for `_enable_trainable_scope` (Phase 7.2 routers_frozen).

The kdr trainer toggles `requires_grad` via `_enable_trainable_scope` BEFORE
constructing the optimizer. Adding a `routers_frozen` scope freezes the
student's router params (which router-replay already overrides in the
forward, so the gradient signal is meaningless anyway).
"""

from __future__ import annotations

import pytest
import torch.nn as nn

from kdr.training.loop import _enable_trainable_scope


class _MoELike(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(100, 16)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "router": nn.Linear(16, 4, bias=False),
                        "expert": nn.Linear(16, 16, bias=True),
                        "norm": nn.LayerNorm(16),
                    }
                )
                for _ in range(2)
            ]
        )
        self.lm_head = nn.Linear(16, 100, bias=False)


def test_full_scope_marks_every_param_trainable() -> None:
    m = _MoELike()
    for p in m.parameters():
        p.requires_grad_(False)  # freeze first to isolate effect
    n = _enable_trainable_scope(m, scope="full")
    expected = sum(p.numel() for p in m.parameters())
    assert n == expected
    assert all(p.requires_grad for p in m.parameters())


def test_routers_frozen_scope_freezes_router_params_only() -> None:
    m = _MoELike()
    for p in m.parameters():
        p.requires_grad_(False)
    n = _enable_trainable_scope(m, scope="routers_frozen")
    # Every router weight must be frozen.
    for layer in m.layers:
        assert not layer["router"].weight.requires_grad
        # Other layer components must be trainable.
        assert layer["expert"].weight.requires_grad
        assert layer["expert"].bias.requires_grad
        assert layer["norm"].weight.requires_grad
        assert layer["norm"].bias.requires_grad
    # Embeddings + lm_head must be trainable too.
    assert m.embed.weight.requires_grad
    assert m.lm_head.weight.requires_grad
    # `n` returns the trainable numel, not the total.
    expected_trainable = sum(
        p.numel() for p in m.parameters() if p.requires_grad
    )
    assert n == expected_trainable
    assert n < sum(p.numel() for p in m.parameters())


def test_routers_frozen_substring_match_is_dotted_name() -> None:
    """The frozen set is driven by 'router' substring in the dotted name,
    not by module type. A Linear that doesn't have 'router' in its path
    stays trainable even if some heuristic might guess otherwise.
    """

    class _Mixed(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Has 'router' substring → frozen.
            self.router_gate = nn.Linear(8, 4, bias=False)
            # No 'router' substring → trainable.
            self.expert_routing_layer = nn.Linear(8, 8, bias=False)
            # 'router' substring in module path → frozen.
            self.block = nn.ModuleDict({"router": nn.Linear(8, 4, bias=False)})

    m = _Mixed()
    _enable_trainable_scope(m, scope="routers_frozen")
    assert not m.router_gate.weight.requires_grad
    assert m.expert_routing_layer.weight.requires_grad
    assert not m.block["router"].weight.requires_grad


def test_unknown_scope_raises_notimplemented() -> None:
    m = _MoELike()
    with pytest.raises(NotImplementedError, match="trainable_scope"):
        _enable_trainable_scope(m, scope="experts_only")
