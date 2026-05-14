"""Tests for `kdr.training.optim` (LLR-0039).

# VERIFIES: LLR-0039
"""

from __future__ import annotations

from typing import Any, Literal, cast

import pytest
import torch
import torch.nn as nn

from kdr.config import DistillationConfig
from kdr.training._muon import SingleDeviceMuon
from kdr.training.optim import (
    ChainedOptimizer,
    build_optimizer,
    classify_params,
    cosine_with_warmup,
    set_lr,
)

_OptName = Literal["adamw_bnb_8bit", "deepspeed_cpu_adam", "muon_with_adamw"]


def _make_dconf(optimizer: str, **overrides: Any) -> DistillationConfig:
    base: dict[str, Any] = dict(
        loss="forward_kld",
        temperature=1.0,
        optimizer=cast("_OptName", optimizer),
        learning_rate=3e-5,
        min_learning_rate=3e-7,
        weight_decay=0.0,
        betas=[0.9, 0.95],
        grad_clip_norm=1.0,
        warmup_steps=10,
        total_tokens=1_000_000,
        per_device_batch_size=1,
        gradient_accumulation=1,
        sequence_length=128,
        log_every_n_steps=1,
        eval_every_n_steps=10,
        save_every_n_steps=0,
        trainable_scope="full",
    )
    base.update(overrides)
    return DistillationConfig(**base)


def test_build_optimizer_rejects_no_trainable_params() -> None:
    student = nn.Linear(8, 8)
    for p in student.parameters():
        p.requires_grad_(False)
    with pytest.raises(RuntimeError, match="no trainable parameters"):
        build_optimizer(student, _make_dconf("adamw_bnb_8bit"))


def test_build_optimizer_rejects_unknown_optimizer() -> None:
    """Bypass Pydantic to feed an unknown name — verifies the runtime
    branch's error path independently of the schema's Literal."""
    student = nn.Linear(8, 8)
    dconf = _make_dconf("adamw_bnb_8bit")
    object.__setattr__(dconf, "optimizer", "unknown_optimizer")
    with pytest.raises(ValueError, match="Unknown optimizer"):
        build_optimizer(student, dconf)


def test_build_optimizer_bnb_path_raises_when_bnb_missing() -> None:
    """Smoke: bnb is not installed in the kdr venv. The branch must call into
    the import → ImportError. We assert it propagates rather than swallowing."""
    student = nn.Linear(8, 8)
    with pytest.raises(ModuleNotFoundError):
        build_optimizer(student, _make_dconf("adamw_bnb_8bit"))


def test_build_optimizer_dscpuadam_path_raises_when_deepspeed_missing() -> None:
    student = nn.Linear(8, 8)
    with pytest.raises(ModuleNotFoundError):
        build_optimizer(student, _make_dconf("deepspeed_cpu_adam"))


def test_cosine_with_warmup_warmup_phase() -> None:
    # Step 0 → first warmup increment.
    lr = cosine_with_warmup(
        step=0, warmup_steps=10, total_steps=100, lr_max=1e-3, lr_min=1e-6
    )
    assert lr == pytest.approx(1e-3 * 1 / 10)

    # Step 9 → end of warmup.
    lr = cosine_with_warmup(
        step=9, warmup_steps=10, total_steps=100, lr_max=1e-3, lr_min=1e-6
    )
    assert lr == pytest.approx(1e-3)


def test_cosine_with_warmup_decay_phase() -> None:
    # Step warmup_steps → cosine starts at lr_max.
    lr = cosine_with_warmup(
        step=10, warmup_steps=10, total_steps=100, lr_max=1e-3, lr_min=1e-6
    )
    assert lr == pytest.approx(1e-3, rel=1e-9)

    # Step total_steps - 1 → near lr_min.
    lr_end = cosine_with_warmup(
        step=99, warmup_steps=10, total_steps=100, lr_max=1e-3, lr_min=1e-6
    )
    assert lr_end < 1e-3
    assert lr_end >= 1e-6 - 1e-9


def test_cosine_with_warmup_clamps_past_total() -> None:
    lr = cosine_with_warmup(
        step=200, warmup_steps=10, total_steps=100, lr_max=1e-3, lr_min=1e-6
    )
    assert lr == 1e-6


def test_set_lr_overwrites_all_param_groups() -> None:
    student = nn.Linear(8, 8)
    optim = torch.optim.AdamW(student.parameters(), lr=1.0)
    set_lr(optim, 0.42)
    for g in optim.param_groups:
        assert g["lr"] == 0.42


# ---------------------------------------------------------------------------
# Muon-with-AdamW: classify_params, ChainedOptimizer, ratio-preserving set_lr
# ---------------------------------------------------------------------------


def test_classify_params_separates_muon_from_adamw() -> None:
    """2D Linear weights go to Muon; embeddings/biases/norms go to AdamW."""

    class _Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(100, 16)
            self.proj = nn.Linear(16, 32, bias=True)
            self.norm = nn.LayerNorm(32)
            self.head = nn.Linear(32, 100, bias=False)

    m = _Tiny()
    for p in m.parameters():
        p.requires_grad_(True)
    muon, adamw = classify_params(m, carve_out_patterns=[])

    muon_ids = {id(p) for p in muon}
    adamw_ids = {id(p) for p in adamw}
    assert id(m.proj.weight) in muon_ids
    assert id(m.head.weight) in muon_ids
    assert id(m.embed.weight) in adamw_ids
    assert id(m.proj.bias) in adamw_ids
    assert id(m.norm.weight) in adamw_ids
    assert id(m.norm.bias) in adamw_ids


def test_classify_params_carve_out_patterns_respected() -> None:
    """A Linear named with a carve-out substring lands in AdamW, not Muon."""

    class _Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.expert = nn.Linear(8, 8, bias=False)
            self.router = nn.Linear(8, 4, bias=False)

    m = _Tiny()
    muon, adamw = classify_params(m, carve_out_patterns=["router"])

    muon_ids = {id(p) for p in muon}
    adamw_ids = {id(p) for p in adamw}
    assert id(m.expert.weight) in muon_ids
    assert id(m.router.weight) in adamw_ids


def test_classify_params_dedupes_tied_embedding() -> None:
    """Two Linears sharing a weight tensor must appear in exactly one group."""
    a = nn.Linear(8, 8, bias=False)
    b = nn.Linear(8, 8, bias=False)
    b.weight = a.weight  # tie

    class _Tied(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = a
            self.b = b

    m = _Tied()
    muon, adamw = classify_params(m, carve_out_patterns=[])
    all_ids = [id(p) for p in muon + adamw]
    assert len(all_ids) == len(set(all_ids)), "tied weight appeared in both groups"
    assert id(a.weight) in (set(id(p) for p in muon) | set(id(p) for p in adamw))


def test_classify_params_skips_non_trainable() -> None:
    """`requires_grad=False` params are excluded from both groups."""

    class _Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.frozen = nn.Linear(8, 8, bias=False)
            self.trainable = nn.Linear(8, 8, bias=False)

    m = _Tiny()
    m.frozen.weight.requires_grad_(False)
    muon, adamw = classify_params(m, carve_out_patterns=[])
    all_ids = {id(p) for p in muon + adamw}
    assert id(m.frozen.weight) not in all_ids
    assert id(m.trainable.weight) in all_ids


def test_build_optimizer_muon_requires_carve_outs() -> None:
    """muon_with_adamw without carve_out_patterns must error explicitly."""
    student = nn.Linear(8, 8)
    dconf = _make_dconf("muon_with_adamw", muon_learning_rate=2e-3)
    with pytest.raises(ValueError, match="carve_out_patterns is required"):
        build_optimizer(student, dconf, carve_out_patterns=None)


def test_build_optimizer_muon_path_raises_when_bnb_missing() -> None:
    """Muon path imports both vendored Muon and bnb AdamW; without bnb, error."""
    student = nn.Linear(8, 8)
    dconf = _make_dconf("muon_with_adamw", muon_learning_rate=2e-3)
    with pytest.raises(ModuleNotFoundError):
        build_optimizer(student, dconf, carve_out_patterns=["lm_head"])


def test_chained_optimizer_snapshots_base_lr() -> None:
    """ChainedOptimizer snapshots each group's `lr` as `base_lr` on init."""
    p1 = nn.Parameter(torch.zeros(4, 4))
    p2 = nn.Parameter(torch.zeros(4))
    opt1 = torch.optim.SGD([p1], lr=1.0)
    opt2 = torch.optim.SGD([p2], lr=0.02)
    chained = ChainedOptimizer(opt1, opt2)
    assert chained.param_groups[0]["base_lr"] == 1.0
    assert chained.param_groups[1]["base_lr"] == 0.02


def test_set_lr_ratio_preserving_with_lr_max_ref() -> None:
    """With `lr_max_ref` given, each group's LR scales by ratio × base_lr."""
    p1 = nn.Parameter(torch.zeros(4, 4))
    p2 = nn.Parameter(torch.zeros(4))
    opt1 = torch.optim.SGD([p1], lr=1.0)
    opt2 = torch.optim.SGD([p2], lr=0.02)
    chained = cast(torch.optim.Optimizer, ChainedOptimizer(opt1, opt2))
    set_lr(chained, 0.5, lr_max_ref=1.0)
    assert chained.param_groups[0]["lr"] == pytest.approx(0.5)
    assert chained.param_groups[1]["lr"] == pytest.approx(0.01)


def test_set_lr_backward_compat_when_lr_max_ref_none() -> None:
    """`lr_max_ref=None` keeps the single-group overwrite-with-scalar behavior.

    This pins the existing contract for AdamW-only optimizers — every group
    gets the same scalar regardless of any prior `base_lr` snapshot.
    """
    student = nn.Linear(8, 8)
    optim = torch.optim.AdamW(student.parameters(), lr=1.0)
    set_lr(optim, 0.42)
    for g in optim.param_groups:
        assert g["lr"] == 0.42


def test_chained_optimizer_step_updates_both_groups() -> None:
    """Forward + backward + step on a ChainedOptimizer wraps two real optimizers.

    Uses SingleDeviceMuon over a 2D Linear weight and torch.optim.SGD over a
    1D bias, since bnb isn't installable in the kdr test venv. This pins the
    behaviour that both inner optimizers' `.step()` runs and both params are
    updated.
    """
    proj = nn.Linear(4, 4, bias=True)
    x = torch.randn(2, 4)
    w_before = proj.weight.detach().clone()
    b_before = proj.bias.detach().clone()
    muon_opt = SingleDeviceMuon([proj.weight], lr=0.05, momentum=0.95)
    adamw_opt = torch.optim.SGD([proj.bias], lr=0.05)
    chained = ChainedOptimizer(muon_opt, adamw_opt)
    loss = proj(x).square().mean()
    loss.backward()
    chained.step()
    chained.zero_grad(set_to_none=True)
    assert not torch.equal(proj.weight, w_before), "Muon group did not update"
    assert not torch.equal(proj.bias, b_before), "AdamW group did not update"


def test_set_lr_rejects_zero_lr_max_ref() -> None:
    """`lr_max_ref == 0.0` is a divide-by-zero; must raise ValueError."""
    student = nn.Linear(8, 8)
    optim = torch.optim.AdamW(student.parameters(), lr=1.0)
    with pytest.raises(ValueError, match="lr_max_ref must be > 0"):
        set_lr(optim, 0.5, lr_max_ref=0.0)


def test_chained_optimizer_state_dict_preserves_base_lr() -> None:
    """`base_lr` survives a state_dict round-trip and drives ratio scaling.

    Regression guard: even if an inner optimizer's `state_dict` drops
    arbitrary param-group keys (some bnb/DS wrappers monkey-patch this),
    `ChainedOptimizer.state_dict` mirrors `base_lr` at the top level and
    `load_state_dict` re-injects it on the fresh optimizer.
    """
    p1 = nn.Parameter(torch.zeros(4, 4))
    p2 = nn.Parameter(torch.zeros(4))
    opt1 = torch.optim.SGD([p1], lr=1.0)
    opt2 = torch.optim.SGD([p2], lr=0.02)
    chained = cast(torch.optim.Optimizer, ChainedOptimizer(opt1, opt2))

    # Drive the schedule mid-run: lr should become 0.5 and 0.01 respectively
    # while base_lr stays 1.0 and 0.02.
    set_lr(chained, 0.5, lr_max_ref=1.0)
    assert chained.param_groups[0]["lr"] == pytest.approx(0.5)
    assert chained.param_groups[1]["lr"] == pytest.approx(0.01)
    assert chained.param_groups[0]["base_lr"] == pytest.approx(1.0)
    assert chained.param_groups[1]["base_lr"] == pytest.approx(0.02)

    saved_sd = chained.state_dict()

    # Fresh chained optimizer at the original base LRs. After load,
    # base_lr must be restored from the saved state_dict.
    p1b = nn.Parameter(torch.zeros(4, 4))
    p2b = nn.Parameter(torch.zeros(4))
    opt1b = torch.optim.SGD([p1b], lr=1.0)
    opt2b = torch.optim.SGD([p2b], lr=0.02)
    loaded = cast(torch.optim.Optimizer, ChainedOptimizer(opt1b, opt2b))
    loaded.load_state_dict(saved_sd)

    assert loaded.param_groups[0]["base_lr"] == pytest.approx(1.0)
    assert loaded.param_groups[1]["base_lr"] == pytest.approx(0.02)

    # And the loaded optimizer's ratio scaling still respects the original
    # base-LR snapshot.
    set_lr(loaded, 0.25, lr_max_ref=1.0)
    assert loaded.param_groups[0]["lr"] == pytest.approx(0.25)
    assert loaded.param_groups[1]["lr"] == pytest.approx(0.005)


def test_chained_optimizer_state_dict_roundtrip() -> None:
    """state_dict / load_state_dict round-trips across the inner optimizers."""
    proj = nn.Linear(4, 4, bias=True)
    muon_opt = SingleDeviceMuon([proj.weight], lr=0.05, momentum=0.95)
    sgd = torch.optim.SGD([proj.bias], lr=0.05)
    chained = ChainedOptimizer(muon_opt, sgd)
    proj(torch.randn(2, 4)).square().mean().backward()
    chained.step()
    sd = chained.state_dict()

    proj2 = nn.Linear(4, 4, bias=True)
    proj2.load_state_dict(proj.state_dict())
    muon_opt2 = SingleDeviceMuon([proj2.weight], lr=0.05, momentum=0.95)
    sgd2 = torch.optim.SGD([proj2.bias], lr=0.05)
    chained2 = ChainedOptimizer(muon_opt2, sgd2)
    chained2.load_state_dict(sd)
    proj2(torch.randn(2, 4)).square().mean().backward()
    chained2.step()  # must not raise


def test_chained_optimizer_load_state_dict_rejects_group_mismatch() -> None:
    """`load_state_dict` raises if the saved base_lrs list and the current
    optimizer's `param_groups` disagree in length, AND the rejection happens
    before any inner optimizer's state has been mutated (all-or-nothing load).

    Previously the re-injection used `zip(...)` which silently truncates;
    the explicit length check surfaces a reconstruction mismatch (e.g. the
    optimizer was rebuilt with a different param-group shape between save
    and resume) rather than swallowing it. Additionally, the validation
    pre-pass guards against partial-apply: if opt_1's load would fail, opt_0
    must NOT have been mutated by an earlier inner `load_state_dict` call.
    """
    p1 = nn.Parameter(torch.zeros(4))
    p2 = nn.Parameter(torch.zeros(4))
    opt1 = torch.optim.SGD([p1], lr=1.0)
    opt2 = torch.optim.SGD([p2], lr=1.0)
    chained = cast(torch.optim.Optimizer, ChainedOptimizer(opt1, opt2))

    saved_sd = chained.state_dict()

    # Sanity: each inner optimizer has exactly one param_group / one base_lr.
    assert len(saved_sd["base_lrs_0"]) == 1
    assert len(saved_sd["base_lrs_1"]) == 1

    # Build a fresh ChainedOptimizer with the SAME structure (so load would
    # normally succeed). opt_0's group `lr` is set to a SENTINEL distinct
    # from the saved value (1.0); a successful inner load_state_dict on opt_0
    # would overwrite it back to 1.0. We assert below that the sentinel
    # survives, proving the mismatch was caught before opt_0 was mutated.
    sentinel = 0.123
    p1b = nn.Parameter(torch.zeros(4))
    p2b = nn.Parameter(torch.zeros(4))
    opt1b = torch.optim.SGD([p1b], lr=sentinel)
    opt2b = torch.optim.SGD([p2b], lr=1.0)
    loaded = cast(torch.optim.Optimizer, ChainedOptimizer(opt1b, opt2b))
    assert opt1b.param_groups[0]["lr"] == sentinel

    # Tamper with `base_lrs_1` (NOT `base_lrs_0`) so the mismatch is on the
    # SECOND inner optimizer. A naive implementation would have already
    # mutated opt_0 by the time it reaches opt_1 and fails.
    saved_sd["base_lrs_1"] = [1.0, 1.0]

    with pytest.raises(RuntimeError, match="count mismatch"):
        loaded.load_state_dict(saved_sd)

    # All-or-nothing guarantee: opt_0's group[0]["lr"] must still equal the
    # sentinel (i.e. opt_0.load_state_dict was never called).
    assert opt1b.param_groups[0]["lr"] == sentinel, (
        "ChainedOptimizer.load_state_dict mutated opt_0 before opt_1's "
        "mismatch was detected — partial-apply bug regressed."
    )
