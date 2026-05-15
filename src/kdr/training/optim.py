"""Optimizer tier dispatch + LR scheduler helpers.

Three tiers, picked from the YAML's `distillation.optimizer` field:

* `adamw_bnb_8bit` — single-GPU smoke runs (no DeepSpeed). 8-bit AdamW from
  bitsandbytes; ~75% optimizer-state reduction vs fp32 AdamW. **Does not
  compose with DeepSpeed ZeRO-3** — bnb owns its `optim.step` and would
  fight DS's fp32 reduce.
* `deepspeed_cpu_adam` — multi-GPU under ZeRO-3. Optimizer state lives on
  the host CPU; per-rank GPU footprint is teacher (sharded) + student
  (sharded) + grads (sharded) + activations.
* `muon_with_adamw` — Muon (Keller Jordan, 2024) over 2D Linear weight
  matrices, AdamW-8bit over everything else (embeddings, lm_head, RMSNorm
  scales, routers, val_proj, biases). The split is driven by the adapter's
  `fp32_carve_outs` pattern list, passed in via `carve_out_patterns`. Muon
  is intended only for hidden-layer matrices; the published recipe and the
  Moonlight paper both keep embeddings/heads/norms on AdamW. **Does not
  compose with DeepSpeed ZeRO-3** — the duck-typed `ChainedOptimizer`
  wrapper is not a `torch.optim.Optimizer` subclass; DS3's optimizer wrap
  requires the real subclass.
"""

from __future__ import annotations

import math
from typing import Any, cast

import torch
import torch.nn as nn

from ..config import DistillationConfig


class ChainedOptimizer:
    """Duck-typed wrapper that steps multiple inner optimizers in order.

    Exposes the surface the kdr training loop actually uses:
        - `param_groups` (concatenation of every inner optimizer's groups)
        - `state` (merged dict)
        - `defaults` (from the first inner optimizer)
        - `step()`, `zero_grad(set_to_none)`
        - `state_dict()`, `load_state_dict()`

    Each inner group's original `lr` is snapshotted as `base_lr` on
    construction so the ratio-preserving `set_lr` below can scale every
    group by the same cosine ratio while preserving the absolute LR ratio
    between groups (typically Muon group at e.g. 2e-3, AdamW group at 5e-5).

    NOT a `torch.optim.Optimizer` subclass — we duck-type because Optimizer's
    `__init__` requires its own `params` iterable and would fight us on
    bookkeeping. The caller casts the return value at the `build_optimizer`
    boundary so static type checkers see the expected type.
    """

    def __init__(self, *optimizers: torch.optim.Optimizer) -> None:
        if not optimizers:
            raise ValueError("ChainedOptimizer requires at least one inner optimizer")
        self._opts: tuple[torch.optim.Optimizer, ...] = optimizers
        for opt in self._opts:
            for g in opt.param_groups:
                g.setdefault("base_lr", g["lr"])
        # accelerate touches `.defaults`; expose the first opt's so the
        # most common keys (`lr`, `weight_decay`) are present.
        self.defaults: dict[str, Any] = dict(self._opts[0].defaults)

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return [g for opt in self._opts for g in opt.param_groups]

    @property
    def state(self) -> dict[Any, Any]:
        merged: dict[Any, Any] = {}
        for opt in self._opts:
            merged.update(opt.state)
        return merged

    def step(self, closure: Any = None) -> Any:
        del closure  # we don't compose closures across inner optimizers
        for opt in self._opts:
            opt.step()
        return None

    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt in self._opts:
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        # PyTorch's Optimizer.state_dict preserves arbitrary per-group keys
        # (including our `base_lr`), but we explicitly mirror them at the top
        # level too. This guards against environments where an inner optimizer
        # (e.g. bnb's AdamW8bit, or a monkey-patched DeepSpeed wrapper) drops
        # extra param-group keys during serialization. On `load_state_dict`
        # we re-inject `base_lr` from the saved list so the ratio-preserving
        # `set_lr` always has the original snapshot to scale against.
        out: dict[str, Any] = {}
        for i, opt in enumerate(self._opts):
            out[f"opt_{i}"] = opt.state_dict()
            out[f"base_lrs_{i}"] = [g["base_lr"] for g in opt.param_groups]
        return out

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        # Phase 1: validate every inner optimizer's length BEFORE mutating any
        # inner optimizer. We check against the SAVED `param_groups` in the
        # state dict (what the inner optimizer would restore on load), NOT the
        # current inner optimizer's param_groups — those are about to be
        # replaced. This makes the wrapper an all-or-nothing load: either both
        # inner optimizers get applied, or neither does (no partial mutation).
        for i, opt in enumerate(self._opts):
            inner_sd = sd[f"opt_{i}"]
            saved_groups = inner_sd.get("param_groups", [])
            if len(opt.param_groups) != len(saved_groups):
                raise RuntimeError(
                    f"ChainedOptimizer.load_state_dict: param_group count mismatch "
                    f"for opt_{i}: checkpoint has {len(saved_groups)} groups, current "
                    f"optimizer has {len(opt.param_groups)}. This indicates the "
                    f"optimizer was reconstructed with a different param-group shape "
                    f"between save and resume."
                )
            base_lrs_key = f"base_lrs_{i}"
            if base_lrs_key in sd and len(sd[base_lrs_key]) != len(saved_groups):
                raise RuntimeError(
                    f"ChainedOptimizer.load_state_dict: base_lrs count mismatch "
                    f"for opt_{i}: base_lrs has {len(sd[base_lrs_key])} entries, "
                    f"saved param_groups has {len(saved_groups)}. Refusing to "
                    f"apply a partial / inconsistent load."
                )

        # Phase 2: apply. Validated above, so neither inner load_state_dict
        # nor the base_lr re-injection can hit a length mismatch.
        for i, opt in enumerate(self._opts):
            opt.load_state_dict(sd[f"opt_{i}"])
            base_lrs = sd.get(f"base_lrs_{i}")
            if base_lrs is not None:
                for g, base in zip(opt.param_groups, base_lrs):
                    g["base_lr"] = base
            else:
                # Backward-compat with older state_dicts that didn't mirror
                # base_lr: fall back to whatever PyTorch's inner deserialization
                # preserved (or current `lr` as a last resort).
                for g in opt.param_groups:
                    g.setdefault("base_lr", g["lr"])


def classify_params(
    student: nn.Module,
    carve_out_patterns: list[str],
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Split `student`'s trainable params into (muon_params, adamw_params).

    Rules (first match wins):
      1. If any pattern in `carve_out_patterns` is a substring of the
         param's dotted name → **AdamW** group.
      2. Else if the param's owning module is `nn.Linear`, the param IS
         `module.weight`, and `param.ndim == 2` → **Muon** group.
      3. Else → AdamW group.

    Dedupes by `id(param)` (first-seen wins) so tied tensors (e.g.
    `lm_head` ↔ `embed_tokens`) appear exactly once.

    Raises `RuntimeError` if the final two-group partition fails to cover
    every unique trainable param — silent param drop would mirror the
    run-1 collapse failure mode.
    """
    # Build dotted-name + owning-module index for every parameter the model
    # exposes; first-seen name wins. Walking named_modules + named_parameters
    # with `recurse=False` is the standard idiom for "name with parent module".
    name_of: dict[int, str] = {}
    owner_of: dict[int, nn.Module] = {}
    attrname_of: dict[int, str] = {}
    # Pre-pass: transparently follow `nn.utils.parametrize.register_parametrization`
    # (used by NativeBackend's weight STE). After registration, the backing
    # Parameter lives at `module.parametrizations.weight.original`, while
    # `module.weight` becomes a recomputed Tensor (not a Parameter). A naive
    # named_parameters walk buckets that Parameter under owner=ParametrizationList,
    # attr='original' — which fails the Muon predicate below and silently
    # pushes every quantized Linear into AdamW.
    for mod_name, mod in student.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        parametrizations = getattr(mod, "parametrizations", None)
        wp = getattr(parametrizations, "weight", None) if parametrizations is not None else None
        weight_param = wp.original if wp is not None else mod.weight
        if not isinstance(weight_param, nn.Parameter) or not weight_param.requires_grad:
            continue
        key = id(weight_param)
        if key in name_of:
            continue
        name_of[key] = f"{mod_name}.weight" if mod_name else "weight"
        owner_of[key] = mod
        attrname_of[key] = "weight"

    for mod_name, mod in student.named_modules():
        for attr, p in mod.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            key = id(p)
            if key in name_of:
                continue
            dotted = f"{mod_name}.{attr}" if mod_name else attr
            name_of[key] = dotted
            owner_of[key] = mod
            attrname_of[key] = attr

    unique_trainable: list[nn.Parameter] = []
    seen: set[int] = set()
    for p in student.parameters():
        if not p.requires_grad:
            continue
        if id(p) in seen:
            continue
        seen.add(id(p))
        unique_trainable.append(p)

    muon: list[nn.Parameter] = []
    adamw: list[nn.Parameter] = []
    unclassified: list[str] = []
    for p in unique_trainable:
        key = id(p)
        dotted = name_of.get(key, "<unnamed>")
        owner = owner_of.get(key)
        attr = attrname_of.get(key, "")

        if any(pat in dotted for pat in carve_out_patterns):
            adamw.append(p)
            continue

        if (
            isinstance(owner, nn.Linear)
            and attr == "weight"
            and p.ndim == 2
        ):
            muon.append(p)
            continue

        # Catches: biases, Embedding weights, RMSNorm scales, custom non-Linear
        # parameters, 1D scales. All go to AdamW.
        adamw.append(p)

    if len(muon) + len(adamw) != len(unique_trainable):
        raise RuntimeError(
            "classify_params: partition does not cover every trainable "
            f"param. unique={len(unique_trainable)} muon={len(muon)} "
            f"adamw={len(adamw)} unclassified={unclassified}"
        )
    muon_ids = {id(p) for p in muon}
    adamw_ids = {id(p) for p in adamw}
    overlap = muon_ids & adamw_ids
    if overlap:
        overlapping_names = [name_of.get(i, "<unnamed>") for i in overlap]
        raise RuntimeError(
            f"classify_params: parameter(s) in both groups: {overlapping_names}"
        )

    return muon, adamw


# REQ: LLR-0039
def build_optimizer(
    student: nn.Module,
    dconf: DistillationConfig,
    *,
    carve_out_patterns: list[str] | None = None,
) -> torch.optim.Optimizer:
    """Return an optimizer instance for `student` per `dconf.optimizer`.

    `'adamw_bnb_8bit'` returns `bitsandbytes.optim.AdamW8bit` (a
    `torch.optim.Optimizer` subclass). `'deepspeed_cpu_adam'` returns
    `deepspeed.ops.adam.DeepSpeedCPUAdam(adamw_mode=True)` (also a
    `torch.optim.Optimizer` subclass). `'muon_with_adamw'` returns a
    `ChainedOptimizer` (duck-typed); requires `carve_out_patterns` to be
    passed in (typically the adapter's `fp32_carve_outs(student)`).

    Both AdamW branches collect only `requires_grad=True` params — calling
    this after toggling trainable scope to e.g. `experts_only` would put
    only the expert params under the optimizer.
    """
    trainable = [p for p in student.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError(
            "build_optimizer: no trainable parameters. Did you set "
            "`requires_grad=True` on the student before calling?"
        )
    name = dconf.optimizer
    lr = dconf.learning_rate
    betas = (dconf.betas[0], dconf.betas[1])
    wd = dconf.weight_decay

    if name == "adamw_bnb_8bit":
        import bitsandbytes as bnb

        return cast(
            torch.optim.Optimizer,
            bnb.optim.AdamW8bit(trainable, lr=lr, betas=betas, weight_decay=wd),
        )

    if name == "deepspeed_cpu_adam":
        # adamw_mode=True selects AdamW (decoupled weight decay), matching bnb.
        from deepspeed.ops.adam import DeepSpeedCPUAdam

        return cast(
            torch.optim.Optimizer,
            DeepSpeedCPUAdam(
                trainable, lr=lr, betas=betas, weight_decay=wd, adamw_mode=True
            ),
        )

    if name == "muon_with_adamw":
        if carve_out_patterns is None:
            raise ValueError(
                "build_optimizer('muon_with_adamw'): carve_out_patterns is "
                "required. Pass `adapter.fp32_carve_outs(student)` from the "
                "training loop's Stage 2 dispatch."
            )
        if dconf.muon_learning_rate is None:
            # Defensive: the schema validator already rejects this, but make
            # the runtime branch self-defending.
            raise ValueError(
                "build_optimizer('muon_with_adamw'): muon_learning_rate is "
                "required when optimizer='muon_with_adamw'."
            )
        import bitsandbytes as bnb

        from ._muon import SingleDeviceMuon

        muon_params, adamw_params = classify_params(student, carve_out_patterns)
        if not muon_params:
            raise RuntimeError(
                "muon_with_adamw: no 2D Linear weights to optimize after "
                "applying carve-outs. Check the carve-out pattern list."
            )
        if not adamw_params:
            raise RuntimeError(
                "muon_with_adamw: no AdamW-tier params found (embeddings, "
                "norms, biases all missing?). Check the model structure."
            )
        adamw_lr = dconf.adamw_group_learning_rate or lr
        muon_opt = SingleDeviceMuon(
            muon_params,
            lr=dconf.muon_learning_rate,
            weight_decay=wd,
            momentum=dconf.muon_momentum,
            nesterov=dconf.muon_nesterov,
            ns_steps=dconf.muon_ns_steps,
        )
        adamw_opt = bnb.optim.AdamW8bit(
            adamw_params, lr=adamw_lr, betas=betas, weight_decay=wd
        )
        return cast(torch.optim.Optimizer, ChainedOptimizer(muon_opt, adamw_opt))

    raise ValueError(
        f"Unknown optimizer: {name!r}. "
        "Expected 'adamw_bnb_8bit', 'deepspeed_cpu_adam', or 'muon_with_adamw'."
    )


def cosine_with_warmup(
    step: int,
    *,
    warmup_steps: int,
    total_steps: int,
    lr_max: float,
    lr_min: float,
) -> float:
    """Linear warmup `[0..warmup-1]` → cosine decay `[warmup..total_steps-1]`.

    `step` is the 0-based optimizer-step index about to be taken (called
    BEFORE `optim.step`). The cosine approaches `lr_min` from above as
    `step` approaches `total_steps - 1` (it never dips below `lr_min`); for
    `step >= total_steps` the function hard-clamps to `lr_min`.
    """
    if step < warmup_steps:
        return lr_max * (step + 1) / max(1, warmup_steps)
    if step >= total_steps:
        return lr_min
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))


def set_lr(
    optim: torch.optim.Optimizer,
    lr_ref: float,
    *,
    lr_max_ref: float | None = None,
) -> None:
    """Set every param-group's `lr` according to the schedule.

    Single-group call (existing behavior): pass `lr_max_ref=None` (or omit).
    Every group's `lr` is overwritten with `lr_ref`. This is the path every
    AdamW-only optimizer uses; `test_set_lr_overwrites_all_param_groups`
    pins it.

    Multi-group call (ChainedOptimizer with two base LRs): pass
    `lr_max_ref=dconf.learning_rate` (the AdamW-tier base LR that the
    cosine schedule was computed against). Each group's `lr` becomes
    `(lr_ref / lr_max_ref) * group['base_lr']`, preserving the absolute
    LR ratio between Muon and AdamW groups while sharing the cosine *shape*.
    """
    if lr_max_ref is None:
        for g in optim.param_groups:
            g["lr"] = lr_ref
        return
    if lr_max_ref == 0.0:
        raise ValueError(f"set_lr: lr_max_ref must be > 0 (got {lr_max_ref})")
    ratio = lr_ref / lr_max_ref
    for g in optim.param_groups:
        base = g.get("base_lr", lr_ref)
        g["lr"] = base * ratio
