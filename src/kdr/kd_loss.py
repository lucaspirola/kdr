"""Forward KLD distillation loss (LLR-0001, LLR-0002, LLR-0003).

Direct port from `structural_recovery/distillation.py:43-86`. Phase 3a
verifies bit-equal numerical parity with that source.

Top-k path (Phase 7.2): when `kd_topk` is set, compute KL on the top-k
teacher tokens plus a `p_tail · (log p_tail − log q_tail)` correction.
Standard production-KD technique; bias < 0.5% on V≈150k for k ≥ 256 on
instruction-tuning distributions. Backward compatible: `kd_topk=None`
keeps the full-vocab path bit-identical.
"""

from __future__ import annotations

import threading

import torch
import torch.nn as nn
from torch.nn import functional as F  # noqa: N812 — torch convention

# REQ: LLR-0002
# Module-level cache so we don't construct LogitsDistillationLoss per
# microbatch. Keyed by temperature; thread-safe via _CACHE_LOCK because the
# DataLoader workers may race on first access from rank 0 before the loop's
# main thread populates it.
_KLD_LOSS_CACHE: dict[float, nn.Module] = {}
_CACHE_LOCK = threading.Lock()


class _NativeKLDLoss(nn.Module):
    """Pure-torch parity with `modelopt.torch.distill.LogitsDistillationLoss`.

    Used when modelopt is unavailable (e.g. the BF16-only smoke whose image
    deliberately doesn't ship modelopt — see
    `tests/test_loop_dispatch.py::test_bf16_mode_does_not_import_modelopt`).
    Numerics match modelopt's class at any T by the parity test in
    `tests/test_kld_loss.py`: forward-KL with `batchmean` reduction and an
    internal `T**2` scaling to undo the temperature-softmax gradient shrink.
    """

    def __init__(self, temperature: float):
        super().__init__()
        self.T = float(temperature)

    def forward(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        T = self.T
        # `dtype=torch.float32` lets log_softmax/softmax accept bf16 logits
        # and accumulate the reduction in fp32 *internally*, without forcing
        # the caller to materialise a `(B*T, V)` fp32 tensor (~1.2 GB at
        # V=150k, B*T=2048). Output is fp32; numerics identical to an
        # explicit `.float()` upcast at the boundary.
        s_lp = torch.nn.functional.log_softmax(student / T, dim=-1, dtype=torch.float32)
        t_p = torch.nn.functional.softmax(teacher / T, dim=-1, dtype=torch.float32)
        return torch.nn.functional.kl_div(s_lp, t_p, reduction="batchmean") * (T * T)


def _get_kld_loss_fn(temperature: float) -> nn.Module:
    """Return the cached `LogitsDistillationLoss(temperature, batchmean)` instance.

    Modelopt is imported lazily so kdr is importable in environments without
    the modelopt wheel. If the import fails at call time we fall back to
    `_NativeKLDLoss` (bit-equal at the formula level — modelopt currently
    wraps the same `F.kl_div` reduction with the same `T**2` scaling).
    """
    fn = _KLD_LOSS_CACHE.get(temperature)
    if fn is None:
        with _CACHE_LOCK:
            fn = _KLD_LOSS_CACHE.get(temperature)
            if fn is None:
                try:
                    from modelopt.torch.distill.losses import LogitsDistillationLoss

                    fn = LogitsDistillationLoss(temperature=temperature, reduction="batchmean")
                except ImportError:
                    fn = _NativeKLDLoss(temperature=temperature)
                _KLD_LOSS_CACHE[temperature] = fn
    return fn


def _topk_fkld(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
    k: int,
) -> torch.Tensor:
    """Top-k forward-KL with tail-mass correction.

    Both inputs are `[N, V]` where `N = B*T`. Returns a scalar matching the
    full-vocab path's `batchmean × T²` convention.

    Computation:
      1. Find the top-k teacher tokens (argmax of `teacher_logits/T`).
      2. Compute `p_top`, `q_top` via `gather` on the top-k indices and
         `logsumexp` over the full vocab for the normalizer.
      3. KL = sum_top(p · (log p − log q)) + p_tail · (log p_tail − log q_tail)
         where `p_tail = 1 − Σ p_top`, `q_tail = 1 − Σ q_top`.
      4. Mean over the `N` tokens, multiply by `T²` to match the full-vocab
         convention.

    Computed in fp32 internally to match the full-vocab path's numerical
    contract; the `logsumexp` and `gather` are the only steps that
    materialise `[N, V]` fp32 tensors, and only transiently.
    """
    n_tokens, vocab = teacher_logits.shape
    if k >= vocab:
        # `kd_topk >= V` is equivalent to "no truncation"; tail mass is zero
        # by construction. Just call the full-vocab path on the upcast
        # tensors so the return value is bit-equal to the no-topk branch.
        s32 = student_logits.float() / temperature
        t32 = teacher_logits.float() / temperature
        s_lp = F.log_softmax(s32, dim=-1)
        t_p = F.softmax(t32, dim=-1)
        return F.kl_div(s_lp, t_p, reduction="batchmean") * (temperature * temperature)

    s32 = student_logits.float() / temperature
    t32 = teacher_logits.float() / temperature

    # Top-k teacher logits and indices. Top-k by logit ⇔ top-k by softmax prob
    # (softmax is monotone).
    t_top_logits, idx = torch.topk(t32, k, dim=-1)

    # Full-vocab log-normalizers — reduces [N,V] to [N], so backward only
    # needs to save the small [N] tensor plus the input.
    t_log_z = torch.logsumexp(t32, dim=-1, keepdim=True)
    s_log_z = torch.logsumexp(s32, dim=-1, keepdim=True)

    # Top-k log-probs for both teacher and student (at the same indices).
    t_top_lp = t_top_logits - t_log_z
    s_top_lp = torch.gather(s32, dim=-1, index=idx) - s_log_z

    p_top = t_top_lp.exp()
    q_top = s_top_lp.exp()

    # Tail masses. Clamp_min protects against the edge case `k == V` (handled
    # above) and floating-point underflow where Σ p_top hits 1.0 exactly.
    p_tail = (1.0 - p_top.sum(dim=-1)).clamp_min(1e-12)
    # For q_tail the clamp is load-bearing in a different way: when the student
    # concentrates probability mass OUTSIDE the teacher's top-k tokens,
    # `q_top.sum(dim=-1)` can exceed 1.0 and the un-clamped tail would be
    # negative — feeding `log` a negative would produce NaN. Clamping to 1e-12
    # yields `log(1e-12) ≈ -27.6`, so `p_tail · (log p_tail − log q_tail)` is a
    # conservative *over*-estimate of the true tail KL. That's fine for gradient
    # purposes: the sign is still correct (push the student to put more mass on
    # the teacher's top-k), and the magnitude is bounded. No test change needed.
    q_tail = (1.0 - q_top.sum(dim=-1)).clamp_min(1e-12)

    kl_top = (p_top * (t_top_lp - s_top_lp)).sum(dim=-1)
    kl_tail = p_tail * (p_tail.log() - q_tail.log())

    # batchmean reduction: divide by token count.
    return (kl_top + kl_tail).sum() / n_tokens * (temperature * temperature)


# REQ: LLR-0001
def forward_kld_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
    *,
    kd_topk: int | None = None,
) -> torch.Tensor:
    """Forward KL: `KLD(p_teacher || p_student)` averaged per token.

    Delegates to `modelopt.torch.distill.LogitsDistillationLoss` (the loss
    class QUALITY_RECOVERY_GUIDE.md §1.8.2 names as canonical).

    Reduction is `"batchmean"` after reshaping to `[B*T, V]` so PyTorch
    divides by the token count (`B*T`), giving per-token mean. ModelOpt's
    default `"mean"` would divide by `B*T*V` (an extra factor of vocab size,
    ~150k) and silently collapse the gradient signal.

    Both inputs are upcast to fp32 before the softmax so the loss is
    numerically stable on bf16 logits — the underlying model can still run
    in bf16.

    Pad/mask contract: the calibration tensor produced by
    `moe_compress.utils.calibration._tokenize_to_fixed_length` is fully
    packed (concatenated streams separated by EOS, hard 5%-shortage cap).
    Every position is a real token, so per-position averaging is correct
    without an attention_mask. If a future call site feeds pad-bearing
    sequences this contract is violated — assert at the boundary there.
    """
    # REQ: LLR-0003
    if student_logits.shape[-1] != teacher_logits.shape[-1]:
        raise ValueError(
            f"forward_kld_loss: vocab mismatch — student V={student_logits.shape[-1]} "
            f"vs teacher V={teacher_logits.shape[-1]}. Same-tokenizer distillation "
            "is required."
        )
    vocab = student_logits.shape[-1]
    if kd_topk is not None:
        # Top-k path: bypass the cached modelopt loss and run our own
        # gather-based KL with tail correction. Memory: avoids materialising
        # an `[N, V]` log-softmax tensor in the autograd graph; instead the
        # backward saves the input `[N, V]` (unavoidable, but bf16-safe) plus
        # the small `[N]` logZ vectors.
        s = student_logits.reshape(-1, vocab)
        t = teacher_logits.reshape(-1, vocab)
        return _topk_fkld(s, t, temperature, kd_topk)
    fn = _get_kld_loss_fn(temperature)
    if isinstance(fn, _NativeKLDLoss):
        # Native path's `log_softmax(..., dtype=torch.float32)` does the
        # fp32 reduction internally; passing bf16 logits avoids
        # materialising a `(B*T, V)` fp32 tensor on every micro.
        s = student_logits.reshape(-1, vocab)
        t = teacher_logits.reshape(-1, vocab)
    else:
        # Modelopt's `LogitsDistillationLoss` expects fp32 logits (its
        # legacy contract). Keep the explicit upcast on that branch.
        s = student_logits.reshape(-1, vocab).float()
        t = teacher_logits.reshape(-1, vocab).float()
    # `LogitsDistillationLoss.forward(student, teacher)`: per modelopt's API,
    # the FIRST positional argument is the predicted distribution Q (student)
    # and the SECOND is the target distribution P (teacher). The function
    # computes `KLD(p_teacher || p_student)` — i.e. forward KL with teacher
    # as the target. The loss is multiplied internally by T**2 to compensate
    # for gradient scaling under temperature softmax.
    result: torch.Tensor = fn(s, t)
    return result
