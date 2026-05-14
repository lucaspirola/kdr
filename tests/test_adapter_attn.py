"""Mode-aware attention selection tests (LLR-0026).

Phase 7 finalized LLR-0026 with a mode-aware `required_attn_implementation`:

- ``mode == 'da_qad'`` → ``'eager'`` (K/V hookability for KV-quant).
- ``mode == 'bf16'`` → ``'sdpa'`` (no hooks placed, SDPA is ~2-3× faster
  on Hopper/Blackwell and numerically equivalent at BF16).
- Any other mode → raises ``ValueError`` (no silent fallback).

These tests cover the four cells of `{Zaya1Adapter} × {bf16, da_qad}` plus
the rejection of unknown modes. The actual `from_pretrained(..., attn_*)`
behavior on real ZAYA1 is only exercised in Phase 7 GPU validation; here
we exercise the policy method in isolation (no GPU, no HF downloads).

# VERIFIES: LLR-0026
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from kdr.adapters.zaya1_8b import Zaya1Adapter
from kdr.config import StudentConfig, TeacherConfig


def test_required_attn_da_qad_student_is_eager() -> None:
    """LLR-0026: da_qad student must return eager so KV-quant hooks see
    post-projection K/V tensors (CCA layer is wired only under eager in the
    Zyphra fork). Default role is student so the bare call still returns
    eager — preserves the original contract."""
    assert Zaya1Adapter().required_attn_implementation("da_qad") == "eager"
    assert (
        Zaya1Adapter().required_attn_implementation("da_qad", role="student")
        == "eager"
    )


def test_required_attn_da_qad_teacher_is_sdpa() -> None:
    """LLR-0026 (Phase 7.2 split): da_qad teacher has no KV hooks, so SDPA
    is allowed for ~2× teacher-forward throughput."""
    assert (
        Zaya1Adapter().required_attn_implementation("da_qad", role="teacher")
        == "sdpa"
    )


def test_required_attn_bf16_is_sdpa_both_roles() -> None:
    """LLR-0026: BF16 has no hooks in either role; both return sdpa."""
    adapter = Zaya1Adapter()
    assert adapter.required_attn_implementation("bf16") == "sdpa"
    assert adapter.required_attn_implementation("bf16", role="teacher") == "sdpa"
    assert adapter.required_attn_implementation("bf16", role="student") == "sdpa"


def test_required_attn_rejects_unknown_mode() -> None:
    """LLR-0026 generic-tool requirement: any unrecognized mode raises
    rather than silently falling back. Future modes must be added as
    explicit branches with deliberate eager/sdpa choices."""
    adapter = Zaya1Adapter()
    with pytest.raises(ValueError, match="unsupported mode"):
        # `cast` would normally be used, but the runtime check must fire
        # regardless of static typing — `ignore[arg-type]` is the right
        # signal here that we're testing a runtime guard.
        adapter.required_attn_implementation("fp8_qad")  # type: ignore[arg-type]


def test_required_attn_returns_only_allowed_values() -> None:
    """LLR-0026 AC: the return type is `Literal['eager', 'sdpa']`. Flash-
    attn is rejected. Regress against future drift where someone returns
    `'flash_attention_2'` etc."""
    adapter = Zaya1Adapter()
    for mode in ("bf16", "da_qad"):
        result = adapter.required_attn_implementation(mode)  # type: ignore[arg-type]
        assert result in ("eager", "sdpa"), (
            f"required_attn_implementation({mode!r}) returned {result!r}; "
            f"only 'eager' and 'sdpa' are permitted by LLR-0026."
        )


@pytest.mark.parametrize(
    ("mode", "expected_teacher_attn", "expected_student_attn"),
    [
        ("bf16", "sdpa", "sdpa"),
        # Phase 7.2: teacher has no KV hooks; only the student needs eager.
        ("da_qad", "sdpa", "eager"),
    ],
)
def test_load_teacher_and_student_threads_mode_into_loaders(
    mode: str, expected_teacher_attn: str, expected_student_attn: str
) -> None:
    """LLR-0026 AC #4 (Phase 7.2): `load_teacher_and_student(mode=mode)` MUST
    thread the mode into `required_attn_implementation(mode, role=...)` for
    each model separately. Teacher (no KV hooks) and student (KV-quant
    hooks) may receive different `attn_implementation` values.

    No GPU / no HF download needed — `_load_one` is patched to a MagicMock,
    and we verify the call kwargs.
    """
    adapter = Zaya1Adapter()
    teacher_cfg = TeacherConfig(
        name_or_path="dummy/teacher",
        revision="main",
        torch_dtype="bfloat16",
        attn_implementation="sdpa",  # YAML value — adapter overrides per mode.
    )
    student_cfg = StudentConfig(
        source="dummy/student",
        torch_dtype="bfloat16",
        attn_implementation="sdpa",  # YAML value — adapter overrides per mode.
    )
    fake_teacher = MagicMock(name="teacher_model")
    fake_student = MagicMock(name="student_model")
    fake_tokenizer = MagicMock(name="tokenizer")
    fake_tokenizer.pad_token_id = 0

    accelerator = MagicMock(name="accelerator")
    accelerator.is_main_process = False  # skip the layer-count log path.

    with (
        patch.object(
            Zaya1Adapter,
            "_load_one",
            side_effect=[fake_teacher, fake_student],
        ) as mock_load_one,
        patch(
            "kdr.adapters.zaya1_8b.AutoTokenizer.from_pretrained",
            return_value=fake_tokenizer,
        ),
    ):
        teacher, student, tokenizer = adapter.load_teacher_and_student(
            accelerator,
            teacher_cfg=teacher_cfg,
            student_cfg=student_cfg,
            mode=mode,  # type: ignore[arg-type]
        )

    assert teacher is fake_teacher
    assert student is fake_student
    assert tokenizer is fake_tokenizer

    # First call is teacher (LLR-0023 load order); second is student.
    assert mock_load_one.call_count == 2, "expected one _load_one per model"
    teacher_call = mock_load_one.call_args_list[0]
    student_call = mock_load_one.call_args_list[1]
    assert teacher_call.kwargs["role"] == "TEACHER"
    assert student_call.kwargs["role"] == "STUDENT"
    assert teacher_call.kwargs["attn_implementation"] == expected_teacher_attn, (
        f"teacher _load_one attn_implementation="
        f"{teacher_call.kwargs['attn_implementation']!r}, expected "
        f"{expected_teacher_attn!r} for mode={mode!r}."
    )
    assert student_call.kwargs["attn_implementation"] == expected_student_attn, (
        f"student _load_one attn_implementation="
        f"{student_call.kwargs['attn_implementation']!r}, expected "
        f"{expected_student_attn!r} for mode={mode!r}."
    )
