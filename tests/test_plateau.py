"""Tests for the plateau-guard auto-rewind (Phase 7.2).

The `PlateauCollapseError` exception carries telemetry for the bootstrap script
to resume from the best-pointed partial when the EMA-tracked `raw_kl`
exceeds `plateau_guard_multiplier × best` for `plateau_guard_consecutive`
windows past `plateau_guard_min_step`.

These tests exercise the exception class itself. A full integration test
(driving `_commit_window` end-to-end) requires constructing a `_LoopState`
with a real accelerator and student; that is covered by the B200 dry-launch
verification step in the rollout plan, not at unit-test scope.
"""

from __future__ import annotations

import pytest

from kdr.training.loop import PlateauCollapseError


def test_plateau_collapse_carries_telemetry() -> None:
    """Exception keeps step / ema / best / best_step accessible."""
    exc = PlateauCollapseError(step=42, ema=1.8, best=0.5, best_step=20)
    assert exc.step == 42
    assert exc.ema == pytest.approx(1.8)
    assert exc.best == pytest.approx(0.5)
    assert exc.best_step == 20


def test_plateau_collapse_message_includes_step_and_ratio() -> None:
    """Exception message names the offending step and EMA."""
    exc = PlateauCollapseError(step=42, ema=1.8, best=0.5, best_step=20)
    msg = str(exc)
    assert "step=42" in msg
    assert "1.8" in msg
    assert "0.5" in msg
    assert "step=20" in msg


def test_plateau_collapse_is_exception_subclass() -> None:
    """Subclass `Exception` so bootstrap scripts can catch it as a normal
    Python exception (vs. SystemExit / KeyboardInterrupt)."""
    exc = PlateauCollapseError(step=1, ema=1.0, best=0.1, best_step=0)
    assert isinstance(exc, Exception)
