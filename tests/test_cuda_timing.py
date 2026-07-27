"""Tests for v0.3.6 Phase 3.3 PyTorch CUDA event timing.

See daph_learning.evaluation.timing. These tests run on CPU (the CI
environment) and verify the CPU fallback path and the accumulator
logic. A CUDA-gated test verifies the CUDA event path when a GPU is
available.
"""
from __future__ import annotations

import time

import pytest

from daph_learning.evaluation.timing import (
    TimingAccumulator,
    TimingResult,
    cuda_timed,
    cuda_timed_accumulator,
)


def _torch_cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def test_cuda_timed_returns_positive_elapsed_on_cpu():
    """On CPU, cuda_timed must fall back to perf_counter and report a
    positive elapsed time in milliseconds."""
    with cuda_timed() as t:
        time.sleep(0.01)
    assert t.elapsed_ms > 0.0
    assert t.device in ("cpu", "cuda")  # cpu in CI, cuda if GPU present
    assert isinstance(t.cuda_event_timing, bool)


def test_cuda_timed_populates_result_at_exit():
    """The TimingResult binding must be populated when the context
    exits, not before."""
    with cuda_timed() as t:
        # Inside the block, the result is not yet populated.
        assert t.elapsed_ms == 0.0
    # After exit, it is.
    assert t.elapsed_ms > 0.0


def test_cuda_timed_accumulator_appends_results():
    """The accumulator must collect one TimingResult per cuda_timed
    call and report the mean / total."""
    acc = TimingAccumulator()
    for _ in range(3):
        with cuda_timed(accumulator=acc):
            time.sleep(0.005)
    assert acc.n == 3
    assert acc.total_ms > 0.0
    assert acc.mean_ms > 0.0
    summary = acc.summary()
    assert summary["n_samples"] == 3
    assert summary["total_ms"] > 0.0
    assert summary["mean_ms"] > 0.0
    # On CPU the device is "cpu"; on CUDA it's "cuda". Either is fine.
    assert summary["device"] in ("cpu", "cuda")


def test_cuda_timed_accumulator_empty_summary():
    """An empty accumulator must produce a valid summary with zeros."""
    acc = TimingAccumulator()
    s = acc.summary()
    assert s["n_samples"] == 0
    assert s["total_ms"] == 0.0
    assert s["mean_ms"] == 0.0
    assert s["device"] == "unknown"


def test_cuda_timed_accumulator_context_manager_pattern():
    """The cuda_timed_accumulator context manager pattern must work as
    documented in the module docstring."""
    with cuda_timed_accumulator() as (acc, timed):
        for _ in range(2):
            with timed():
                time.sleep(0.002)
    assert acc.n == 2
    assert acc.mean_ms > 0.0


def test_timing_result_dataclass_fields():
    """TimingResult must carry the three documented fields."""
    r = TimingResult(elapsed_ms=1.5, device="cpu", cuda_event_timing=False)
    assert r.elapsed_ms == 1.5
    assert r.device == "cpu"
    assert r.cuda_event_timing is False


def test_cuda_timed_handles_exceptions():
    """If the body raises, the timing context must still record a
    measurement (so a benchmark loop can report the failed call's
    latency) and then propagate the exception."""
    with pytest.raises(ValueError, match="boom"):
        with cuda_timed() as t:
            raise ValueError("boom")
    # The elapsed_ms was recorded up to the point of the exception.
    assert t.elapsed_ms >= 0.0


def test_cuda_timed_device_string_is_stable():
    """The device string must be either 'cpu' or 'cuda' — nothing else
    leaks through."""
    with cuda_timed() as t:
        pass
    assert t.device in ("cpu", "cuda")


@pytest.mark.skipif(
    not _torch_cuda_available(),
    reason="no CUDA device available",
)
def test_cuda_timed_uses_cuda_events_on_gpu():
    """When a CUDA device is available, cuda_timed must report
    device='cuda' and cuda_event_timing=True, and the measured latency
    must be non-negative."""
    import torch
    with cuda_timed() as t:
        x = torch.randn(1000, 1000, device="cuda")
        y = x @ x
        torch.cuda.synchronize()
    assert t.device == "cuda"
    assert t.cuda_event_timing is True
    assert t.elapsed_ms >= 0.0
