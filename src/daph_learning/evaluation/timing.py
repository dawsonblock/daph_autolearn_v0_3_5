"""v0.3.6 Phase 3.3: accurate GPU timing via PyTorch CUDA events.

The v0.3.5 harness used ``time.perf_counter()`` for all latency
measurements. On CUDA this is wrong: kernel launches are asynchronous,
so the host clock returns before the kernel finishes, and the measured
latency reflects Python overhead + queue depth rather than the actual
GPU work. The reported numbers were systematically understated and
noisy, which made the latency-vs-accuracy trade-off in the utility
oracle unreliable.

This module provides a small context manager that records CUDA start
and stop events around a region of interest and synchronizes before
reading the elapsed time. On CPU (no CUDA available) it falls back to
``time.perf_counter()`` so callers do not need to branch.

Usage::

    from daph_learning.evaluation.timing import cuda_timed

    with cuda_timed() as t:
        out = model.generate(**encoded, max_new_tokens=64)
    print(t.elapsed_ms)  # GPU time in milliseconds

For batched benchmarking, ``cuda_timed_accumulator`` accumulates the
total GPU time across multiple regions and reports the per-call mean.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Iterator


def _torch_cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except (ImportError, ModuleNotFoundError):
        return False


@dataclass
class TimingResult:
    """One timing measurement."""
    elapsed_ms: float
    device: str  # "cuda" or "cpu"
    cuda_event_timing: bool


@dataclass
class TimingAccumulator:
    """Accumulates multiple timing measurements for mean / total reporting."""
    samples: list[TimingResult] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.samples)

    @property
    def total_ms(self) -> float:
        return sum(s.elapsed_ms for s in self.samples)

    @property
    def mean_ms(self) -> float:
        return self.total_ms / self.n if self.n else 0.0

    @property
    def all_cuda(self) -> bool:
        return bool(self.samples) and all(s.device == "cuda" for s in self.samples)

    def add(self, result: TimingResult) -> None:
        self.samples.append(result)

    def summary(self) -> dict:
        """A JSON-serializable summary suitable for the run manifest."""
        return {
            "n_samples": self.n,
            "total_ms": round(self.total_ms, 4),
            "mean_ms": round(self.mean_ms, 4),
            "device": self.samples[0].device if self.samples else "unknown",
            "cuda_event_timing": bool(self.samples) and self.samples[0].cuda_event_timing,
        }


@contextmanager
def cuda_timed(
    *,
    device: object | None = None,
    accumulator: TimingAccumulator | None = None,
) -> Iterator[TimingResult]:
    """Time a region of interest using CUDA events when available.

    On CUDA, this records ``start`` and ``stop`` events around the
    ``with`` block, calls ``torch.cuda.synchronize()``, and reads
    ``start.elapsed_time(stop)`` for an accurate GPU-side millisecond
    measurement.

    On CPU, falls back to ``time.perf_counter()``. The returned
    ``TimingResult`` is populated at context exit and may be read from
    the ``as`` binding or from the optional accumulator.

    Parameters
    ----------
    device : torch.device or int, optional
        The CUDA device to record events on. Defaults to the current
        CUDA device. Ignored on CPU.
    accumulator : TimingAccumulator, optional
        If supplied, the result is appended to this accumulator for
        batched mean / total reporting.
    """
    import time

    result = TimingResult(elapsed_ms=0.0, device="cpu", cuda_event_timing=False)
    cuda_ready = False
    if _torch_cuda_available():
        import torch
        try:
            if device is None:
                dev = torch.cuda.current_device()
            elif isinstance(device, int):
                dev = device
            else:
                dev = device.index if hasattr(device, "index") and device.index is not None else torch.cuda.current_device()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            cuda_ready = True
        except (RuntimeError, AttributeError, IndexError):
            # Fall through to CPU path if CUDA event setup fails (no device,
            # driver issue, device index out of range).
            pass

    if cuda_ready:
        try:
            yield result
        finally:
            end.record()
            torch.cuda.synchronize(dev)
            result.elapsed_ms = float(start.elapsed_time(end))
            result.device = "cuda"
            result.cuda_event_timing = True
            if accumulator is not None:
                accumulator.add(result)
        return

    t0 = time.perf_counter()
    try:
        yield result
    finally:
        result.elapsed_ms = (time.perf_counter() - t0) * 1000.0
        result.device = "cpu"
        result.cuda_event_timing = False
        if accumulator is not None:
            accumulator.add(result)


@contextmanager
def cuda_timed_accumulator() -> Iterator[tuple[TimingAccumulator, Callable]]:
    """Convenience: a fresh accumulator + a context-manager factory.

    Yields ``(accumulator, timed_ctx)`` where ``timed_ctx`` is a
    callable that returns a ``cuda_timed`` context manager bound to the
    accumulator. This is the cleanest pattern for benchmarking loops::

        with cuda_timed_accumulator() as (acc, timed):
            for batch in batches:
                with timed():
                    out = model.generate(**batch)
        print(acc.summary())
    """
    acc = TimingAccumulator()

    @contextmanager
    def _bound():
        with cuda_timed(accumulator=acc) as r:
            yield r

    yield acc, _bound
