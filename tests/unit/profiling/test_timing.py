"""Timing instruments (spec §11.1).

CUDA-event behavior is unit-tested against a fake ``torch.cuda`` so the
contract (paired events, stream recording, synchronize-before-read,
device milliseconds) is verified on CPU-only hosts. Real-GPU verification
is a P2B DoD item on the RTX host and out of scope here.
"""

from __future__ import annotations

import time
from typing import Any, ClassVar

import pytest
import torch

from edgeshard.profiling.domain.experiment import ProfilingErrorCategory
from edgeshard.profiling.domain.measurement import TimeUnit
from edgeshard.profiling.errors import ProfilingError
from edgeshard.profiling.instrumentation.timing import (
    CudaEventTimer,
    Timer,
    WallClockTimer,
)


def test_wall_clock_timer_measures_elapsed_milliseconds() -> None:
    timer = WallClockTimer()
    assert timer.unit is TimeUnit.MILLISECONDS
    timer.start()
    time.sleep(0.02)
    elapsed = timer.stop()
    assert 5.0 <= elapsed < 5_000.0


def test_wall_clock_timer_requires_start_before_stop() -> None:
    with pytest.raises(ValueError, match="before start"):
        WallClockTimer().stop()


def test_wall_clock_timer_intervals_are_independent() -> None:
    timer = WallClockTimer()
    timer.start()
    time.sleep(0.01)
    first = timer.stop()
    timer.start()
    second = timer.stop()
    assert second < first + 50.0  # restart does not accumulate


def test_timers_satisfy_timer_protocol() -> None:
    assert isinstance(WallClockTimer(), Timer)


def test_cuda_timer_fails_explicitly_without_cuda() -> None:
    """§11.1: no silent wall-clock fallback on a CPU-only host."""
    if torch.cuda.is_available():
        pytest.skip("CUDA available: explicit-failure path not observable")
    with pytest.raises(ProfilingError) as excinfo:
        CudaEventTimer()
    assert excinfo.value.category is ProfilingErrorCategory.INTERNAL_ERROR


class _FakeEvent:
    """Stand-in for ``torch.cuda.Event(enable_timing=True)``."""

    log: ClassVar[list[tuple[str, Any]]] = []
    elapsed_ms: float = 12.5

    def __init__(self, enable_timing: bool = False) -> None:
        assert enable_timing, "CUDA latency timing requires enable_timing=True"
        _FakeEvent.log.append(("create", None))

    def record(self, stream: Any = None) -> None:
        _FakeEvent.log.append(("record", stream))

    def synchronize(self) -> None:
        _FakeEvent.log.append(("synchronize", None))

    def elapsed_time(self, end_event: _FakeEvent) -> float:
        return _FakeEvent.elapsed_ms


@pytest.fixture
def fake_cuda(monkeypatch: pytest.MonkeyPatch) -> type[_FakeEvent]:
    _FakeEvent.log = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "Event", _FakeEvent)
    return _FakeEvent


def test_cuda_timer_contract(fake_cuda: type[_FakeEvent]) -> None:
    timer = CudaEventTimer()
    assert isinstance(timer, Timer)
    assert timer.unit is TimeUnit.MILLISECONDS
    timer.start()
    elapsed = timer.stop()
    kinds = [kind for kind, _ in fake_cuda.log]
    # create start+end events, record both, synchronize, then read.
    assert kinds == ["create", "create", "record", "record", "synchronize"]
    assert elapsed == 12.5


def test_cuda_timer_records_on_given_stream(fake_cuda: type[_FakeEvent]) -> None:
    sentinel = object()
    timer = CudaEventTimer(stream=sentinel)  # type: ignore[arg-type]
    timer.start()
    timer.stop()
    recorded = [stream for kind, stream in fake_cuda.log if kind == "record"]
    assert recorded == [sentinel, sentinel]


def test_cuda_timer_requires_start_before_stop(fake_cuda: type[_FakeEvent]) -> None:
    with pytest.raises(ValueError, match="before start"):
        CudaEventTimer(device_index=0).stop()


def test_cuda_timer_is_reusable_across_intervals(fake_cuda: type[_FakeEvent]) -> None:
    timer = CudaEventTimer()
    timer.start()
    timer.stop()
    fake_cuda.log.clear()
    timer.start()
    assert timer.stop() == 12.5
    # A fresh event pair is created for every interval.
    assert len([kind for kind, _ in fake_cuda.log if kind == "create"]) == 2
