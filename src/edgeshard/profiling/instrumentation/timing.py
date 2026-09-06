"""Latency timing instruments (Phase 2 spec §11.1).

``CudaEventTimer`` is the primary GPU timing mechanism: paired
``torch.cuda.Event(enable_timing=True)`` events recorded on the relevant
stream, synchronized before reading, returning *device* elapsed
milliseconds. Host wall-clock time is never the primary GPU latency —
kernel launches are asynchronous, so a wall clock around a launch
measures dispatch, not execution.

``WallClockTimer`` (``time.perf_counter_ns``) serves CPU, network, and
control-path measurements and diagnostics only.

Both implement the :class:`Timer` protocol so the benchmark harness can
switch timing backends without touching workload code.
"""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable

import torch

from edgeshard.profiling.domain.experiment import ProfilingErrorCategory
from edgeshard.profiling.domain.measurement import TimeUnit
from edgeshard.profiling.errors import ProfilingError


@runtime_checkable
class Timer(Protocol):
    """A start/stop latency instrument reporting elapsed milliseconds."""

    @property
    def unit(self) -> TimeUnit:
        """The time unit ``stop`` returns (always milliseconds in v1)."""
        ...

    def start(self) -> None:
        """Begin one measurement interval."""
        ...

    def stop(self) -> float:
        """End the interval and return its elapsed duration."""
        ...


class WallClockTimer:
    """Host wall-clock timing via ``time.perf_counter_ns`` (spec §11.1).

    For CPU/network/control-path measurements and diagnostics; never the
    primary GPU latency mechanism.
    """

    def __init__(self) -> None:
        self._start_ns: int | None = None

    @property
    def unit(self) -> TimeUnit:
        return TimeUnit.MILLISECONDS

    def start(self) -> None:
        self._start_ns = time.perf_counter_ns()

    def stop(self) -> float:
        if self._start_ns is None:
            raise ValueError("WallClockTimer.stop() called before start()")
        elapsed_ns = time.perf_counter_ns() - self._start_ns
        self._start_ns = None
        return elapsed_ns / 1_000_000.0


class CudaEventTimer:
    """Device-side timing with paired CUDA events (spec §11.1).

    Contract:

    1. create start/end ``torch.cuda.Event(enable_timing=True)``;
    2. record both on the relevant stream (the current stream by
       default, or an explicit one);
    3. synchronize before reading;
    4. return device elapsed milliseconds;
    5. never fall back to host wall-clock latency.

    Construction fails explicitly with a typed
    :class:`~edgeshard.profiling.errors.ProfilingError` when CUDA is
    unavailable — a GPU benchmark on a CPU-only host is a configuration
    error, not a silently degraded measurement.
    """

    def __init__(
        self,
        device_index: int | None = None,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        if not torch.cuda.is_available():
            raise ProfilingError(
                ProfilingErrorCategory.INTERNAL_ERROR,
                "CUDA event timing requires an available CUDA device",
            )
        self._device = (
            torch.device("cuda", device_index)
            if device_index is not None
            else torch.device("cuda", torch.cuda.current_device())
        )
        self._stream = stream
        self._start_event: torch.cuda.Event | None = None
        self._end_event: torch.cuda.Event | None = None

    @property
    def unit(self) -> TimeUnit:
        return TimeUnit.MILLISECONDS

    def start(self) -> None:
        self._start_event = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        self._end_event = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        self._start_event.record(self._stream)

    def stop(self) -> float:
        if self._start_event is None or self._end_event is None:
            raise ValueError("CudaEventTimer.stop() called before start()")
        start_event, end_event = self._start_event, self._end_event
        self._start_event = None
        self._end_event = None
        end_event.record(self._stream)
        end_event.synchronize()
        return float(start_event.elapsed_time(end_event))
