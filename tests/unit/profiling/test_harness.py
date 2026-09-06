"""Benchmark harness lifecycle (spec §12, §42, §52)."""

from __future__ import annotations

import statistics

import pytest

from edgeshard.profiling.benchmark.harness import (
    BenchmarkHarness,
    BenchmarkWorkload,
    InstrumentationBundle,
)
from edgeshard.profiling.benchmark.sampling import (
    DurationSamplingPolicy,
    SamplingPolicy,
    SamplingState,
)
from edgeshard.profiling.domain.experiment import ProfilingErrorCategory
from edgeshard.profiling.domain.measurement import (
    AllocatorMemoryMetrics,
    SampleSummary,
    TelemetrySample,
    TimeUnit,
)
from edgeshard.profiling.errors import ProfilingError
from edgeshard.profiling.instrumentation.memory import PhysicalMemoryProbe
from edgeshard.profiling.instrumentation.telemetry import (
    ContaminationAction,
    DeviceObservation,
    TelemetryContextCollector,
)

Events = list[str]


class ScriptedTimer:
    """Deterministic stand-in for a device timer."""

    def __init__(self, events: Events, elapsed_ms: float) -> None:
        self._events = events
        self._elapsed_ms = elapsed_ms

    @property
    def unit(self) -> TimeUnit:
        return TimeUnit.MILLISECONDS

    def start(self) -> None:
        self._events.append("timer_start")

    def stop(self) -> float:
        self._events.append("timer_stop")
        return self._elapsed_ms


class RecordingWorkload:
    """Records lifecycle calls; optionally fails on a specific run_once."""

    def __init__(self, events: Events, fail_on_run: int | None = None) -> None:
        self._events = events
        self._fail_on_run = fail_on_run
        self.run_calls = 0

    def prepare(self) -> None:
        self._events.append("prepare")

    def run_once(self) -> None:
        self.run_calls += 1
        self._events.append("run_once")
        if self.run_calls == self._fail_on_run:
            raise RuntimeError("workload exploded")

    def reset(self) -> None:
        self._events.append("reset")

    def cleanup(self) -> None:
        self._events.append("cleanup")


class FakeMemory:
    def __init__(self, events: Events, metrics: AllocatorMemoryMetrics | None) -> None:
        self._events = events
        self._metrics = metrics

    def open(self) -> None:
        self._events.append("memory_open")

    def close(self) -> AllocatorMemoryMetrics | None:
        self._events.append("memory_close")
        return self._metrics


class ScriptedTelemetry:
    def __init__(self, observations: list[DeviceObservation | None]) -> None:
        self._observations = list(observations)

    def capture(self) -> DeviceObservation | None:
        return self._observations.pop(0) if self._observations else None


ALLOCATOR_METRICS = AllocatorMemoryMetrics(
    allocated_before=10,
    reserved_before=20,
    peak_allocated=30,
    peak_reserved=40,
    allocated_after=15,
    reserved_after=20,
)
IDLE = DeviceObservation(device_id="GPU-uuid-1", utilization=1.0)
BUSY = DeviceObservation(device_id="GPU-uuid-1", utilization=95.0)


def _policy(**overrides: object) -> DurationSamplingPolicy:
    # target below min_runs x scripted elapsed so tests stop at min_runs
    base: dict[str, object] = {
        "min_warmups": 2,
        "min_runs": 3,
        "max_runs": 20,
        "target_duration_ms": 5.0,
    }
    base.update(overrides)
    return DurationSamplingPolicy(**base)  # type: ignore[arg-type]


def test_workload_and_policy_satisfy_protocols() -> None:
    events: Events = []
    assert isinstance(RecordingWorkload(events), BenchmarkWorkload)
    assert isinstance(_policy(), SamplingPolicy)


def test_lifecycle_order_and_minimum_runs() -> None:
    """§12: prepare → validate → warmup → reset/peaks → loop → telemetry →
    summary → cleanup, with the policy alone deciding run counts."""
    events: Events = []
    workload = RecordingWorkload(events)
    memory = FakeMemory(events, ALLOCATOR_METRICS)
    bundle = InstrumentationBundle(timer=ScriptedTimer(events, 10.0), memory=memory)
    result = BenchmarkHarness().run(
        workload, sampling_policy=_policy(), instrumentation=bundle
    )
    assert events == [
        "prepare",
        "run_once",  # warmup 1
        "run_once",  # warmup 2
        "reset",
        "memory_open",  # peaks reset after warmup, before measuring (§14.1)
        "timer_start", "run_once", "timer_stop",
        "timer_start", "run_once", "timer_stop",
        "timer_start", "run_once", "timer_stop",
        "memory_close",
        "cleanup",
    ]
    assert result.warmup_runs == 2
    assert result.measured_runs == 3
    assert result.samples_ms == (10.0, 10.0, 10.0)
    assert result.allocator_memory == ALLOCATOR_METRICS


def test_duration_target_stops_the_loop() -> None:
    events: Events = []
    bundle = InstrumentationBundle(timer=ScriptedTimer(events, 250.0))
    result = BenchmarkHarness().run(
        RecordingWorkload(events),
        sampling_policy=_policy(min_runs=2, target_duration_ms=1000.0),
        instrumentation=bundle,
    )
    # 4 x 250 ms reaches the 1000 ms target; a 5th run never starts.
    assert result.measured_runs == 4


def test_maximum_runs_cap_the_loop() -> None:
    events: Events = []
    bundle = InstrumentationBundle(timer=ScriptedTimer(events, 1.0))
    result = BenchmarkHarness().run(
        RecordingWorkload(events),
        sampling_policy=_policy(min_runs=1, max_runs=7, target_duration_ms=1000.0),
        instrumentation=bundle,
    )
    assert result.measured_runs == 7


def test_summary_describes_measured_samples_only() -> None:
    events: Events = []
    bundle = InstrumentationBundle(timer=ScriptedTimer(events, 12.0))
    result = BenchmarkHarness().run(
        RecordingWorkload(events), sampling_policy=_policy(), instrumentation=bundle
    )
    assert result.summary == SampleSummary(12.0, 12.0, 0.0, 12.0, 12.0, None)
    assert result.summary.mean == pytest.approx(statistics.fmean(result.samples_ms))
    assert result.started_at.tzinfo is not None
    assert result.finished_at >= result.started_at


def test_workload_failure_is_typed_and_cleanup_still_runs() -> None:
    """§42: failures are typed errors, never zero latency or empty results."""
    events: Events = []
    # Warmups are calls 1-2; the second measured run is call 4.
    workload = RecordingWorkload(events, fail_on_run=4)
    bundle = InstrumentationBundle(timer=ScriptedTimer(events, 10.0))
    with pytest.raises(ProfilingError) as excinfo:
        BenchmarkHarness().run(
            workload, sampling_policy=_policy(), instrumentation=bundle
        )
    assert excinfo.value.category is ProfilingErrorCategory.BENCHMARK_FAILED
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert events[-1] == "cleanup"


def test_prepare_failure_skips_cleanup_and_is_typed() -> None:
    events: Events = []

    class BrokenPrepare(RecordingWorkload):
        def prepare(self) -> None:
            events.append("prepare")
            raise RuntimeError("cannot materialize inputs")

    bundle = InstrumentationBundle(timer=ScriptedTimer(events, 10.0))
    with pytest.raises(ProfilingError) as excinfo:
        BenchmarkHarness().run(
            BrokenPrepare(events), sampling_policy=_policy(), instrumentation=bundle
        )
    assert excinfo.value.category is ProfilingErrorCategory.BENCHMARK_FAILED
    assert events == ["prepare"]  # nothing was prepared, nothing to clean up


def test_typed_environment_errors_pass_through() -> None:
    events: Events = []

    def check() -> None:
        raise ProfilingError(
            ProfilingErrorCategory.INSUFFICIENT_MEMORY, "free VRAM below model footprint"
        )

    bundle = InstrumentationBundle(
        timer=ScriptedTimer(events, 10.0), environment_check=check
    )
    with pytest.raises(ProfilingError) as excinfo:
        BenchmarkHarness().run(
            RecordingWorkload(events), sampling_policy=_policy(), instrumentation=bundle
        )
    assert excinfo.value.category is ProfilingErrorCategory.INSUFFICIENT_MEMORY
    assert "run_once" not in events  # rejected before any warmup
    assert events[-1] == "cleanup"


def test_contamination_reject_stops_before_measuring() -> None:
    """§15: an obviously busy device must not pass as a clean run."""
    events: Events = []
    collector = TelemetryContextCollector(
        ScriptedTelemetry([BUSY]),
        contamination_threshold_percent=50.0,
        contamination_action=ContaminationAction.REJECT,
    )
    bundle = InstrumentationBundle(timer=ScriptedTimer(events, 10.0), telemetry=collector)
    with pytest.raises(ProfilingError) as excinfo:
        BenchmarkHarness().run(
            RecordingWorkload(events), sampling_policy=_policy(), instrumentation=bundle
        )
    assert excinfo.value.category is ProfilingErrorCategory.DEVICE_BUSY
    assert "run_once" not in events
    assert events[-1] == "cleanup"


def test_contamination_mark_flows_into_result_context() -> None:
    events: Events = []
    collector = TelemetryContextCollector(
        ScriptedTelemetry([BUSY, IDLE]), contamination_threshold_percent=50.0
    )
    bundle = InstrumentationBundle(timer=ScriptedTimer(events, 10.0), telemetry=collector)
    result = BenchmarkHarness().run(
        RecordingWorkload(events), sampling_policy=_policy(), instrumentation=bundle
    )
    assert result.telemetry is not None
    assert result.telemetry.contaminated is True
    assert result.telemetry.initial == TelemetrySample(device_id="GPU-uuid-1", utilization=95.0)
    assert result.telemetry.final == TelemetrySample(device_id="GPU-uuid-1", utilization=1.0)
    # Telemetry is context only: samples are untouched (§15).
    assert result.samples_ms == (10.0, 10.0, 10.0)


def test_physical_memory_probe_polls_each_measured_run() -> None:
    events: Events = []
    readings = [400, 300, 250, 350, 500]  # open, 3 polls, close
    queue = list(readings)
    probe = PhysicalMemoryProbe(
        "vram:GPU-uuid-1",
        total_bytes=1_000,
        read_available_bytes=lambda: queue.pop(0),
    )
    bundle = InstrumentationBundle(
        timer=ScriptedTimer(events, 10.0), physical_memory=probe
    )
    result = BenchmarkHarness().run(
        RecordingWorkload(events), sampling_policy=_policy(), instrumentation=bundle
    )
    assert queue == []
    assert result.physical_memory is not None
    assert result.physical_memory.used_before == 600
    assert result.physical_memory.used_peak == 750
    assert result.physical_memory.used_after == 500


def test_missing_instruments_yield_none_metrics() -> None:
    """§52.2: metrics that were not measured are None, never guessed."""
    events: Events = []
    bundle = InstrumentationBundle(timer=ScriptedTimer(events, 10.0))
    result = BenchmarkHarness().run(
        RecordingWorkload(events), sampling_policy=_policy(), instrumentation=bundle
    )
    assert result.allocator_memory is None
    assert result.physical_memory is None
    assert result.telemetry is None


def test_invalid_timer_readings_are_typed_failures() -> None:
    events: Events = []
    bundle = InstrumentationBundle(timer=ScriptedTimer(events, -1.0))
    with pytest.raises(ProfilingError, match="invalid elapsed"):
        BenchmarkHarness().run(
            RecordingWorkload(events), sampling_policy=_policy(), instrumentation=bundle
        )


def test_policy_without_runs_is_a_typed_failure() -> None:
    class NoRunPolicy:
        @property
        def warmup_runs(self) -> int:
            return 0

        def should_continue(self, state: SamplingState) -> bool:
            return False

    events: Events = []
    bundle = InstrumentationBundle(timer=ScriptedTimer(events, 10.0))
    with pytest.raises(ProfilingError) as excinfo:
        BenchmarkHarness().run(
            RecordingWorkload(events),
            sampling_policy=NoRunPolicy(),
            instrumentation=bundle,
        )
    assert excinfo.value.category is ProfilingErrorCategory.BENCHMARK_FAILED
    assert events[-1] == "cleanup"


def test_cleanup_failure_after_success_is_typed() -> None:
    events: Events = []

    class BrokenCleanup(RecordingWorkload):
        def cleanup(self) -> None:
            events.append("cleanup")
            raise RuntimeError("release failed")

    bundle = InstrumentationBundle(timer=ScriptedTimer(events, 10.0))
    with pytest.raises(ProfilingError) as excinfo:
        BenchmarkHarness().run(
            BrokenCleanup(events), sampling_policy=_policy(), instrumentation=bundle
        )
    assert excinfo.value.category is ProfilingErrorCategory.BENCHMARK_FAILED


def test_cleanup_failure_does_not_mask_benchmark_failure() -> None:
    events: Events = []

    class DoublyBroken(RecordingWorkload):
        def cleanup(self) -> None:
            events.append("cleanup")
            raise RuntimeError("release failed")

    workload = DoublyBroken(events, fail_on_run=1)  # fails in warmup
    bundle = InstrumentationBundle(timer=ScriptedTimer(events, 10.0))
    with pytest.raises(ProfilingError) as excinfo:
        BenchmarkHarness().run(
            workload, sampling_policy=_policy(), instrumentation=bundle
        )
    # The original failure stays primary; the cleanup failure is logged.
    assert "warmup" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert str(excinfo.value.__cause__) == "workload exploded"
