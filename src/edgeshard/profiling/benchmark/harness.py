"""The one reusable benchmark lifecycle (Phase 2 spec §12).

Every profiler (layer, module, operator, and later network control
paths) measures through this harness so the lifecycle is identical
everywhere::

    prepare → validate environment → warmup → reset memory peaks
    → measurement loop → telemetry → summary → cleanup

Guarantees:

- ``cleanup`` runs even when any earlier step fails (``finally``);
- workload and timer failures are wrapped into typed
  :class:`~edgeshard.profiling.errors.ProfilingError`
  (``BENCHMARK_FAILED``) with the original exception as ``__cause__`` —
  failures are never zero latency or empty results (§42);
- warmup durations are discarded; only measured runs produce samples;
- memory peak counters are reset after warmup, immediately before the
  measurement interval (§14.1);
- the sampling policy alone decides run counts (§13) — the harness
  hard-codes no repetition count.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from edgeshard.profiling.benchmark.sampling import SamplingPolicy, SamplingState
from edgeshard.profiling.domain.experiment import ProfilingErrorCategory
from edgeshard.profiling.domain.measurement import (
    AllocatorMemoryMetrics,
    PhysicalMemoryMetrics,
    SampleSummary,
    TelemetryContextMetrics,
    summarize_samples,
)
from edgeshard.profiling.errors import ProfilingError
from edgeshard.profiling.instrumentation.memory import (
    MemoryInstrumentation,
    PhysicalMemoryProbe,
)
from edgeshard.profiling.instrumentation.telemetry import TelemetryContextCollector
from edgeshard.profiling.instrumentation.timing import Timer

logger = logging.getLogger("profiling.benchmark.harness")


@runtime_checkable
class BenchmarkWorkload(Protocol):
    """Anything the harness can measure (spec §12).

    ``prepare`` loads/materializes inputs; ``run_once`` executes exactly
    one iteration (timed only in the measurement loop); ``reset`` clears
    transient state after warmup so measurement starts from a defined
    state; ``cleanup`` releases resources and is always called.
    """

    def prepare(self) -> None: ...

    def run_once(self) -> None: ...

    def reset(self) -> None: ...

    def cleanup(self) -> None: ...


@dataclass(frozen=True)
class InstrumentationBundle:
    """The instruments one benchmark run uses (spec §12).

    Only ``timer`` is mandatory; memory/telemetry instruments are
    optional and their metrics are ``None`` when absent or unavailable
    (§52.2). ``environment_check`` runs during environment validation
    and must itself raise typed errors (e.g. ``INSUFFICIENT_MEMORY``).
    """

    timer: Timer
    memory: MemoryInstrumentation | None = None
    physical_memory: PhysicalMemoryProbe | None = None
    telemetry: TelemetryContextCollector | None = None
    environment_check: Callable[[], None] | None = None


@dataclass(frozen=True)
class BenchmarkResult:
    """Outcome of one harness run: samples plus every context metric."""

    started_at: datetime
    finished_at: datetime
    warmup_runs: int
    measured_runs: int
    samples_ms: tuple[float, ...]
    summary: SampleSummary
    allocator_memory: AllocatorMemoryMetrics | None = None
    physical_memory: PhysicalMemoryMetrics | None = None
    telemetry: TelemetryContextMetrics | None = None


class BenchmarkHarness:
    """Executes the §12 lifecycle against any workload/policy/instruments."""

    def run(
        self,
        workload: BenchmarkWorkload,
        *,
        sampling_policy: SamplingPolicy,
        instrumentation: InstrumentationBundle,
    ) -> BenchmarkResult:
        started_at = datetime.now(UTC)
        self._guard(workload.prepare, "workload prepare failed")
        completed = False
        try:
            self._validate_environment(instrumentation)
            self._warmup(workload, sampling_policy)
            self._guard(workload.reset, "workload reset failed")
            if instrumentation.memory is not None:
                self._guard(instrumentation.memory.open, "memory instrumentation open failed")
            if instrumentation.physical_memory is not None:
                self._guard(
                    instrumentation.physical_memory.open,
                    "physical memory probe open failed",
                )
            samples_ms = self._measure(workload, sampling_policy, instrumentation)
            allocator = (
                self._guard(instrumentation.memory.close, "memory instrumentation close failed")
                if instrumentation.memory is not None
                else None
            )
            physical = (
                self._guard(
                    instrumentation.physical_memory.close, "physical memory probe close failed"
                )
                if instrumentation.physical_memory is not None
                else None
            )
            telemetry = (
                self._guard(
                    instrumentation.telemetry.capture_final, "telemetry capture failed"
                )
                if instrumentation.telemetry is not None
                else None
            )
            completed = True
        finally:
            self._cleanup(workload, completed)
        if not samples_ms:
            raise ProfilingError(
                ProfilingErrorCategory.BENCHMARK_FAILED,
                "sampling policy produced no measured runs",
            )
        finished_at = datetime.now(UTC)
        return BenchmarkResult(
            started_at=started_at,
            finished_at=finished_at,
            warmup_runs=sampling_policy.warmup_runs,
            measured_runs=len(samples_ms),
            samples_ms=tuple(samples_ms),
            summary=summarize_samples(samples_ms),
            allocator_memory=allocator,
            physical_memory=physical,
            telemetry=telemetry,
        )

    def _cleanup(self, workload: BenchmarkWorkload, completed: bool) -> None:
        """Always run cleanup; never mask the failure already in flight."""
        try:
            workload.cleanup()
        except Exception as exc:
            if completed:
                raise ProfilingError(
                    ProfilingErrorCategory.BENCHMARK_FAILED,
                    f"workload cleanup failed: {exc}",
                ) from exc
            logger.warning("workload cleanup failed during error handling: %s", exc)

    def _validate_environment(self, instrumentation: InstrumentationBundle) -> None:
        if instrumentation.environment_check is not None:
            self._guard(instrumentation.environment_check, "environment check failed")
        if instrumentation.telemetry is not None:
            # May raise DEVICE_BUSY when contamination is rejected (§15);
            # typed errors pass through the guard untouched.
            self._guard(instrumentation.telemetry.capture_initial, "telemetry capture failed")

    def _warmup(self, workload: BenchmarkWorkload, policy: SamplingPolicy) -> None:
        for _ in range(policy.warmup_runs):
            self._guard(workload.run_once, "workload warmup run failed")

    def _measure(
        self,
        workload: BenchmarkWorkload,
        policy: SamplingPolicy,
        instrumentation: InstrumentationBundle,
    ) -> list[float]:
        timer = instrumentation.timer
        samples: list[float] = []
        state = SamplingState(warmup_runs=policy.warmup_runs)
        while policy.should_continue(state):
            self._guard(timer.start, "timer start failed")
            self._guard(workload.run_once, "workload measured run failed")
            elapsed_ms = self._guard(timer.stop, "timer stop failed")
            if not math.isfinite(elapsed_ms) or elapsed_ms < 0.0:
                raise ProfilingError(
                    ProfilingErrorCategory.BENCHMARK_FAILED,
                    f"timer returned an invalid elapsed duration: {elapsed_ms!r}",
                )
            samples.append(elapsed_ms)
            if instrumentation.physical_memory is not None:
                self._guard(
                    instrumentation.physical_memory.poll, "physical memory probe poll failed"
                )
            state = SamplingState(
                warmup_runs=state.warmup_runs,
                measured_runs=state.measured_runs + 1,
                accumulated_measurement_ms=state.accumulated_measurement_ms + elapsed_ms,
            )
        return samples

    @staticmethod
    def _guard[T](action: Callable[[], T], failure_message: str) -> T:
        """Run ``action``, wrapping non-typed failures (§42)."""
        try:
            return action()
        except ProfilingError:
            raise
        except Exception as exc:
            raise ProfilingError(
                ProfilingErrorCategory.BENCHMARK_FAILED,
                f"{failure_message}: {exc}",
            ) from exc
