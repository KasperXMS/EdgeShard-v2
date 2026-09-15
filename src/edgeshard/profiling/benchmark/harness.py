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

import json
import logging
import math
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from edgeshard.profiling.benchmark.sampling import SamplingPolicy, SamplingState
from edgeshard.profiling.domain.experiment import ProfilingErrorCategory
from edgeshard.profiling.domain.measurement import (
    AllocatorMemoryMetrics,
    MeasurementQuality,
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
    requested_min_warmup_runs: int = 0
    warmup_converged: bool = True
    quality: MeasurementQuality | None = None


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
            warmup_samples, warmup_converged = self._warmup(
                workload, sampling_policy, instrumentation.timer
            )
            actual_warmup_runs = (
                len(warmup_samples) if warmup_samples else sampling_policy.warmup_runs
            )
            if not warmup_converged:
                self._raise_unstable_warmup(warmup_samples, sampling_policy)
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
                self._guard(instrumentation.telemetry.capture_final, "telemetry capture failed")
                if instrumentation.telemetry is not None
                else None
            )
            stationarity_reliable = getattr(
                instrumentation.timer, "stationarity_reliable", True
            )
            quality = (
                self._quality(samples_ms)
                if samples_ms and stationarity_reliable
                else None
            )
            enforce_stationarity = getattr(sampling_policy, "enforce_stationarity", True)
            if quality is not None and not quality.stationary and enforce_stationarity:
                self._raise_unstable_measurement(
                    samples_ms, quality, actual_warmup_runs, warmup_converged
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
            warmup_runs=actual_warmup_runs,
            requested_min_warmup_runs=sampling_policy.warmup_runs,
            warmup_converged=warmup_converged,
            measured_runs=len(samples_ms),
            samples_ms=tuple(samples_ms),
            summary=summarize_samples(samples_ms),
            allocator_memory=allocator,
            physical_memory=physical,
            telemetry=telemetry,
            quality=quality,
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

    def _warmup(
        self, workload: BenchmarkWorkload, policy: SamplingPolicy, timer: Timer
    ) -> tuple[tuple[float, ...], bool]:
        adaptive = getattr(policy, "should_continue_warmup", None)
        converged = getattr(policy, "warmup_converged", None)
        adaptive_enabled = getattr(policy, "adaptive_warmup", True) and getattr(
            timer, "stationarity_reliable", True
        )
        if not adaptive_enabled or not callable(adaptive) or not callable(converged):
            for _ in range(policy.warmup_runs):
                self._guard(workload.run_once, "workload warmup run failed")
            return (), True

        samples: list[float] = []
        while adaptive(tuple(samples)):
            samples.append(self._timed_iteration(workload, timer, warmup=True))
        frozen = tuple(samples)
        return frozen, bool(converged(frozen))

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
            elapsed_ms = self._timed_iteration(workload, timer, warmup=False)
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

    def _timed_iteration(self, workload: BenchmarkWorkload, timer: Timer, *, warmup: bool) -> float:
        phase = "warmup" if warmup else "measured"
        self._guard(timer.start, f"timer start failed during {phase}")
        self._guard(workload.run_once, f"workload {phase} run failed")
        elapsed_ms = self._guard(timer.stop, f"timer stop failed during {phase}")
        if not math.isfinite(elapsed_ms) or elapsed_ms < 0.0:
            raise ProfilingError(
                ProfilingErrorCategory.BENCHMARK_FAILED,
                f"timer returned an invalid elapsed duration: {elapsed_ms!r}",
            )
        return elapsed_ms

    @staticmethod
    def _quality(samples_ms: list[float]) -> MeasurementQuality:
        summary = summarize_samples(samples_ms)
        coefficient = summary.stddev / max(abs(summary.mean), 1e-12)
        split = len(samples_ms) // 2
        if split == 0:
            drift = 0.0
        else:
            first = statistics.median(samples_ms[:split])
            second = statistics.median(samples_ms[split:])
            drift = abs(second - first) / max(abs(first), abs(second), 1e-12)
        stationary = drift <= 0.05
        return MeasurementQuality(
            stationary=stationary,
            drift_ratio=drift,
            coefficient_of_variation=coefficient,
            eligible_for_calibration=stationary,
        )

    @staticmethod
    def _raise_unstable_warmup(samples_ms: tuple[float, ...], policy: SamplingPolicy) -> None:
        summary = summarize_samples(samples_ms) if samples_ms else None
        raise ProfilingError(
            ProfilingErrorCategory.UNSTABLE_PERFORMANCE_STATE,
            "benchmark warmup did not converge before its run limit",
            {
                "actual_warmup_runs": len(samples_ms),
                "requested_min_warmup_runs": policy.warmup_runs,
                "warmup_converged": False,
                "warmup_median_ms": summary.median if summary else None,
                "warmup_samples_ms_json": json.dumps(samples_ms),
            },
        )

    @staticmethod
    def _raise_unstable_measurement(
        samples_ms: list[float],
        quality: MeasurementQuality,
        actual_warmup_runs: int,
        warmup_converged: bool,
    ) -> None:
        raise ProfilingError(
            ProfilingErrorCategory.UNSTABLE_PERFORMANCE_STATE,
            "measured latency samples changed performance state during the run",
            {
                "actual_warmup_runs": actual_warmup_runs,
                "warmup_converged": warmup_converged,
                "measured_runs": len(samples_ms),
                "stationary": quality.stationary,
                "drift_ratio": quality.drift_ratio,
                "coefficient_of_variation": quality.coefficient_of_variation,
                "samples_ms_json": json.dumps(samples_ms),
            },
        )

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
