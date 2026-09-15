"""Sampling policies (Phase 2 spec §13).

The policy decides *how many* warmup and measured runs a benchmark
performs. It is a replaceable interface: the v1 default is duration-based
with minimum/maximum sample counts; convergence-based sampling,
confidence-interval stopping, and profiling-budget algorithms are future
policies behind the same protocol (§13 — explicitly not v1 work).
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class SamplingState:
    """Progress of one benchmark run, as seen by the policy."""

    warmup_runs: int = 0
    measured_runs: int = 0
    accumulated_measurement_ms: float = 0.0


@runtime_checkable
class SamplingPolicy(Protocol):
    """Decides warmup count and whether to keep measuring."""

    @property
    def warmup_runs(self) -> int:
        """Warmup iterations executed before the measurement loop (§12)."""
        ...

    def should_continue(self, state: SamplingState) -> bool:
        """Whether another measured run should start."""
        ...


class DurationSamplingPolicy:
    """Duration-bounded measurement with adaptive warmup convergence.

    The production default waits for two stable rolling-window comparisons
    (3% median drift) between 5 and 50 timed warmups, then retains exactly
    20 measured samples. ``min_warmups`` remains as a compatibility alias
    for fixed-count test and extension policies.
    """

    def __init__(
        self,
        *,
        min_warmup_runs: int = 5,
        max_warmup_runs: int | None = None,
        window_size: int = 5,
        relative_tolerance: float = 0.03,
        stable_windows_required: int = 2,
        enforce_stationarity: bool | None = None,
        min_warmups: int | None = None,
        min_runs: int = 20,
        max_runs: int = 20,
        target_duration_ms: float = 1000.0,
    ) -> None:
        legacy_fixed = min_warmups is not None and max_warmup_runs is None
        if min_warmups is not None:
            min_warmup_runs = min_warmups
        if max_warmup_runs is None:
            max_warmup_runs = min_warmup_runs if legacy_fixed else 50
        if min_warmup_runs < 0:
            raise ValueError(
                f"min_warmups/min_warmup_runs must not be negative, got {min_warmup_runs}"
            )
        if max_warmup_runs < min_warmup_runs:
            raise ValueError(
                f"max_warmup_runs ({max_warmup_runs}) must not be below "
                f"min_warmup_runs ({min_warmup_runs})"
            )
        if window_size < 1:
            raise ValueError(f"window_size must be positive, got {window_size}")
        if not math.isfinite(relative_tolerance) or relative_tolerance < 0:
            raise ValueError(
                f"relative_tolerance must be finite and non-negative, got {relative_tolerance}"
            )
        if stable_windows_required < 1:
            raise ValueError(
                f"stable_windows_required must be positive, got {stable_windows_required}"
            )
        if min_runs < 1:
            raise ValueError(f"min_runs must be at least 1, got {min_runs}")
        if max_runs < min_runs:
            raise ValueError(f"max_runs ({max_runs}) must not be below min_runs ({min_runs})")
        if not math.isfinite(target_duration_ms) or target_duration_ms <= 0.0:
            raise ValueError(
                f"target_duration_ms must be positive and finite, got {target_duration_ms}"
            )
        self._min_warmups = min_warmup_runs
        self._max_warmups = max_warmup_runs
        self._window_size = window_size
        self._relative_tolerance = relative_tolerance
        self._stable_windows_required = stable_windows_required
        self._legacy_fixed_warmup = legacy_fixed
        self._enforce_stationarity = (
            not legacy_fixed if enforce_stationarity is None else enforce_stationarity
        )
        self._min_runs = min_runs
        self._max_runs = max_runs
        self._target_duration_ms = target_duration_ms

    @property
    def min_runs(self) -> int:
        return self._min_runs

    @property
    def max_runs(self) -> int:
        return self._max_runs

    @property
    def target_duration_ms(self) -> float:
        return self._target_duration_ms

    @property
    def warmup_runs(self) -> int:
        return self._min_warmups

    @property
    def min_warmup_runs(self) -> int:
        return self._min_warmups

    @property
    def max_warmup_runs(self) -> int:
        return self._max_warmups

    @property
    def adaptive_warmup(self) -> bool:
        return not self._legacy_fixed_warmup

    @property
    def enforce_stationarity(self) -> bool:
        return self._enforce_stationarity

    def warmup_converged(self, samples_ms: tuple[float, ...]) -> bool:
        """Whether the latest rolling-window comparisons are stable."""
        count = len(samples_ms)
        if self._legacy_fixed_warmup:
            return count >= self._min_warmups
        required = 2 * self._window_size + self._stable_windows_required - 1
        if count < max(self._min_warmups, required):
            return False
        epsilon = 1e-12
        for offset in range(self._stable_windows_required):
            end = count - offset
            current = samples_ms[end - self._window_size : end]
            previous = samples_ms[end - (2 * self._window_size) : end - self._window_size]
            current_median = statistics.median(current)
            previous_median = statistics.median(previous)
            relative_change = abs(current_median - previous_median) / max(
                abs(current_median), epsilon
            )
            if relative_change > self._relative_tolerance:
                return False
        return True

    def should_continue_warmup(self, samples_ms: tuple[float, ...]) -> bool:
        if len(samples_ms) < self._min_warmups:
            return True
        if self.warmup_converged(samples_ms):
            return False
        return len(samples_ms) < self._max_warmups

    def should_continue(self, state: SamplingState) -> bool:
        if state.measured_runs >= self._max_runs:
            return False
        if state.measured_runs < self._min_runs:
            return True
        return state.accumulated_measurement_ms < self._target_duration_ms
