"""Sampling policies (Phase 2 spec §13).

The policy decides *how many* warmup and measured runs a benchmark
performs. It is a replaceable interface: the v1 default is duration-based
with minimum/maximum sample counts; convergence-based sampling,
confidence-interval stopping, and profiling-budget algorithms are future
policies behind the same protocol (§13 — explicitly not v1 work).
"""

from __future__ import annotations

import math
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
    """Default v1 policy (§12): duration target within run-count bounds.

    Defaults: minimum 3 warmups, 5-20 measured runs, ~1 second of
    accumulated measurement duration. Between the bounds the policy
    stops as soon as the accumulated duration of *measured* runs reaches
    the target — warmups never count toward it.
    """

    def __init__(
        self,
        *,
        min_warmups: int = 3,
        min_runs: int = 5,
        max_runs: int = 20,
        target_duration_ms: float = 1000.0,
    ) -> None:
        if min_warmups < 0:
            raise ValueError(f"min_warmups must not be negative, got {min_warmups}")
        if min_runs < 1:
            raise ValueError(f"min_runs must be at least 1, got {min_runs}")
        if max_runs < min_runs:
            raise ValueError(
                f"max_runs ({max_runs}) must not be below min_runs ({min_runs})"
            )
        if not math.isfinite(target_duration_ms) or target_duration_ms <= 0.0:
            raise ValueError(
                f"target_duration_ms must be positive and finite, got {target_duration_ms}"
            )
        self._min_warmups = min_warmups
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

    def should_continue(self, state: SamplingState) -> bool:
        if state.measured_runs >= self._max_runs:
            return False
        if state.measured_runs < self._min_runs:
            return True
        return state.accumulated_measurement_ms < self._target_duration_ms
