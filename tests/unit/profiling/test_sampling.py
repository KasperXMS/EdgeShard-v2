"""Sampling policies (spec §12-§13)."""

from __future__ import annotations

import pytest

from edgeshard.profiling.benchmark.sampling import (
    DurationSamplingPolicy,
    SamplingPolicy,
    SamplingState,
)


def test_default_policy_matches_spec_v1() -> None:
    """§12: >=3 warmups, 5-20 measured runs, ~1 s target duration."""
    policy = DurationSamplingPolicy()
    assert policy.warmup_runs == 3
    assert policy.min_runs == 5
    assert policy.max_runs == 20
    assert policy.target_duration_ms == 1000.0
    assert isinstance(policy, SamplingPolicy)


def test_minimum_runs_take_precedence_over_duration() -> None:
    policy = DurationSamplingPolicy(min_runs=5, max_runs=20, target_duration_ms=1000.0)
    # Even after 5 s accumulated, fewer than min_runs means keep going.
    state = SamplingState(measured_runs=4, accumulated_measurement_ms=5_000.0)
    assert policy.should_continue(state) is True


def test_duration_target_stops_between_bounds() -> None:
    policy = DurationSamplingPolicy(min_runs=5, max_runs=20, target_duration_ms=1000.0)
    below = SamplingState(measured_runs=5, accumulated_measurement_ms=999.9)
    reached = SamplingState(measured_runs=5, accumulated_measurement_ms=1000.0)
    assert policy.should_continue(below) is True
    assert policy.should_continue(reached) is False


def test_maximum_runs_are_a_hard_cap() -> None:
    policy = DurationSamplingPolicy(min_runs=5, max_runs=20, target_duration_ms=1000.0)
    state = SamplingState(measured_runs=20, accumulated_measurement_ms=10.0)
    assert policy.should_continue(state) is False


def test_warmups_do_not_count_toward_duration() -> None:
    policy = DurationSamplingPolicy(min_warmups=3, min_runs=5, target_duration_ms=1000.0)
    state = SamplingState(warmup_runs=3, measured_runs=5, accumulated_measurement_ms=0.0)
    assert policy.should_continue(state) is True


def test_policy_validation() -> None:
    with pytest.raises(ValueError, match="min_warmups"):
        DurationSamplingPolicy(min_warmups=-1)
    with pytest.raises(ValueError, match="min_runs"):
        DurationSamplingPolicy(min_runs=0)
    with pytest.raises(ValueError, match="max_runs"):
        DurationSamplingPolicy(min_runs=10, max_runs=5)
    with pytest.raises(ValueError, match="target_duration_ms"):
        DurationSamplingPolicy(target_duration_ms=0.0)
    with pytest.raises(ValueError, match="target_duration_ms"):
        DurationSamplingPolicy(target_duration_ms=float("inf"))


def test_custom_policy_satisfies_protocol() -> None:
    """§13: the policy is a replaceable extension point."""

    class FixedCountPolicy:
        def __init__(self, runs: int) -> None:
            self._runs = runs

        @property
        def warmup_runs(self) -> int:
            return 0

        def should_continue(self, state: SamplingState) -> bool:
            return state.measured_runs < self._runs

    assert isinstance(FixedCountPolicy(2), SamplingPolicy)
