"""Typed profiling errors (spec §42)."""

from __future__ import annotations

import pytest

from edgeshard.model.errors import EdgeShardError
from edgeshard.profiling.domain.experiment import ProfilingErrorCategory
from edgeshard.profiling.errors import ProfilingError


def test_error_is_edgeshard_error_with_category() -> None:
    error = ProfilingError(ProfilingErrorCategory.BENCHMARK_FAILED, "run exploded")
    assert isinstance(error, EdgeShardError)
    assert str(error) == "run exploded"
    assert error.category is ProfilingErrorCategory.BENCHMARK_FAILED
    assert error.details == ()


def test_error_details_are_normalized() -> None:
    error = ProfilingError(
        ProfilingErrorCategory.DEVICE_BUSY,
        "gpu busy",
        {"utilization": 97.5, "device_id": "GPU-uuid-1"},
    )
    assert error.details == (("device_id", "GPU-uuid-1"), ("utilization", 97.5))


def test_error_requires_message() -> None:
    with pytest.raises(ValueError, match="message"):
        ProfilingError(ProfilingErrorCategory.INTERNAL_ERROR, "")


def test_to_failure_round_trips_into_domain_record() -> None:
    error = ProfilingError(
        ProfilingErrorCategory.TIMEOUT, "case timed out", {"timeout_s": 30}
    )
    failure = error.to_failure()
    assert failure.category is ProfilingErrorCategory.TIMEOUT
    assert failure.message == "case timed out"
    assert failure.details == (("timeout_s", 30),)
