"""Telemetry context and contamination checks (spec §15)."""

from __future__ import annotations

import pytest

from edgeshard.profiling.domain.experiment import ProfilingErrorCategory
from edgeshard.profiling.domain.measurement import TelemetrySample
from edgeshard.profiling.errors import ProfilingError
from edgeshard.profiling.instrumentation.telemetry import (
    ContaminationAction,
    DeviceObservation,
    TelemetryContextCollector,
    TelemetryInstrumentation,
)


class ScriptedTelemetry:
    """Returns scripted observations; ``None`` when the script runs out."""

    def __init__(self, observations: list[DeviceObservation | None]) -> None:
        self._observations = list(observations)
        self.calls = 0

    def capture(self) -> DeviceObservation | None:
        self.calls += 1
        return self._observations.pop(0) if self._observations else None


IDLE = DeviceObservation(
    device_id="GPU-uuid-1", utilization=2.0, temperature_c=41.0, power_w=28.5
)
BUSY = DeviceObservation(device_id="GPU-uuid-1", utilization=93.0, temperature_c=71.0)


def test_observation_conforms_to_protocol() -> None:
    assert isinstance(ScriptedTelemetry([]), TelemetryInstrumentation)


def test_observation_to_sample_maps_fields() -> None:
    sample = IDLE.to_sample()
    assert sample == TelemetrySample(
        device_id="GPU-uuid-1", utilization=2.0, temperature_c=41.0, power_w=28.5
    )


def test_observation_to_sample_inherits_domain_validation() -> None:
    with pytest.raises(ValueError, match="utilization"):
        DeviceObservation(device_id="GPU-uuid-1", utilization=150.0).to_sample()


def test_collector_without_instrumentation_has_no_context() -> None:
    collector = TelemetryContextCollector()
    collector.capture_initial()
    assert collector.contaminated is None
    assert collector.capture_final() is None


def test_collector_without_threshold_never_checks() -> None:
    collector = TelemetryContextCollector(ScriptedTelemetry([BUSY, IDLE]))
    collector.capture_initial()
    metrics = collector.capture_final()
    assert collector.contaminated is None
    assert metrics is not None and metrics.contaminated is None
    assert metrics.initial is not None and metrics.initial.utilization == 93.0


def test_mark_mode_records_contamination_and_continues() -> None:
    collector = TelemetryContextCollector(
        ScriptedTelemetry([BUSY, IDLE]), contamination_threshold_percent=50.0
    )
    collector.capture_initial()  # no raise in MARK mode
    assert collector.contaminated is True
    metrics = collector.capture_final()
    assert metrics is not None
    assert metrics.contaminated is True
    assert metrics.final is not None and metrics.final.utilization == 2.0


def test_clean_device_is_marked_not_contaminated() -> None:
    collector = TelemetryContextCollector(
        ScriptedTelemetry([IDLE, IDLE]), contamination_threshold_percent=50.0
    )
    collector.capture_initial()
    assert collector.contaminated is False
    metrics = collector.capture_final()
    assert metrics is not None and metrics.contaminated is False


def test_reject_mode_raises_device_busy_before_measuring() -> None:
    backend = ScriptedTelemetry([BUSY, IDLE])
    collector = TelemetryContextCollector(
        backend,
        contamination_threshold_percent=50.0,
        contamination_action=ContaminationAction.REJECT,
    )
    with pytest.raises(ProfilingError) as excinfo:
        collector.capture_initial()
    assert excinfo.value.category is ProfilingErrorCategory.DEVICE_BUSY
    assert excinfo.value.details == (
        ("device_id", "GPU-uuid-1"),
        ("threshold", 50.0),
        ("utilization", 93.0),
    )
    # The final capture never happened: the benchmark was rejected.
    assert backend.calls == 1


def test_threshold_requires_utilization_to_apply() -> None:
    """No utilization in the initial sample -> no check applied (None)."""
    no_utilization = DeviceObservation(device_id="GPU-uuid-1", temperature_c=55.0)
    collector = TelemetryContextCollector(
        ScriptedTelemetry([no_utilization]), contamination_threshold_percent=50.0
    )
    collector.capture_initial()
    assert collector.contaminated is None


def test_missing_initial_observation_skips_check() -> None:
    collector = TelemetryContextCollector(
        ScriptedTelemetry([None, IDLE]), contamination_threshold_percent=50.0
    )
    collector.capture_initial()
    assert collector.contaminated is None
    metrics = collector.capture_final()
    assert metrics is not None and metrics.initial is None and metrics.final is not None


def test_threshold_validation() -> None:
    with pytest.raises(ValueError, match="contamination_threshold_percent"):
        TelemetryContextCollector(contamination_threshold_percent=150.0)
