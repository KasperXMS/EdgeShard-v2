"""Measurement records, typed metrics, and summary statistics (spec §8.3, §14-15, §52)."""

from __future__ import annotations

import statistics
from datetime import UTC, datetime, timedelta

import pytest

from edgeshard.profiling.domain.measurement import (
    AllocatorMemoryMetrics,
    BandwidthMetrics,
    LatencyMetrics,
    MeasurementMetrics,
    MeasurementRecord,
    PhysicalMemoryMetrics,
    RttMetrics,
    SampleSummary,
    TelemetryContextMetrics,
    TelemetrySample,
    TimeUnit,
    summarize_samples,
)

START = datetime(2026, 9, 7, 12, 0, tzinfo=UTC)
FINISH = START + timedelta(seconds=2)


def test_summarize_samples_known_series() -> None:
    samples = [1.0, 2.0, 3.0, 4.0, 100.0]
    summary = summarize_samples(samples)
    assert summary.mean == pytest.approx(statistics.fmean(samples))
    assert summary.median == 3.0
    assert summary.stddev == pytest.approx(statistics.stdev(samples))
    assert summary.minimum == 1.0
    assert summary.maximum == 100.0
    # 5 samples ≥ p95_min_samples=5 → nearest-rank ceil(0.95*5)=5th value.
    assert summary.p95 == 100.0


def test_summarize_samples_p95_needs_enough_samples() -> None:
    assert summarize_samples([1.0, 2.0, 3.0], p95_min_samples=5).p95 is None
    twenty = [float(i) for i in range(1, 21)]
    # nearest-rank ceil(0.95*20)=19th sorted value
    assert summarize_samples(twenty).p95 == 19.0


def test_summarize_samples_single_sample_has_zero_stddev() -> None:
    summary = summarize_samples([42.0])
    assert summary.stddev == 0.0
    assert summary.mean == 42.0


def test_summarize_samples_rejects_empty_and_non_finite() -> None:
    with pytest.raises(ValueError, match="empty"):
        summarize_samples([])
    with pytest.raises(ValueError, match="finite"):
        summarize_samples([1.0, float("nan")])


def test_sample_summary_validation() -> None:
    with pytest.raises(ValueError, match="finite"):
        SampleSummary(
            mean=float("nan"), median=0.0, stddev=0.0, minimum=0.0, maximum=0.0, p95=None
        )
    with pytest.raises(ValueError, match="must not exceed"):
        SampleSummary(mean=1.0, median=1.0, stddev=0.0, minimum=5.0, maximum=1.0, p95=None)
    with pytest.raises(ValueError, match="p95"):
        SampleSummary(mean=1.0, median=1.0, stddev=0.0, minimum=0.0, maximum=2.0, p95=9.0)
    with pytest.raises(ValueError, match="stddev"):
        SampleSummary(mean=1.0, median=1.0, stddev=-1.0, minimum=0.0, maximum=2.0, p95=None)


def test_metrics_bundle_requires_at_least_one_observation() -> None:
    with pytest.raises(ValueError, match="at least one"):
        MeasurementMetrics()
    assert MeasurementMetrics(
        latency=LatencyMetrics(
            summary=SampleSummary(1.0, 1.0, 0.0, 1.0, 1.0, None), unit=TimeUnit.MILLISECONDS
        )
    )


def test_allocator_metrics_reject_negative_bytes() -> None:
    with pytest.raises(ValueError, match="peak_allocated"):
        AllocatorMemoryMetrics(
            allocated_before=0,
            reserved_before=0,
            peak_allocated=-1,
            peak_reserved=0,
            allocated_after=0,
            reserved_after=0,
        )


def test_physical_memory_metrics_allow_missing_readings() -> None:
    """§52.2: a metric that cannot be measured is None, never guessed."""
    metrics = PhysicalMemoryMetrics(
        pool_id="pool-0", used_before=None, used_peak=None, used_after=None
    )
    assert metrics.used_peak is None
    with pytest.raises(ValueError, match="used_before"):
        PhysicalMemoryMetrics(pool_id="pool-0", used_before=-1, used_peak=None, used_after=None)


def test_telemetry_sample_ranges() -> None:
    with pytest.raises(ValueError, match="utilization"):
        TelemetrySample(device_id="gpu-0", utilization=101.0)
    with pytest.raises(ValueError, match="power_w"):
        TelemetrySample(device_id="gpu-0", power_w=-5.0)
    with pytest.raises(ValueError, match="device_id"):
        TelemetrySample(device_id="")


def test_telemetry_context_contamination_is_tri_state() -> None:
    """§15: None means no contamination check was applied."""
    assert TelemetryContextMetrics().contaminated is None


def test_rtt_metrics_packet_accounting() -> None:
    summary = SampleSummary(0.5, 0.5, 0.1, 0.3, 0.8, None)
    with pytest.raises(ValueError, match="exceeds"):
        RttMetrics(
            summary=summary, unit=TimeUnit.MILLISECONDS, packets_sent=5, packets_received=6
        )
    with pytest.raises(ValueError, match="packets_sent"):
        RttMetrics(summary=summary, unit=TimeUnit.MILLISECONDS, packets_sent=0, packets_received=0)


def test_bandwidth_metrics_validation() -> None:
    with pytest.raises(ValueError, match="bits_per_second"):
        BandwidthMetrics(bits_per_second=-1.0)
    with pytest.raises(ValueError, match="duration_s"):
        BandwidthMetrics(bits_per_second=1e9, duration_s=0.0)


def _record(**overrides: object) -> MeasurementRecord:
    base: dict[str, object] = {
        "measurement_id": "m-1",
        "case_id": "c-1",
        "environment_fingerprint": "f" * 64,
        "started_at": START,
        "finished_at": FINISH,
        "sample_count": 3,
        "samples": (1.0, 2.0, 3.0),
        "metrics": MeasurementMetrics(
            latency=LatencyMetrics(
                summary=SampleSummary(2.0, 2.0, 1.0, 1.0, 3.0, None),
                unit=TimeUnit.MILLISECONDS,
            )
        ),
    }
    base.update(overrides)
    return MeasurementRecord(**base)  # type: ignore[arg-type]


def test_record_requires_consistent_samples_and_count() -> None:
    with pytest.raises(ValueError, match="sample_count"):
        _record(sample_count=5)


def test_record_allows_summary_without_raw_samples() -> None:
    """§45: retaining summaries without the series is a storage decision."""
    record = _record(samples=None)
    assert record.sample_count == 3


def test_record_rejects_naive_or_inverted_timestamps() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        _record(started_at=datetime(2026, 9, 7, 12, 0))
    with pytest.raises(ValueError, match="precede"):
        _record(finished_at=START - timedelta(seconds=1))


def test_record_metadata_is_normalized_and_readable() -> None:
    record = _record(metadata=MeasurementRecord.normalize_metadata({"b": 1, "a": "x"}))
    assert record.metadata == (("a", "x"), ("b", 1))
    assert record.metadata_mapping == {"a": "x", "b": 1}
    with pytest.raises(ValueError, match="sorted"):
        _record(metadata=(("b", 1), ("a", 2)))


def test_record_rejects_empty_identifiers() -> None:
    with pytest.raises(ValueError, match="measurement_id"):
        _record(measurement_id="")
    with pytest.raises(ValueError, match="environment_fingerprint"):
        _record(environment_fingerprint="")
