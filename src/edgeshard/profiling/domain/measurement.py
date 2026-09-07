"""Empirical measurement records (Phase 2 spec §8.3, §14-15, §33-35).

A ``MeasurementRecord`` stores actual observations only — never estimates
or predictions (§52.1). A metric that could not be observed is absent
(``None``), not guessed (§52.2), and failures are never encoded as zero
latency, empty samples, or ``None`` measurement values (§42) — a failed
case produces a typed ``ProfilingFailure``, not a record.

Observations are grouped into typed metric objects instead of one giant
flat record with many nullable top-level fields (§8.3)::

    MeasurementRecord.metrics
        ├── latency            benchmark timing summary (CUDA events, §11)
        ├── allocator_memory   PyTorch allocator view (§14.1)
        ├── physical_memory    Phase 1 MemoryPool view (§14.2, §52.8)
        ├── telemetry          contextual device observations (§15)
        ├── rtt                network round-trip probes (§33)
        └── bandwidth          iperf3-style throughput (§34)

Telemetry is context, never a latency correction (§15); the contamination
flag records that a run was marked contaminated by a threshold check, and
``None`` means no check was applied.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from edgeshard.profiling.domain.environment import (
    EnvironmentFingerprint,
    environment_fingerprint_id,
)
from edgeshard.profiling.domain.hashing import (
    JsonScalar,
    check_normalized_items,
    normalized_items,
)


class TimeUnit(StrEnum):
    """Unit of a timing summary."""

    SECONDS = "s"
    MILLISECONDS = "ms"
    MICROSECONDS = "us"
    NANOSECONDS = "ns"


def _require_aware(value: datetime, field: str) -> None:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(f"{field} must be timezone-aware")


@dataclass(frozen=True)
class SampleSummary:
    """Summary statistics of one sample series (spec §8.3).

    All values are finite floats in the unit documented by the owning
    metric object; ``p95`` is ``None`` when too few samples were collected
    (§33: "p95 if enough samples").
    """

    mean: float
    median: float
    stddev: float
    minimum: float
    maximum: float
    p95: float | None

    def __post_init__(self) -> None:
        for field_name in ("mean", "median", "stddev", "minimum", "maximum", "p95"):
            value = getattr(self, field_name)
            if value is None:
                continue
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite, got {value}")
        if self.stddev < 0:
            raise ValueError(f"stddev must not be negative, got {self.stddev}")
        if self.minimum > self.maximum:
            raise ValueError(
                f"minimum ({self.minimum}) must not exceed maximum ({self.maximum})"
            )
        if self.p95 is not None and not self.minimum <= self.p95 <= self.maximum:
            raise ValueError(f"p95 ({self.p95}) outside [min, max]")


def summarize_samples(
    samples: Sequence[float], *, p95_min_samples: int = 5
) -> SampleSummary:
    """Compute summary statistics of a sample series (stdlib only).

    ``stddev`` is the sample standard deviation (n-1); a single sample
    yields ``0.0``. ``p95`` uses the nearest-rank method on the sorted
    series and stays ``None`` below ``p95_min_samples`` observations.
    Non-finite samples are rejected loudly — a NaN would silently poison
    every downstream statistic.
    """
    if not samples:
        raise ValueError("cannot summarize an empty sample series")
    if p95_min_samples < 1:
        raise ValueError(f"p95_min_samples must be positive, got {p95_min_samples}")
    for sample in samples:
        if not math.isfinite(sample):
            raise ValueError(f"samples must be finite, got {sample}")
    ordered = sorted(samples)
    count = len(ordered)
    p95: float | None = None
    if count >= p95_min_samples:
        rank = math.ceil(0.95 * count)  # nearest-rank, 1-based
        p95 = ordered[min(rank, count) - 1]
    return SampleSummary(
        mean=statistics.fmean(samples),
        median=statistics.median(ordered),
        stddev=statistics.stdev(samples) if count >= 2 else 0.0,
        minimum=ordered[0],
        maximum=ordered[-1],
        p95=p95,
    )


@dataclass(frozen=True)
class LatencyMetrics:
    """Benchmark timing observation (CUDA events on GPU, §11)."""

    summary: SampleSummary
    unit: TimeUnit


@dataclass(frozen=True)
class AllocatorMemoryMetrics:
    """PyTorch allocator view of one benchmark interval (spec §14.1).

    Byte counts captured around the measurement interval; peak counters are
    reset before it (the reset itself is harness behavior, §12).
    """

    allocated_before: int
    reserved_before: int
    peak_allocated: int
    peak_reserved: int
    allocated_after: int
    reserved_after: int

    def __post_init__(self) -> None:
        for field_name in (
            "allocated_before",
            "reserved_before",
            "peak_allocated",
            "peak_reserved",
            "allocated_after",
            "reserved_after",
        ):
            value: int = getattr(self, field_name)
            if value < 0:
                raise ValueError(f"{field_name} must not be negative, got {value}")


@dataclass(frozen=True)
class PhysicalMemoryMetrics:
    """Physical MemoryPool view of one benchmark interval (spec §14.2).

    ``pool_id`` references the Phase 1 ``MemoryPoolCapability`` — the RTX
    VRAM pool or the Jetson shared ``system-memory`` pool; Phase 2 never
    creates a second "GPU memory" model (§14.2, §52.8). A reading the
    backend cannot supply is ``None`` (§52.2).
    """

    pool_id: str
    used_before: int | None
    used_peak: int | None
    used_after: int | None

    def __post_init__(self) -> None:
        if not self.pool_id:
            raise ValueError("pool_id must not be empty")
        for field_name in ("used_before", "used_peak", "used_after"):
            value = getattr(self, field_name)
            if value is not None and value < 0:
                raise ValueError(f"{field_name} must not be negative, got {value}")


@dataclass(frozen=True)
class TelemetrySample:
    """One contextual device telemetry observation (spec §15).

    Same shape language as the Phase 1 device state (utilization is a
    0-100 percentage; anything the backend cannot supply is ``None``).
    """

    device_id: str
    utilization: float | None = None
    temperature_c: float | None = None
    power_w: float | None = None
    clock_mhz: float | None = None
    memory_used_bytes: int | None = None

    def __post_init__(self) -> None:
        if not self.device_id:
            raise ValueError("device_id must not be empty")
        if self.utilization is not None and not 0.0 <= self.utilization <= 100.0:
            raise ValueError(
                f"utilization must be within [0, 100], got {self.utilization}"
            )
        if self.power_w is not None and self.power_w < 0:
            raise ValueError(f"power_w must not be negative, got {self.power_w}")
        if self.clock_mhz is not None and self.clock_mhz <= 0:
            raise ValueError(f"clock_mhz must be positive, got {self.clock_mhz}")
        if self.memory_used_bytes is not None and self.memory_used_bytes < 0:
            raise ValueError(
                f"memory_used_bytes must not be negative, got {self.memory_used_bytes}"
            )


@dataclass(frozen=True)
class TelemetryContextMetrics:
    """Telemetry context around a benchmark (spec §15).

    Context only — telemetry never corrects latency in Phase 2.
    ``contaminated``: ``True``/``False`` records the outcome of a
    contamination check (e.g. initial utilization above threshold);
    ``None`` means no check was applied.
    """

    initial: TelemetrySample | None = None
    final: TelemetrySample | None = None
    contaminated: bool | None = None


@dataclass(frozen=True)
class RttMetrics:
    """Round-trip probe observation (spec §33).

    ``summary`` covers the per-packet RTT samples. Packet loss is the
    derived ratio of ``packets_sent - packets_received`` to
    ``packets_sent``; jitter is ``summary.stddev``.
    """

    summary: SampleSummary
    unit: TimeUnit
    packets_sent: int
    packets_received: int

    def __post_init__(self) -> None:
        if self.packets_sent <= 0:
            raise ValueError(f"packets_sent must be positive, got {self.packets_sent}")
        if self.packets_received < 0:
            raise ValueError(
                f"packets_received must not be negative, got {self.packets_received}"
            )
        if self.packets_received > self.packets_sent:
            raise ValueError(
                f"packets_received ({self.packets_received}) exceeds "
                f"packets_sent ({self.packets_sent})"
            )


@dataclass(frozen=True)
class BandwidthMetrics:
    """Single-flow baseline throughput observation (spec §34-35).

    ``bits_per_second`` follows iperf3 JSON conventions. This is an idle
    single-flow baseline, never a guaranteed concurrent bandwidth (§35,
    §52.7); the measurement regime is recorded on the case spec.
    """

    bits_per_second: float
    payload_bytes: int | None = None
    retransmits: int | None = None
    duration_s: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.bits_per_second) or self.bits_per_second < 0:
            raise ValueError(
                f"bits_per_second must be finite and non-negative, "
                f"got {self.bits_per_second}"
            )
        if self.payload_bytes is not None and self.payload_bytes <= 0:
            raise ValueError(f"payload_bytes must be positive, got {self.payload_bytes}")
        if self.retransmits is not None and self.retransmits < 0:
            raise ValueError(f"retransmits must not be negative, got {self.retransmits}")
        if self.duration_s is not None and self.duration_s <= 0:
            raise ValueError(f"duration_s must be positive, got {self.duration_s}")


@dataclass(frozen=True)
class MeasurementMetrics:
    """Typed metric bundle of one measurement (spec §8.3).

    At least one metric object must be present: a record without any
    observation is not a measurement.
    """

    latency: LatencyMetrics | None = None
    allocator_memory: AllocatorMemoryMetrics | None = None
    physical_memory: PhysicalMemoryMetrics | None = None
    telemetry: TelemetryContextMetrics | None = None
    rtt: RttMetrics | None = None
    bandwidth: BandwidthMetrics | None = None

    def __post_init__(self) -> None:
        if all(
            metric is None
            for metric in (
                self.latency,
                self.allocator_memory,
                self.physical_memory,
                self.telemetry,
                self.rtt,
                self.bandwidth,
            )
        ):
            raise ValueError("measurement metrics must contain at least one observation")


@dataclass(frozen=True)
class MeasurementRecord:
    """One persisted empirical observation set (spec §8.3).

    ``environment_fingerprint`` is the canonical id produced by
    ``environment_fingerprint_id`` (§9) — the reuse-judgment context.
    ``samples`` may be omitted (``None``) as a storage decision while
    ``sample_count`` still records how many samples the summary covers;
    when present, the series length must match. ``metadata`` holds
    normalized ``(key, value)`` JSON-scalar items (build them with
    ``normalized_items``).

    Records are append-oriented (§44): a changed environment yields a new
    record under a new fingerprint, never an update of history.
    """

    measurement_id: str
    case_id: str
    environment_fingerprint: str

    started_at: datetime
    finished_at: datetime

    sample_count: int
    samples: tuple[float, ...] | None

    metrics: MeasurementMetrics

    metadata: tuple[tuple[str, JsonScalar], ...] = ()
    environment: EnvironmentFingerprint | None = None

    def __post_init__(self) -> None:
        if not self.measurement_id:
            raise ValueError("measurement_id must not be empty")
        if not self.case_id:
            raise ValueError("case_id must not be empty")
        if not self.environment_fingerprint:
            raise ValueError("environment_fingerprint must not be empty")
        if (
            self.environment is not None
            and environment_fingerprint_id(self.environment)
            != self.environment_fingerprint
        ):
            raise ValueError(
                "environment_fingerprint does not match the embedded "
                "EnvironmentFingerprint"
            )
        _require_aware(self.started_at, "started_at")
        _require_aware(self.finished_at, "finished_at")
        if self.finished_at < self.started_at:
            raise ValueError("finished_at must not precede started_at")
        if self.sample_count < 0:
            raise ValueError(f"sample_count must not be negative, got {self.sample_count}")
        if self.samples is not None and len(self.samples) != self.sample_count:
            raise ValueError(
                f"samples length ({len(self.samples)}) does not match "
                f"sample_count ({self.sample_count})"
            )
        check_normalized_items(self.metadata, "metadata")

    @staticmethod
    def normalize_metadata(
        metadata: dict[str, JsonScalar] | None,
    ) -> tuple[tuple[str, JsonScalar], ...]:
        """Convenience wrapper building canonical metadata items."""
        if not metadata:
            return ()
        return normalized_items(metadata, "metadata")

    @property
    def metadata_mapping(self) -> dict[str, JsonScalar]:
        """Metadata as a read-friendly mapping view."""
        return dict(self.metadata)
