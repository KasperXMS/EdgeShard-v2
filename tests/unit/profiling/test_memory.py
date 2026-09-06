"""Memory instrumentation (spec §14, §52.2, §52.8)."""

from __future__ import annotations

import pytest
import torch

from edgeshard.profiling.domain.measurement import AllocatorMemoryMetrics
from edgeshard.profiling.instrumentation import memory as memory_module
from edgeshard.profiling.instrumentation.memory import (
    AllocatorReading,
    CudaAllocatorMemoryProbe,
    PhysicalMemoryProbe,
)

BEFORE = AllocatorReading(
    allocated_bytes=100, reserved_bytes=200, peak_allocated_bytes=999, peak_reserved_bytes=999
)
AFTER = AllocatorReading(
    allocated_bytes=150, reserved_bytes=200, peak_allocated_bytes=5_000, peak_reserved_bytes=8_000
)


def test_allocator_probe_reports_none_without_cuda() -> None:
    """§52.2: on a CPU-only host the allocator view is None, never zeros."""
    if torch.cuda.is_available():
        pytest.skip("CUDA available: None-path not observable")
    probe = CudaAllocatorMemoryProbe()
    probe.open()
    assert probe.close() is None


def test_allocator_probe_resets_peaks_between_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    readings = [BEFORE, AFTER]

    def fake_read(device_index: int | None = None) -> AllocatorReading | None:
        events.append(f"read:{device_index}")
        return readings.pop(0)

    def fake_reset(device_index: int | None = None) -> None:
        events.append(f"reset:{device_index}")

    monkeypatch.setattr(memory_module, "read_cuda_allocator", fake_read)
    monkeypatch.setattr(memory_module, "reset_cuda_peak_stats", fake_reset)

    probe = CudaAllocatorMemoryProbe(device_index=0)
    probe.open()
    metrics = probe.close()
    # §14.1: read "before", reset peaks, then measure; peaks come from the
    # post-reset read so they describe the interval only.
    assert events == ["read:0", "reset:0", "read:0"]
    assert metrics == AllocatorMemoryMetrics(
        allocated_before=100,
        reserved_before=200,
        peak_allocated=5_000,
        peak_reserved=8_000,
        allocated_after=150,
        reserved_after=200,
    )


def test_allocator_probe_close_without_open_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(memory_module, "read_cuda_allocator", lambda device_index=None: AFTER)
    probe = CudaAllocatorMemoryProbe()
    assert probe.close() is None


def test_read_cuda_allocator_on_cpu_host() -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA available: CPU None-path not observable")
    assert memory_module.read_cuda_allocator() is None


def _probe(
    total_bytes: int | None, readings: list[int | None]
) -> tuple[PhysicalMemoryProbe, list[int | None]]:
    queue = list(readings)

    def read_available() -> int | None:
        return queue.pop(0) if queue else None

    return (
        PhysicalMemoryProbe(
            "vram:GPU-uuid-1", total_bytes=total_bytes, read_available_bytes=read_available
        ),
        queue,
    )


def test_physical_probe_computes_used_and_tracks_peak() -> None:
    probe, _ = _probe(1_000, [400, 300, 500])
    probe.open()  # used_before = 600
    probe.poll()  # used 700 -> new peak
    metrics = probe.close()  # used_after = 500
    assert metrics.pool_id == "vram:GPU-uuid-1"
    assert metrics.used_before == 600
    assert metrics.used_peak == 700
    assert metrics.used_after == 500


def test_physical_probe_missing_total_yields_none_readings() -> None:
    """§52.2: without a total, usage is not computable — never guessed."""
    probe, _ = _probe(None, [400, 300])
    probe.open()
    probe.poll()
    metrics = probe.close()
    assert (metrics.used_before, metrics.used_peak, metrics.used_after) == (None, None, None)


def test_physical_probe_missing_and_inconsistent_readings() -> None:
    # available > total and available < 0 are inconsistent -> missing.
    probe, _ = _probe(1_000, [400, 5_000, -1, 250])
    probe.open()  # used_before = 600, peak = 600
    probe.poll()  # inconsistent -> ignored
    probe.poll()  # negative -> ignored
    metrics = probe.close()  # used_after = 750, peak = 750
    assert metrics.used_before == 600
    assert metrics.used_peak == 750
    assert metrics.used_after == 750


def test_physical_probe_reader_exceptions_propagate() -> None:
    """A crashing backend is a failure (§42), not a silent missing metric."""

    def explode() -> int | None:
        raise OSError("telemetry backend unavailable")

    probe = PhysicalMemoryProbe("system-memory", total_bytes=1_000, read_available_bytes=explode)
    with pytest.raises(OSError):
        probe.open()


def test_physical_probe_validation() -> None:
    with pytest.raises(ValueError, match="pool_id"):
        PhysicalMemoryProbe("", total_bytes=1, read_available_bytes=lambda: None)
    with pytest.raises(ValueError, match="total_bytes"):
        PhysicalMemoryProbe("p", total_bytes=0, read_available_bytes=lambda: None)
