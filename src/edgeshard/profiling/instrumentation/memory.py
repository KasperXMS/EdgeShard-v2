"""Memory instrumentation (Phase 2 spec §14).

Two independent views of the same measurement interval — they answer
different questions and are never merged or double counted (§52.8):

- :class:`CudaAllocatorMemoryProbe` — the PyTorch caching allocator's
  view (§14.1): allocated/reserved bytes before and after, and peak
  counters that are *reset before the interval* so peaks describe the
  measured work, not warmup or history. CUDA-only; ``None`` elsewhere
  (§52.2).
- :class:`PhysicalMemoryProbe` — the physical Phase 1 ``MemoryPool``
  view (§14.2): on RTX the VRAM pool, on Jetson the shared
  ``system-memory`` pool. Reuses the pool identity by value; Phase 2
  creates no second "GPU memory" domain model. Physical pools have no
  hardware peak counter, so ``used_peak`` is the maximum over the
  probe's own samples (``open``/``poll``/``close``).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

from edgeshard.profiling.domain.measurement import (
    AllocatorMemoryMetrics,
    PhysicalMemoryMetrics,
)

logger = logging.getLogger("profiling.instrumentation.memory")

AvailableBytesReader = Callable[[], int | None]
"""Supplies a pool's currently available bytes; ``None`` when the backend
cannot answer (missing metric, §52.2 — never a guessed value)."""


@dataclass(frozen=True)
class AllocatorReading:
    """One PyTorch caching-allocator observation (spec §14.1)."""

    allocated_bytes: int
    reserved_bytes: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int


def read_cuda_allocator(device_index: int | None = None) -> AllocatorReading | None:
    """Current allocator counters, or ``None`` without CUDA (§52.2)."""
    if not torch.cuda.is_available():
        return None
    device = torch.device("cuda", device_index) if device_index is not None else "cuda"
    return AllocatorReading(
        allocated_bytes=int(torch.cuda.memory_allocated(device)),
        reserved_bytes=int(torch.cuda.memory_reserved(device)),
        peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
    )


def reset_cuda_peak_stats(device_index: int | None = None) -> None:
    """Reset allocator peak counters (spec §14.1: reset *before* measuring)."""
    if torch.cuda.is_available():
        device = (
            torch.device("cuda", device_index) if device_index is not None else "cuda"
        )
        torch.cuda.reset_peak_memory_stats(device)


@runtime_checkable
class MemoryInstrumentation(Protocol):
    """What the harness needs around its measurement interval (§12).

    ``open`` marks the interval start (and resets peak counters);
    ``close`` returns the interval's metrics, or ``None`` when this host
    cannot measure them (§52.2).
    """

    def open(self) -> None: ...

    def close(self) -> AllocatorMemoryMetrics | None: ...


class CudaAllocatorMemoryProbe:
    """PyTorch allocator view of one benchmark interval (spec §14.1)."""

    def __init__(self, device_index: int | None = None) -> None:
        self._device_index = device_index
        self._before: AllocatorReading | None = None

    def open(self) -> None:
        self._before = read_cuda_allocator(self._device_index)
        if self._before is not None:
            # Peaks must describe the measured interval only.
            reset_cuda_peak_stats(self._device_index)

    def close(self) -> AllocatorMemoryMetrics | None:
        before = self._before
        self._before = None
        after = read_cuda_allocator(self._device_index)
        if before is None or after is None:
            return None
        return AllocatorMemoryMetrics(
            allocated_before=before.allocated_bytes,
            reserved_before=before.reserved_bytes,
            peak_allocated=after.peak_allocated_bytes,
            peak_reserved=after.peak_reserved_bytes,
            allocated_after=after.allocated_bytes,
            reserved_after=after.reserved_bytes,
        )


class PhysicalMemoryProbe:
    """Physical MemoryPool usage around one benchmark interval (§14.2).

    ``used = total_bytes - available_bytes``. Either side missing yields
    ``None`` for that reading (§52.2); an available-bytes reading outside
    ``[0, total_bytes]`` is inconsistent and also treated as missing
    (logged, never clamped or guessed). ``poll`` may be called any number
    of times during the interval to refine ``used_peak``.
    """

    def __init__(
        self,
        pool_id: str,
        *,
        total_bytes: int | None,
        read_available_bytes: AvailableBytesReader,
    ) -> None:
        if not pool_id:
            raise ValueError("pool_id must not be empty")
        if total_bytes is not None and total_bytes <= 0:
            raise ValueError(f"total_bytes must be positive, got {total_bytes}")
        self._pool_id = pool_id
        self._total_bytes = total_bytes
        self._read_available_bytes = read_available_bytes
        self._used_before: int | None = None
        self._used_peak: int | None = None

    def open(self) -> None:
        self._used_before = self._read_used()
        self._used_peak = self._used_before

    def poll(self) -> None:
        self._raise_peak(self._read_used())

    def close(self) -> PhysicalMemoryMetrics:
        used_after = self._read_used()
        self._raise_peak(used_after)
        return PhysicalMemoryMetrics(
            pool_id=self._pool_id,
            used_before=self._used_before,
            used_peak=self._used_peak,
            used_after=used_after,
        )

    def _raise_peak(self, used: int | None) -> None:
        if used is not None and (self._used_peak is None or used > self._used_peak):
            self._used_peak = used

    def _read_used(self) -> int | None:
        if self._total_bytes is None:
            return None
        available = self._read_available_bytes()
        if available is None:
            return None
        if available < 0 or available > self._total_bytes:
            logger.warning(
                "pool=%s inconsistent available_bytes=%d (total=%d); treating as missing",
                self._pool_id,
                available,
                self._total_bytes,
            )
            return None
        return self._total_bytes - available
