"""Dynamic telemetry probe interface (Phase 1 spec §25).

Separate from ``CapabilityProbe`` because telemetry sampling runs on every
heartbeat while capability discovery runs once per Agent start.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from edgeshard.cluster.state import DeviceState, MemoryPoolState


@dataclass(frozen=True)
class StateFragment:
    """Dynamic facts one telemetry probe contributes to the Worker state."""

    device_states: tuple[DeviceState, ...] = ()
    memory_states: tuple[MemoryPoolState, ...] = ()


@dataclass(frozen=True)
class FreshDeviceTelemetry:
    """Profiling-only detail from an existing physical telemetry backend.

    These fields do not alter the Phase 1 wire contract.  They let Phase 2
    retain accelerator and EMC clocks available from the physical backend but
    intentionally not part of ``DeviceState``.
    """

    device_id: str
    utilization: float | None = None
    temperature_c: float | None = None
    power_w: float | None = None
    clock_mhz: float | None = None
    emc_clock_mhz: float | None = None


class TelemetryProbe(Protocol):
    """Dynamic telemetry backend (spec §25): sampled per heartbeat."""

    async def sample(self) -> StateFragment: ...


class FreshTelemetryProbe(Protocol):
    """Synchronous lightweight sample from an already-created backend."""

    def sample_fresh(self) -> StateFragment: ...
