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


class TelemetryProbe(Protocol):
    """Dynamic telemetry backend (spec §25): sampled per heartbeat."""

    async def sample(self) -> StateFragment: ...
