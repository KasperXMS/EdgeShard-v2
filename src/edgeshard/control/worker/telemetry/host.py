"""Host telemetry via psutil (Phase 1 spec §21, §25).

Samples host-memory availability and CPU utilization. Missing metrics stay
``None`` - never "unavailable" (spec §17).

``psutil.cpu_percent(interval=None)`` is non-blocking and compares against
the previous call, which suits steady heartbeat loops; one-shot callers
(``worker inspect``) pass a short ``cpu_sample_interval_s`` so the single
sample is meaningful.
"""

from __future__ import annotations

import asyncio
import logging

import psutil

from edgeshard.cluster.state import DeviceAvailability, DeviceState, MemoryPoolState
from edgeshard.control.worker.discovery.host import HOST_MEMORY_POOL_ID
from edgeshard.control.worker.identity import derive_cpu_device_id
from edgeshard.control.worker.telemetry.base import StateFragment

logger = logging.getLogger("worker.telemetry.host")


class HostTelemetryProbe:
    """psutil-based host telemetry (spec §21)."""

    def __init__(self, worker_id: str, cpu_sample_interval_s: float | None = None) -> None:
        self._device_id = derive_cpu_device_id(worker_id)
        self._cpu_sample_interval_s = cpu_sample_interval_s

    async def sample(self) -> StateFragment:
        memory = psutil.virtual_memory()
        utilization = await asyncio.to_thread(
            psutil.cpu_percent, interval=self._cpu_sample_interval_s
        )
        # Clamp defensively: psutil already reports 0-100 system-wide, and
        # the domain rejects anything outside.
        utilization = max(0.0, min(100.0, float(utilization)))
        return StateFragment(
            device_states=(
                DeviceState(
                    device_id=self._device_id,
                    utilization=utilization,
                    # Generic hosts expose no CPU temperature/power sensors
                    # through psutil; hardware probes (P1C/P1D) report them.
                    temperature_c=None,
                    power_w=None,
                    availability=DeviceAvailability.AVAILABLE,
                    running_runtime_ids=(),
                ),
            ),
            memory_states=(
                MemoryPoolState(
                    memory_pool_id=HOST_MEMORY_POOL_ID,
                    available_bytes=int(memory.available),
                ),
            ),
        )
