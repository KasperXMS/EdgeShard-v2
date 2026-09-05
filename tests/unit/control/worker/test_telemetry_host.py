"""Host telemetry probe tests (Phase 1 spec §21, §25)."""

from __future__ import annotations

import psutil

from edgeshard.cluster.state import DeviceAvailability
from edgeshard.control.worker.discovery.host import HOST_MEMORY_POOL_ID
from edgeshard.control.worker.identity import derive_cpu_device_id
from edgeshard.control.worker.telemetry.host import HostTelemetryProbe

WORKER_ID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


async def test_sample_reports_cpu_and_host_memory() -> None:
    probe = HostTelemetryProbe(WORKER_ID, cpu_sample_interval_s=0.05)
    fragment = await probe.sample()

    assert len(fragment.device_states) == 1
    cpu = fragment.device_states[0]
    assert cpu.device_id == derive_cpu_device_id(WORKER_ID)
    assert cpu.utilization is not None
    assert 0.0 <= cpu.utilization <= 100.0
    # Generic hosts expose no CPU temperature/power through psutil (spec §17):
    # missing telemetry is None, not unavailability.
    assert cpu.temperature_c is None
    assert cpu.power_w is None
    assert cpu.availability is DeviceAvailability.AVAILABLE
    assert cpu.running_runtime_ids == ()

    assert len(fragment.memory_states) == 1
    memory = fragment.memory_states[0]
    assert memory.memory_pool_id == HOST_MEMORY_POOL_ID
    assert memory.available_bytes is not None
    assert 0 <= memory.available_bytes <= psutil.virtual_memory().total


async def test_default_interval_samples_without_blocking() -> None:
    fragment = await HostTelemetryProbe(WORKER_ID).sample()
    assert fragment.device_states[0].utilization is not None
