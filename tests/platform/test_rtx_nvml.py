"""NVIDIA discrete GPU hardware validation (Phase 1 spec §49-50, P1C).

Runs the NVML probes against real hardware and is selected with
``pytest -m rtx`` on the RTX4090 host. Without an NVIDIA driver/library
these tests skip, so the default suite never requires physical hardware
(spec §50).
"""

from __future__ import annotations

import pytest

from edgeshard.cluster.capability import MemoryModel
from edgeshard.cluster.identity import DeviceKind
from edgeshard.cluster.state import DeviceAvailability
from edgeshard.control.worker.discovery.nvidia import (
    NvidiaCapabilityProbe,
    gpu_memory_pool_id,
)
from edgeshard.control.worker.telemetry.nvidia import NvidiaTelemetryProbe

pytestmark = pytest.mark.rtx


def _nvml_available() -> bool:
    try:
        import pynvml
    except ImportError:
        return False
    try:
        pynvml.nvmlInit()
        pynvml.nvmlShutdown()
    except Exception:
        return False
    return True


requires_nvml = pytest.mark.skipif(
    not _nvml_available(), reason="no NVIDIA driver / NVML library on this host"
)


@requires_nvml
def test_real_gpu_capability_fragment() -> None:
    fragment = NvidiaCapabilityProbe().discover()
    assert fragment.devices, "RTX host must report at least one GPU"
    # Scalar host facts stay with the host probe (spec §25 separation).
    assert fragment.architecture is None
    assert fragment.os is None

    pool_ids = {pool.memory_pool_id for pool in fragment.memory_pools}
    for device in fragment.devices:
        assert device.identity.kind is DeviceKind.GPU
        assert device.vendor == "nvidia"
        assert device.memory_pool_id is not None
        assert device.memory_pool_id == gpu_memory_pool_id(device.identity.device_id)
        assert device.memory_pool_id in pool_ids
    assert all(pool.model is MemoryModel.DISCRETE for pool in fragment.memory_pools)


@requires_nvml
async def test_real_gpu_telemetry_fragment() -> None:
    fragment = await NvidiaTelemetryProbe().sample()
    assert fragment.device_states, "RTX host must report GPU telemetry"
    for state in fragment.device_states:
        assert state.availability is DeviceAvailability.AVAILABLE
        if state.utilization is not None:
            assert 0.0 <= state.utilization <= 100.0
        if state.temperature_c is not None:
            assert state.temperature_c > 0.0
        if state.power_w is not None:
            assert state.power_w > 0.0


@requires_nvml
async def test_real_gpu_ids_consistent_across_probes() -> None:
    """Discovery and telemetry must derive identical device/pool ids."""
    capability = NvidiaCapabilityProbe().discover()
    state = await NvidiaTelemetryProbe().sample()

    capability_device_ids = {device.identity.device_id for device in capability.devices}
    assert {s.device_id for s in state.device_states} == capability_device_ids

    capability_pool_ids = {pool.memory_pool_id for pool in capability.memory_pools}
    assert {m.memory_pool_id for m in state.memory_states} <= capability_pool_ids


@requires_nvml
def test_real_capability_discovery_is_deterministic() -> None:
    """Two discovery runs must yield identical static facts (spec §16)."""
    first = NvidiaCapabilityProbe().discover()
    second = NvidiaCapabilityProbe().discover()
    assert first.devices == second.devices
    assert first.memory_pools == second.memory_pools
