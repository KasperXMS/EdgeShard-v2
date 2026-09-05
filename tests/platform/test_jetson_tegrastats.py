"""Jetson hardware validation (Phase 1 spec §49-50, §54, milestone P1D).

Runs the Jetson platform probe and tegrastats backend against real AGX
Orin hardware; selected with ``pytest -m jetson``. Off-Jetson hosts skip
these tests, so the default suite never requires physical hardware
(spec §50).
"""

from __future__ import annotations

import uuid

import pytest

from edgeshard.cluster.capability import MemoryModel
from edgeshard.cluster.identity import DeviceKind
from edgeshard.control.worker.discovery.jetson import (
    SYSTEM_MEMORY_POOL_ID,
    JetsonPlatformProbe,
    is_jetson_host,
)
from edgeshard.control.worker.telemetry.jetson import JetsonTelemetryBackend

pytestmark = pytest.mark.jetson

requires_jetson = pytest.mark.skipif(not is_jetson_host(), reason="not a Jetson host")


def _worker_id() -> str:
    return str(uuid.uuid4())


@requires_jetson
def test_platform_probe_reports_shared_system_memory() -> None:
    """§54: aarch64, CPU+GPU, both referencing one shared system-memory."""
    fragment = JetsonPlatformProbe(_worker_id()).discover()

    assert fragment.architecture == "aarch64"

    (pool,) = fragment.memory_pools
    assert pool.memory_pool_id == SYSTEM_MEMORY_POOL_ID
    assert pool.model is MemoryModel.SHARED

    kinds = {device.identity.kind for device in fragment.devices}
    assert kinds == {DeviceKind.CPU, DeviceKind.GPU}
    # Never two independent memory resources (spec §14).
    assert {device.memory_pool_id for device in fragment.devices} == {
        SYSTEM_MEMORY_POOL_ID
    }
    assert all("tegra" in device.platform_tags for device in fragment.devices)


@requires_jetson
async def test_tegrastats_backend_reports_state() -> None:
    backend = JetsonTelemetryBackend(_worker_id(), first_sample_timeout_s=5.0)
    try:
        fragment = await backend.sample()
    finally:
        await backend.close()

    (memory_state,) = fragment.memory_states
    assert memory_state.memory_pool_id == SYSTEM_MEMORY_POOL_ID
    assert memory_state.available_bytes is not None
    assert memory_state.available_bytes > 0

    states = {state.device_id: state for state in fragment.device_states}
    assert len(states) == 2  # CPU + integrated GPU
    for state in fragment.device_states:
        if state.utilization is not None:
            assert 0.0 <= state.utilization <= 100.0
        if state.temperature_c is not None:
            assert state.temperature_c > 0.0


@requires_jetson
def test_jetson_is_not_modeled_as_discrete_nvml_gpu() -> None:
    """§23: the probe emits no per-GPU discrete VRAM pools."""
    fragment = JetsonPlatformProbe(_worker_id()).discover()
    assert all(
        pool.model is not MemoryModel.DISCRETE for pool in fragment.memory_pools
    )
    gpu = next(
        device
        for device in fragment.devices
        if device.identity.kind is DeviceKind.GPU
    )
    assert gpu.identity.local_locator == "igpu"
