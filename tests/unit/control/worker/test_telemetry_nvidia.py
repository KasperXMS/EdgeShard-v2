"""NVIDIA GPU telemetry probe tests (Phase 1 spec §22, §25)."""

from __future__ import annotations

from fake_nvml import DEFAULT_UUID, FakeGpu, FakeNvml

from edgeshard.cluster.state import DeviceAvailability
from edgeshard.control.worker.discovery.nvidia import (
    NvidiaCapabilityProbe,
    gpu_memory_pool_id,
)
from edgeshard.control.worker.telemetry.base import StateFragment
from edgeshard.control.worker.telemetry.nvidia import NvidiaTelemetryProbe


async def test_init_failure_reports_empty_fragment() -> None:
    nvml = FakeNvml(init_error="driver not loaded")
    fragment = await NvidiaTelemetryProbe(nvml).sample()
    assert fragment.device_states == ()
    assert fragment.memory_states == ()
    assert nvml.shutdown_calls == 0


async def test_sample_maps_all_metrics() -> None:
    fragment = await NvidiaTelemetryProbe(FakeNvml([FakeGpu()])).sample()

    (device_state,) = fragment.device_states
    assert device_state.device_id == DEFAULT_UUID
    assert device_state.utilization == 17.0
    assert device_state.temperature_c == 45.0
    # NVML reports power in milliwatts; the domain models watts.
    assert device_state.power_w == 75.0
    assert device_state.availability is DeviceAvailability.AVAILABLE
    assert device_state.running_runtime_ids == ()

    (memory_state,) = fragment.memory_states
    assert memory_state.memory_pool_id == gpu_memory_pool_id(DEFAULT_UUID)
    assert memory_state.available_bytes == 18 * 2**30


async def test_utilization_is_clamped_into_the_domain_range() -> None:
    probe = NvidiaTelemetryProbe(FakeNvml([FakeGpu(utilization_gpu=150)]))
    (state,) = (await probe.sample()).device_states
    assert state.utilization == 100.0

    probe = NvidiaTelemetryProbe(FakeNvml([FakeGpu(utilization_gpu=-5)]))
    (state,) = (await probe.sample()).device_states
    assert state.utilization == 0.0


async def test_power_failure_degrades_only_that_metric() -> None:
    """Spec §46: metric-level failure warns and reports None, never crashes."""
    gpu = FakeGpu(fails=frozenset({"power"}))
    (state,) = (await NvidiaTelemetryProbe(FakeNvml([gpu])).sample()).device_states
    assert state.power_w is None
    assert state.utilization == 17.0
    assert state.temperature_c == 45.0
    assert state.availability is DeviceAvailability.AVAILABLE


async def test_negative_vram_free_omits_pool_state() -> None:
    gpu = FakeGpu(vram_free=-1)
    fragment = await NvidiaTelemetryProbe(FakeNvml([gpu])).sample()
    assert fragment.memory_states == ()
    assert len(fragment.device_states) == 1


async def test_device_and_pool_ids_match_discovery() -> None:
    """State fragments must only reference capability-known ids (§27/§38)."""
    nvml = FakeNvml([FakeGpu(uuid="GPU-aaa"), FakeGpu(uuid="GPU-bbb")])
    capability = NvidiaCapabilityProbe(nvml).discover()
    state = await NvidiaTelemetryProbe(FakeNvml(nvml.gpus)).sample()

    capability_device_ids = {device.identity.device_id for device in capability.devices}
    assert {s.device_id for s in state.device_states} == capability_device_ids

    capability_pool_ids = {pool.memory_pool_id for pool in capability.memory_pools}
    assert {m.memory_pool_id for m in state.memory_states} <= capability_pool_ids


async def test_handle_failure_skips_only_that_device() -> None:
    class ExplodingNvml(FakeNvml):
        def nvmlDeviceGetHandleByIndex(self, index: int) -> FakeGpu:
            if index == 1:
                raise AssertionError("handle 1 unavailable")
            return super().nvmlDeviceGetHandleByIndex(index)

    fragment = await NvidiaTelemetryProbe(
        ExplodingNvml([FakeGpu(uuid="GPU-aaa"), FakeGpu(uuid="GPU-bbb")])
    ).sample()
    assert [state.device_id for state in fragment.device_states] == ["GPU-aaa"]


async def test_default_probe_without_nvidia_hardware_never_crashes() -> None:
    """Real pynvml on a GPU-less host: empty fragment, no exception (spec §47)."""
    fragment = await NvidiaTelemetryProbe().sample()
    assert isinstance(fragment, StateFragment)
    for state in fragment.device_states:  # GPU hosts: metrics stay in range
        if state.utilization is not None:
            assert 0.0 <= state.utilization <= 100.0
