"""NVIDIA capability probe tests (Phase 1 spec §11-13, §22)."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from fake_nvml import DEFAULT_UUID, FakeGpu, FakeNvml

from edgeshard.cluster.capability import DeviceCapability, MemoryModel
from edgeshard.cluster.identity import DeviceKind
from edgeshard.control.worker.discovery.base import CapabilityFragment
from edgeshard.control.worker.discovery.nvidia import (
    NvidiaCapabilityProbe,
    as_nvml_text,
    dtypes_for_compute_capability,
    fallback_device_id,
    gpu_memory_pool_id,
)


def test_init_failure_reports_empty_fragment_and_skips_shutdown() -> None:
    nvml = FakeNvml(init_error="driver not loaded")
    fragment = NvidiaCapabilityProbe(nvml).discover()
    assert fragment.architecture is None
    assert fragment.devices == ()
    assert fragment.memory_pools == ()
    assert nvml.shutdown_calls == 0


def test_single_rtx_gpu_maps_all_capability_fields() -> None:
    fragment = NvidiaCapabilityProbe(FakeNvml([FakeGpu()])).discover()

    (device,) = fragment.devices
    assert device.identity.device_id == DEFAULT_UUID
    assert device.identity.kind is DeviceKind.GPU
    assert device.identity.local_locator == "cuda:0"
    assert device.vendor == "nvidia"
    assert device.model == "NVIDIA GeForce RTX 4090"
    assert device.compute_capability == "8.9"
    assert device.supported_dtypes == ("fp32", "fp16", "bf16")
    assert device.driver_version == "566.14"
    assert device.platform_tags == ("cuda",)
    assert device.memory_pool_id == gpu_memory_pool_id(DEFAULT_UUID)

    (pool,) = fragment.memory_pools
    assert pool.memory_pool_id == gpu_memory_pool_id(DEFAULT_UUID)
    assert pool.model is MemoryModel.DISCRETE
    assert pool.total_bytes == 24 * 2**30


def test_two_gpus_both_reported_with_distinct_pools() -> None:
    nvml = FakeNvml(
        [FakeGpu(uuid="GPU-aaa"), FakeGpu(uuid="GPU-bbb", vram_total=16 * 2**30)]
    )
    fragment = NvidiaCapabilityProbe(nvml).discover()
    assert [device.identity.device_id for device in fragment.devices] == [
        "GPU-aaa",
        "GPU-bbb",
    ]
    assert {pool.memory_pool_id for pool in fragment.memory_pools} == {
        "gpu-GPU-aaa-vram",
        "gpu-GPU-bbb-vram",
    }
    totals = {pool.memory_pool_id: pool.total_bytes for pool in fragment.memory_pools}
    assert totals["gpu-GPU-bbb-vram"] == 16 * 2**30


def test_bytes_strings_are_decoded() -> None:
    gpu = FakeGpu(uuid=b"GPU-abc\x00", name=b"NVIDIA RTX\x00  ")
    fragment = NvidiaCapabilityProbe(FakeNvml([gpu])).discover()
    (device,) = fragment.devices
    assert device.identity.device_id == "GPU-abc"
    assert device.model == "NVIDIA RTX"


@pytest.mark.parametrize(
    ("failed_metric", "check"),
    [
        ("uuid", lambda device: device.identity.device_id == fallback_device_id(0)),
        ("name", lambda device: device.model == "unknown-nvidia-gpu"),
        ("compute_capability", lambda device: device.compute_capability is None),
        ("compute_capability", lambda device: device.supported_dtypes == ("fp32",)),
        ("memory", lambda device: device.memory_pool_id is None),
    ],
)
def test_metric_failure_degrades_only_that_metric(
    failed_metric: str, check: Callable[[DeviceCapability], bool]
) -> None:
    gpu = FakeGpu(fails=frozenset({failed_metric}))
    fragment = NvidiaCapabilityProbe(FakeNvml([gpu])).discover()
    (device,) = fragment.devices
    assert check(device)
    if failed_metric == "memory":
        assert fragment.memory_pools == ()


def test_handle_failure_skips_only_that_device() -> None:
    class ExplodingNvml(FakeNvml):
        def nvmlDeviceGetHandleByIndex(self, index: int) -> FakeGpu:
            if index == 0:
                raise AssertionError("handle 0 unavailable")
            return super().nvmlDeviceGetHandleByIndex(index)

    fragment = NvidiaCapabilityProbe(
        ExplodingNvml([FakeGpu(uuid="GPU-broken"), FakeGpu(uuid="GPU-healthy")])
    ).discover()
    assert [device.identity.device_id for device in fragment.devices] == ["GPU-healthy"]


def test_shutdown_called_once_per_successful_init() -> None:
    nvml = FakeNvml([FakeGpu()])
    NvidiaCapabilityProbe(nvml).discover()
    assert nvml.shutdown_calls == 1


def test_shutdown_called_even_when_enumeration_fails() -> None:
    nvml = FakeNvml([FakeGpu()], count_error="enumeration error")
    fragment = NvidiaCapabilityProbe(nvml).discover()
    assert fragment.devices == ()
    assert nvml.shutdown_calls == 1


def test_nonpositive_vram_total_reports_device_without_pool() -> None:
    fragment = NvidiaCapabilityProbe(FakeNvml([FakeGpu(vram_total=0)])).discover()
    (device,) = fragment.devices
    assert device.memory_pool_id is None
    assert fragment.memory_pools == ()


def test_default_probe_without_nvidia_hardware_never_crashes() -> None:
    """Real pynvml on a GPU-less host: empty fragment, no exception (spec §47)."""
    fragment = NvidiaCapabilityProbe().discover()
    assert isinstance(fragment, CapabilityFragment)
    assert fragment.architecture is None  # scalar facts belong to the host probe
    for device in fragment.devices:  # GPU hosts: every fact must be a GPU
        assert device.identity.kind is DeviceKind.GPU


@pytest.mark.parametrize(
    ("major", "minor", "expected"),
    [
        (6, 1, ("fp32",)),
        (7, 0, ("fp32", "fp16")),
        (7, 5, ("fp32", "fp16")),
        (8, 0, ("fp32", "fp16", "bf16")),
        (8, 9, ("fp32", "fp16", "bf16")),
        (9, 0, ("fp32", "fp16", "bf16")),
    ],
)
def test_dtypes_for_compute_capability(major: int, minor: int, expected: tuple[str, ...]) -> None:
    assert dtypes_for_compute_capability(major, minor) == expected


def test_helper_naming() -> None:
    assert gpu_memory_pool_id("GPU-abc") == "gpu-GPU-abc-vram"
    assert fallback_device_id(2) == "nvml-gpu-2"
    assert as_nvml_text("GPU-x") == "GPU-x"
    assert as_nvml_text(b"GPU-x\x00") == "GPU-x"
    assert as_nvml_text(b"") == ""
