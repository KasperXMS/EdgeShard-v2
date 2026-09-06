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
    pci_bus_device_id,
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
        # uuid alone: the §11 chain drops to the PCI bus id, never the ordinal.
        (
            "uuid",
            lambda device: device.identity.device_id
            == pci_bus_device_id("00000000:01:00.0"),
        ),
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


def test_device_id_chain_uuid_then_pci_then_ordinal() -> None:
    """Spec §11: UUID → PCI bus id → enumeration ordinal, in that order."""
    # 1) UUID available: it wins even when PCI would also work.
    gpu = FakeGpu(uuid="GPU-aaa", pci_bus_id="00000000:2a:00.0")
    (device,) = NvidiaCapabilityProbe(FakeNvml([gpu])).discover().devices
    assert device.identity.device_id == "GPU-aaa"

    # 2) UUID fails, PCI available: uppercased PCI bus id, and the VRAM pool
    #    follows the derived id so state/capability stay consistent.
    gpu = FakeGpu(fails=frozenset({"uuid"}), pci_bus_id="00000000:2a:00.0")
    (device,) = NvidiaCapabilityProbe(FakeNvml([gpu])).discover().devices
    assert device.identity.device_id == "pci-00000000:2A:00.0"
    assert device.memory_pool_id == gpu_memory_pool_id("pci-00000000:2A:00.0")

    # 3) Both fail: last-resort ordinal, warned about loudly.
    gpu = FakeGpu(fails=frozenset({"uuid", "pci"}))
    (device,) = NvidiaCapabilityProbe(FakeNvml([gpu])).discover().devices
    assert device.identity.device_id == fallback_device_id(0)


def test_pci_bus_id_bytes_are_decoded() -> None:
    gpu = FakeGpu(fails=frozenset({"uuid"}), pci_bus_id=b"00000000:01:00.0\x00")
    (device,) = NvidiaCapabilityProbe(FakeNvml([gpu])).discover().devices
    assert device.identity.device_id == "pci-00000000:01:00.0"


def test_discovered_device_ids_track_enumeration_order() -> None:
    """Telemetry reuses this mapping; it must mirror exactly what was found."""
    probe = NvidiaCapabilityProbe(
        FakeNvml([FakeGpu(uuid="GPU-aaa"), FakeGpu(uuid="GPU-bbb")])
    )
    assert probe.discovered_device_ids == ()  # nothing discovered yet

    probe.discover()
    assert probe.discovered_device_ids == ("GPU-aaa", "GPU-bbb")

    # A GPU whose discovery failed is absent — the mapping never carries ids
    # the capability does not, and never pads with placeholders.
    class ExplodingNvml(FakeNvml):
        def nvmlDeviceGetHandleByIndex(self, index: int) -> FakeGpu:
            if index == 0:
                raise AssertionError("handle 0 unavailable")
            return super().nvmlDeviceGetHandleByIndex(index)

    degraded = NvidiaCapabilityProbe(
        ExplodingNvml([FakeGpu(uuid="GPU-broken"), FakeGpu(uuid="GPU-healthy")])
    )
    degraded.discover()
    assert degraded.discovered_device_ids == ("GPU-healthy",)


def test_discovered_device_ids_reset_when_nvml_unavailable() -> None:
    probe = NvidiaCapabilityProbe(FakeNvml([FakeGpu(uuid="GPU-aaa")]))
    probe.discover()
    assert probe.discovered_device_ids == ("GPU-aaa",)

    # A later discovery without a driver must not keep serving stale ids.
    probe._nvml = FakeNvml(init_error="driver not loaded")
    probe.discover()
    assert probe.discovered_device_ids == ()


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
