"""Jetson platform probe tests (Phase 1 spec §11, §14, §23, §54)."""

from __future__ import annotations

import platform
import uuid
from pathlib import Path

import psutil
import pytest

from edgeshard.cluster.capability import MemoryModel
from edgeshard.cluster.identity import DeviceKind
from edgeshard.control.worker.agent import assemble_capability
from edgeshard.control.worker.discovery import jetson as jetson_module
from edgeshard.control.worker.discovery.host import canonical_architecture
from edgeshard.control.worker.discovery.jetson import (
    SYSTEM_MEMORY_POOL_ID,
    JetsonPlatformProbe,
    compute_capability_for_model,
    is_jetson_host,
    read_device_tree_model,
    read_l4t_release,
)
from edgeshard.control.worker.identity import (
    derive_cpu_device_id,
    derive_jetson_gpu_device_id,
)

L4T_LINE = (
    "# R35 (release), REVISION: 4.1, GCID: 36144147, BOARD: t186ref, EABI: aarch64\n"
)
ORIN_MODEL = "NVIDIA AGX Orin Developer Kit"

WORKER_ID = str(uuid.uuid4())


def _failing_docker() -> object:
    raise RuntimeError("no docker daemon in this test")


@pytest.fixture
def orin_platform_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the Jetson platform file constants at realistic fixtures."""
    release = tmp_path / "nv_tegra_release"
    release.write_text(L4T_LINE, encoding="utf-8")
    model = tmp_path / "model"
    model.write_bytes(ORIN_MODEL.encode("utf-8") + b"\x00")
    monkeypatch.setattr(jetson_module, "NV_TEGRA_RELEASE", release)
    monkeypatch.setattr(jetson_module, "DEVICE_TREE_MODEL", model)


def test_is_jetson_host_false_without_platform_files() -> None:
    # Neither /etc/nv_tegra_release nor the device tree exists off-Jetson.
    assert is_jetson_host() is False


def test_is_jetson_host_true_from_release_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = tmp_path / "nv_tegra_release"
    release.write_text(L4T_LINE, encoding="utf-8")
    monkeypatch.setattr(jetson_module, "NV_TEGRA_RELEASE", release)
    assert is_jetson_host() is True


def test_is_jetson_host_true_from_device_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compatible = tmp_path / "compatible"
    compatible.write_bytes(b"nvidia,p3767-0000\x00nvidia,tegra234\x00")
    monkeypatch.setattr(jetson_module, "NV_TEGRA_RELEASE", tmp_path / "missing")
    monkeypatch.setattr(jetson_module, "DEVICE_TREE_COMPATIBLE", compatible)
    assert is_jetson_host() is True


def test_is_jetson_host_false_for_other_device_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compatible = tmp_path / "compatible"
    compatible.write_bytes(b"raspberrypi,4-model-b\x00brcm,bcm2711\x00")
    monkeypatch.setattr(jetson_module, "NV_TEGRA_RELEASE", tmp_path / "missing")
    monkeypatch.setattr(jetson_module, "DEVICE_TREE_COMPATIBLE", compatible)
    assert is_jetson_host() is False


def test_read_l4t_release_parses_standard_line(orin_platform_files: None) -> None:
    assert read_l4t_release() == "35.4.1"


def test_read_l4t_release_tolerates_garbage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = tmp_path / "nv_tegra_release"
    release.write_text("not an L4T header", encoding="utf-8")
    monkeypatch.setattr(jetson_module, "NV_TEGRA_RELEASE", release)
    assert read_l4t_release() is None


def test_read_l4t_release_missing_file_is_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(jetson_module, "NV_TEGRA_RELEASE", tmp_path / "missing")
    assert read_l4t_release() is None


def test_read_device_tree_model_strips_nul(orin_platform_files: None) -> None:
    assert read_device_tree_model() == ORIN_MODEL


def test_read_device_tree_model_missing_file_is_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(jetson_module, "DEVICE_TREE_MODEL", tmp_path / "missing")
    assert read_device_tree_model() is None


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("NVIDIA AGX Orin Developer Kit", (8, 7)),
        ("Jetson Orin Nano", (8, 7)),  # "orin" must win over "nano"
        ("NVIDIA Jetson AGX Xavier", (7, 2)),
        ("Jetson TX2", (6, 2)),
        ("Jetson TX1", (5, 3)),
        ("Jetson Nano", (5, 3)),
        ("Raspberry Pi 4", None),
        (None, None),
    ],
)
def test_compute_capability_for_model(
    model: str | None, expected: tuple[int, int] | None
) -> None:
    assert compute_capability_for_model(model) == expected


def test_orin_fragment_shared_memory_topology(orin_platform_files: None) -> None:
    """§14/§54: one shared system-memory pool referenced by CPU and GPU.

    Generic host facts are inherited from the delegated host probe, so the
    architecture matches this test machine (aarch64 on real Orin hardware).
    """
    probe = JetsonPlatformProbe(WORKER_ID, docker_client_factory=_failing_docker)
    fragment = probe.discover()

    assert fragment.architecture == canonical_architecture(platform.machine())

    (pool,) = fragment.memory_pools
    assert pool.memory_pool_id == SYSTEM_MEMORY_POOL_ID
    assert pool.model is MemoryModel.SHARED
    assert pool.total_bytes == int(psutil.virtual_memory().total)

    cpu, gpu = fragment.devices
    assert cpu.identity.device_id == derive_cpu_device_id(WORKER_ID)
    assert cpu.identity.kind is DeviceKind.CPU
    assert cpu.vendor == "nvidia"
    assert "tegra" in cpu.platform_tags
    assert cpu.memory_pool_id == SYSTEM_MEMORY_POOL_ID

    assert gpu.identity.device_id == derive_jetson_gpu_device_id(WORKER_ID)
    assert gpu.identity.kind is DeviceKind.GPU
    assert gpu.identity.local_locator == "igpu"
    assert gpu.model == ORIN_MODEL
    assert gpu.compute_capability == "8.7"
    assert gpu.supported_dtypes == ("fp32", "fp16", "bf16")
    assert gpu.memory_pool_id == SYSTEM_MEMORY_POOL_ID
    assert "sm87" in gpu.platform_tags
    assert "l4t-35.4.1" in gpu.platform_tags

    # Never two independent memory resources (spec §14).
    assert {device.memory_pool_id for device in fragment.devices} == {
        SYSTEM_MEMORY_POOL_ID
    }


def test_fragment_assembles_into_valid_capability(orin_platform_files: None) -> None:
    probe = JetsonPlatformProbe(WORKER_ID, docker_client_factory=_failing_docker)
    capability = assemble_capability([probe.discover()])
    assert capability.capability_revision
    assert {device.identity.kind for device in capability.devices} == {
        DeviceKind.CPU,
        DeviceKind.GPU,
    }


def test_unknown_model_degrades_gpu_facts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No device tree / release file: GPU still reported, facts None (§47)."""
    monkeypatch.setattr(jetson_module, "DEVICE_TREE_MODEL", tmp_path / "missing")
    monkeypatch.setattr(jetson_module, "NV_TEGRA_RELEASE", tmp_path / "missing")
    probe = JetsonPlatformProbe(WORKER_ID, docker_client_factory=_failing_docker)
    fragment = probe.discover()

    _, gpu = fragment.devices
    assert gpu.model == "Jetson integrated GPU"
    assert gpu.compute_capability is None
    assert gpu.supported_dtypes == ("fp32",)
    assert gpu.platform_tags == ("cuda", "tegra")
