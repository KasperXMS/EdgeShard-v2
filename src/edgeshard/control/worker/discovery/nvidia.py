"""NVIDIA discrete GPU capability probe via NVML (Phase 1 spec §22).

Uses ``nvidia-ml-py`` (NVML) directly - never parses ``nvidia-smi``. NVML
types stay inside this adapter (spec §60); the emitted fragment is pure
cluster domain.

Device identity (spec §11): the stable NVIDIA GPU UUID when available. Each
GPU contributes an independent DISCRETE VRAM pool (spec §13) named
``gpu-<uuid>-vram``.

An absent NVIDIA library or driver is non-fatal (spec §47): the probe
reports an empty fragment, so ``worker inspect`` works unchanged on hosts
without NVIDIA GPUs. Failures of individual metrics degrade only that
metric, never the whole device or probe (spec §22).
"""

from __future__ import annotations

import logging
from typing import Any

from edgeshard.cluster.capability import (
    DeviceCapability,
    MemoryModel,
    MemoryPoolCapability,
)
from edgeshard.cluster.identity import DeviceIdentity, DeviceKind
from edgeshard.control.worker.discovery.base import CapabilityFragment

logger = logging.getLogger("worker.discovery.nvidia")

NVIDIA_VENDOR = "nvidia"


def import_nvml() -> Any:
    """Import the NVML binding lazily (``nvidia-ml-py``, worker extra)."""
    import pynvml

    return pynvml


def gpu_memory_pool_id(gpu_device_id: str) -> str:
    """VRAM pool id of one discrete GPU (spec §13: ``gpu-<uuid>-vram``)."""
    return f"gpu-{gpu_device_id}-vram"


def fallback_device_id(index: int) -> str:
    """Device id used when NVML reports no GPU UUID (spec §11 fallback).

    Enumeration-index ids are not guaranteed stable across reboots; callers
    warn whenever this fallback is used.
    """
    return f"nvml-gpu-{index}"


def as_nvml_text(value: Any) -> str:
    """Normalize NVML strings (str, or NUL-padded bytes) to plain ``str``."""
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    return str(value).split("\x00", 1)[0].strip()


def dtypes_for_compute_capability(major: int, minor: int) -> tuple[str, ...]:
    """Conservative dtype set implied by a CUDA compute capability.

    Every CUDA GPU computes fp32; tensor-core half precision arrived with
    Volta (7.0) and bfloat16 with Ampere (8.0).
    """
    dtypes = ["fp32"]
    if (major, minor) >= (7, 0):
        dtypes.append("fp16")
    if (major, minor) >= (8, 0):
        dtypes.append("bf16")
    return tuple(dtypes)


class NvidiaCapabilityProbe:
    """Static discovery of discrete NVIDIA GPUs (spec §22)."""

    def __init__(self, nvml: Any | None = None) -> None:
        self._nvml = nvml

    def discover(self) -> CapabilityFragment:
        nvml = self._nvml if self._nvml is not None else import_nvml()
        try:
            nvml.nvmlInit()
        except Exception as exc:  # no driver/library is non-fatal (spec §47)
            logger.warning("NVML unavailable; reporting no NVIDIA GPUs: %s", exc)
            return CapabilityFragment()
        try:
            return self._discover(nvml)
        finally:
            try:
                nvml.nvmlShutdown()
            except Exception as exc:  # shutdown best-effort; discovery done
                logger.warning("NVML shutdown failed: %s", exc)

    def _discover(self, nvml: Any) -> CapabilityFragment:
        driver_version = self._driver_version(nvml)
        try:
            count = int(nvml.nvmlDeviceGetCount())
        except Exception as exc:
            logger.warning("GPU enumeration failed; reporting no GPUs: %s", exc)
            return CapabilityFragment()

        devices: list[DeviceCapability] = []
        pools: list[MemoryPoolCapability] = []
        for index in range(count):
            try:
                handle = nvml.nvmlDeviceGetHandleByIndex(index)
                device, pool = self._discover_gpu(nvml, handle, index, driver_version)
            except Exception as exc:
                logger.warning("GPU %d discovery failed; skipping device: %s", index, exc)
                continue
            devices.append(device)
            if pool is not None:
                pools.append(pool)
        return CapabilityFragment(devices=tuple(devices), memory_pools=tuple(pools))

    def _driver_version(self, nvml: Any) -> str | None:
        try:
            return as_nvml_text(nvml.nvmlSystemGetDriverVersion()) or None
        except Exception as exc:
            logger.warning("metric=driver_version unavailable: %s", exc)
            return None

    def _discover_gpu(
        self, nvml: Any, handle: Any, index: int, driver_version: str | None
    ) -> tuple[DeviceCapability, MemoryPoolCapability | None]:
        device_id = self._device_id(nvml, handle, index)
        model = self._model(nvml, handle)
        compute_capability, dtypes = self._compute_capability(nvml, handle)
        pool = self._vram_pool(nvml, handle, device_id)
        device = DeviceCapability(
            identity=DeviceIdentity(
                device_id=device_id,
                kind=DeviceKind.GPU,
                # CUDA ordinals are informational only - never identity
                # (spec §11); the UUID above is the stable id.
                local_locator=f"cuda:{index}",
            ),
            vendor=NVIDIA_VENDOR,
            model=model,
            compute_capability=compute_capability,
            memory_pool_id=pool.memory_pool_id if pool is not None else None,
            supported_dtypes=dtypes,
            driver_version=driver_version,
            platform_tags=("cuda",),
        )
        return device, pool

    def _device_id(self, nvml: Any, handle: Any, index: int) -> str:
        try:
            uuid = as_nvml_text(nvml.nvmlDeviceGetUUID(handle))
        except Exception as exc:
            logger.warning("metric=gpu_uuid unavailable for GPU %d: %s", index, exc)
            uuid = ""
        if uuid:
            return uuid
        logger.warning(
            "GPU %d has no NVML UUID; falling back to an enumeration-order id "
            "that is not guaranteed stable (spec §11, §58)",
            index,
        )
        return fallback_device_id(index)

    def _model(self, nvml: Any, handle: Any) -> str:
        try:
            name = as_nvml_text(nvml.nvmlDeviceGetName(handle))
        except Exception as exc:
            logger.warning("metric=gpu_name unavailable: %s", exc)
            return "unknown-nvidia-gpu"
        return name or "unknown-nvidia-gpu"

    def _compute_capability(
        self, nvml: Any, handle: Any
    ) -> tuple[str | None, tuple[str, ...]]:
        try:
            major, minor = nvml.nvmlDeviceGetCudaComputeCapability(handle)
        except Exception as exc:
            logger.warning("metric=compute_capability unavailable: %s", exc)
            return None, ("fp32",)
        major_int, minor_int = int(major), int(minor)
        return (
            f"{major_int}.{minor_int}",
            dtypes_for_compute_capability(major_int, minor_int),
        )

    def _vram_pool(
        self, nvml: Any, handle: Any, device_id: str
    ) -> MemoryPoolCapability | None:
        try:
            total = int(nvml.nvmlDeviceGetMemoryInfo(handle).total)
        except Exception as exc:
            logger.warning("metric=vram_total unavailable for %s: %s", device_id, exc)
            return None
        if total <= 0:
            logger.warning("NVML reported non-positive VRAM total for %s", device_id)
            return None
        return MemoryPoolCapability(
            memory_pool_id=gpu_memory_pool_id(device_id),
            model=MemoryModel.DISCRETE,
            total_bytes=total,
        )
