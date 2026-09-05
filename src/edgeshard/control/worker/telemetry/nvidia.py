"""NVIDIA discrete GPU telemetry via NVML (Phase 1 spec §22, §25).

Dynamic per-GPU facts: utilization, temperature, power, and free VRAM. NVML
calls are synchronous and may block briefly, so sampling runs in a worker
thread and the heartbeat loop stays async (spec §25).

Missing individual metrics are ``None`` and logged (spec §17, §46:
``WARN worker.telemetry probe=nvidia metric=power unavailable``); they never
fail the probe and never imply device unavailability. An absent NVML driver
yields an empty fragment, mirroring discovery.

Device and pool ids are derived exactly as in discovery (``nvidia.py``), so
state fragments always reference capability-known identities.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from edgeshard.cluster.state import DeviceAvailability, DeviceState, MemoryPoolState
from edgeshard.control.worker.discovery.nvidia import (
    as_nvml_text,
    fallback_device_id,
    gpu_memory_pool_id,
    import_nvml,
)
from edgeshard.control.worker.telemetry.base import StateFragment

logger = logging.getLogger("worker.telemetry.nvidia")


class NvidiaTelemetryProbe:
    """Per-GPU dynamic telemetry (spec §22)."""

    def __init__(self, nvml: Any | None = None) -> None:
        self._nvml = nvml

    async def sample(self) -> StateFragment:
        return await asyncio.to_thread(self._sample_blocking)

    def _sample_blocking(self) -> StateFragment:
        nvml = self._nvml if self._nvml is not None else import_nvml()
        try:
            nvml.nvmlInit()
        except Exception as exc:  # no driver/library is non-fatal (spec §47)
            logger.warning("NVML unavailable; reporting no GPU state: %s", exc)
            return StateFragment()
        try:
            return self._sample(nvml)
        finally:
            try:
                nvml.nvmlShutdown()
            except Exception as exc:  # shutdown best-effort; sampling done
                logger.warning("NVML shutdown failed: %s", exc)

    def _sample(self, nvml: Any) -> StateFragment:
        try:
            count = int(nvml.nvmlDeviceGetCount())
        except Exception as exc:
            logger.warning("GPU enumeration failed; reporting no GPU state: %s", exc)
            return StateFragment()

        device_states: list[DeviceState] = []
        memory_states: list[MemoryPoolState] = []
        for index in range(count):
            try:
                handle = nvml.nvmlDeviceGetHandleByIndex(index)
                device_id = self._device_id(nvml, handle, index)
            except Exception as exc:
                logger.warning("GPU %d sampling failed; skipping device: %s", index, exc)
                continue
            device_states.append(
                DeviceState(
                    device_id=device_id,
                    utilization=self._utilization(nvml, handle),
                    temperature_c=self._temperature(nvml, handle),
                    power_w=self._power(nvml, handle),
                    # The handle answers queries; unsupported individual
                    # metrics stay None without downgrading availability
                    # (spec §17).
                    availability=DeviceAvailability.AVAILABLE,
                    running_runtime_ids=(),
                )
            )
            available = self._vram_free(nvml, handle)
            if available is not None:
                memory_states.append(
                    MemoryPoolState(
                        memory_pool_id=gpu_memory_pool_id(device_id),
                        available_bytes=available,
                    )
                )
        return StateFragment(
            device_states=tuple(device_states),
            memory_states=tuple(memory_states),
        )

    def _device_id(self, nvml: Any, handle: Any, index: int) -> str:
        try:
            uuid = as_nvml_text(nvml.nvmlDeviceGetUUID(handle))
        except Exception as exc:
            logger.warning("metric=gpu_uuid unavailable for GPU %d: %s", index, exc)
            uuid = ""
        return uuid or fallback_device_id(index)

    def _utilization(self, nvml: Any, handle: Any) -> float | None:
        try:
            rates = nvml.nvmlDeviceGetUtilizationRates(handle)
        except Exception as exc:
            logger.warning("metric=utilization unavailable: %s", exc)
            return None
        return max(0.0, min(100.0, float(rates.gpu)))

    def _temperature(self, nvml: Any, handle: Any) -> float | None:
        try:
            return float(nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU))
        except Exception as exc:
            logger.warning("metric=temperature unavailable: %s", exc)
            return None

    def _power(self, nvml: Any, handle: Any) -> float | None:
        try:
            return float(nvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0
        except Exception as exc:
            logger.warning("metric=power unavailable: %s", exc)
            return None

    def _vram_free(self, nvml: Any, handle: Any) -> int | None:
        try:
            free = int(nvml.nvmlDeviceGetMemoryInfo(handle).free)
        except Exception as exc:
            logger.warning("metric=vram_free unavailable: %s", exc)
            return None
        return free if free >= 0 else None
