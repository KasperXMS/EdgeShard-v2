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
from collections.abc import Callable, Sequence
from typing import Any

from edgeshard.cluster.state import DeviceAvailability, DeviceState, MemoryPoolState
from edgeshard.control.worker.discovery.nvidia import (
    derive_gpu_device_id,
    gpu_memory_pool_id,
    import_nvml,
)
from edgeshard.control.worker.telemetry.base import StateFragment

logger = logging.getLogger("worker.telemetry.nvidia")

StaticDeviceIds = Callable[[], Sequence[str]]
"""Supplies the device ids static discovery produced, in enumeration order
(``NvidiaCapabilityProbe.discovered_device_ids``)."""


class NvidiaTelemetryProbe:
    """Per-GPU dynamic telemetry (spec §22).

    Device ids must match the capability the Master stores. When
    ``static_device_ids`` covers every currently enumerated GPU, those ids
    are reused verbatim: a transient NVML UUID failure during a heartbeat
    sample can then never degrade a known ``GPU-<uuid>`` id to an
    ``nvml-gpu-0`` ordinal and desynchronize capability from state. Only
    when the mapping is absent or stale (count changed — hot-plug/removal)
    does each GPU fall back to the same UUID → PCI → ordinal chain
    discovery uses.
    """

    def __init__(
        self, nvml: Any | None = None, *, static_device_ids: StaticDeviceIds | None = None
    ) -> None:
        self._nvml = nvml
        self._static_device_ids = static_device_ids

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

        static_ids = self._matching_static_ids(count)
        device_states: list[DeviceState] = []
        memory_states: list[MemoryPoolState] = []
        for index in range(count):
            try:
                handle = nvml.nvmlDeviceGetHandleByIndex(index)
                device_id = (
                    static_ids[index]
                    if static_ids is not None
                    else derive_gpu_device_id(nvml, handle, index)
                )
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

    def _matching_static_ids(self, count: int) -> tuple[str, ...] | None:
        """The static-discovery ids when they cover all ``count`` GPUs."""
        if self._static_device_ids is None:
            return None
        try:
            ids = tuple(self._static_device_ids())
        except Exception as exc:
            logger.warning("static device id mapping unavailable: %s", exc)
            return None
        if len(ids) != count:
            # GPU set changed since discovery (or a GPU was skipped there):
            # the mapping no longer aligns with enumeration order.
            logger.warning(
                "static device id mapping covers %d GPUs but NVML enumerates %d; "
                "re-deriving ids per GPU this sample",
                len(ids),
                count,
            )
            return None
        return ids

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
