"""Fake ``pynvml`` module shape for NVIDIA backend tests (Phase 1 spec §50).

Unit tests must never require physical NVIDIA hardware; this module mirrors
only the NVML entry points the probes use, with per-GPU fault injection:
add a metric name to ``FakeGpu.fails`` and the corresponding NVML call
raises, exercising the spec §22/§47 degradation rules.
"""

from __future__ import annotations

# Reference UUID shared with tests/unit/cluster/factories.py RTX capability.
DEFAULT_UUID = "GPU-69c27179-5df5-d790-4b75-6cf18a4d2b1c"


class NVMLError(Exception):
    """Stand-in for ``pynvml.NVMLError``."""


class UtilizationRates:
    def __init__(self, gpu: int, memory: int = 0) -> None:
        self.gpu = gpu
        self.memory = memory


class MemoryInfo:
    def __init__(self, total: int, free: int, used: int = 0) -> None:
        self.total = total
        self.free = free
        self.used = used


class FakeGpu:
    """One fake GPU handle with default RTX4090-like facts."""

    def __init__(
        self,
        *,
        uuid: str | bytes = DEFAULT_UUID,
        name: str | bytes = "NVIDIA GeForce RTX 4090",
        vram_total: int = 24 * 2**30,
        vram_free: int = 18 * 2**30,
        compute_capability: tuple[int, int] = (8, 9),
        utilization_gpu: int = 17,
        temperature: int = 45,
        power_mw: int = 75_000,
        fails: frozenset[str] = frozenset(),
    ) -> None:
        self.uuid = uuid
        self.name = name
        self.vram_total = vram_total
        self.vram_free = vram_free
        self.compute_capability = compute_capability
        self.utilization_gpu = utilization_gpu
        self.temperature = temperature
        self.power_mw = power_mw
        self.fails = fails


class FakeNvml:
    """Module-shaped fake covering the probe's NVML surface."""

    NVML_TEMPERATURE_GPU = 0

    def __init__(
        self,
        gpus: list[FakeGpu] | None = None,
        *,
        init_error: str | None = None,
        count_error: str | None = None,
    ) -> None:
        self.gpus = gpus if gpus is not None else []
        self._init_error = init_error
        self._count_error = count_error
        self.shutdown_calls = 0

    # -- lifecycle ---------------------------------------------------------
    def nvmlInit(self) -> None:
        if self._init_error is not None:
            raise NVMLError(self._init_error)

    def nvmlShutdown(self) -> None:
        self.shutdown_calls += 1

    # -- enumeration -------------------------------------------------------
    def nvmlDeviceGetCount(self) -> int:
        if self._count_error is not None:
            raise NVMLError(self._count_error)
        return len(self.gpus)

    def nvmlDeviceGetHandleByIndex(self, index: int) -> FakeGpu:
        return self.gpus[index]

    def nvmlSystemGetDriverVersion(self) -> str:
        return "566.14"

    # -- per-device static facts -------------------------------------------
    def nvmlDeviceGetUUID(self, handle: FakeGpu) -> str | bytes:
        self._check(handle, "uuid")
        return handle.uuid

    def nvmlDeviceGetName(self, handle: FakeGpu) -> str | bytes:
        self._check(handle, "name")
        return handle.name

    def nvmlDeviceGetMemoryInfo(self, handle: FakeGpu) -> MemoryInfo:
        self._check(handle, "memory")
        return MemoryInfo(handle.vram_total, handle.vram_free)

    def nvmlDeviceGetCudaComputeCapability(
        self, handle: FakeGpu
    ) -> tuple[int, int]:
        self._check(handle, "compute_capability")
        return handle.compute_capability

    # -- per-device dynamic facts ------------------------------------------
    def nvmlDeviceGetUtilizationRates(
        self, handle: FakeGpu
    ) -> UtilizationRates:
        self._check(handle, "utilization")
        return UtilizationRates(handle.utilization_gpu)

    def nvmlDeviceGetTemperature(self, handle: FakeGpu, sensor: int) -> int:
        self._check(handle, "temperature")
        return handle.temperature

    def nvmlDeviceGetPowerUsage(self, handle: FakeGpu) -> int:
        self._check(handle, "power")
        return handle.power_mw

    @staticmethod
    def _check(handle: FakeGpu, metric: str) -> None:
        if metric in handle.fails:
            raise NVMLError(f"{metric} unsupported")
