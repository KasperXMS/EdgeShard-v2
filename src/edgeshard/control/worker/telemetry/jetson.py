"""Jetson telemetry backend (Phase 1 spec §23-24).

One long-lived ``tegrastats`` subprocess - never a fresh process per
heartbeat - feeds a parser and a latest-sample cache; ``sample()`` reads
the cache, so heartbeats never pay process-startup cost. Shared memory
state comes from ``psutil.virtual_memory()`` (spec §23); tegrastats
strings stay inside this module (spec §60).

The parser tolerates fields that differ across Jetson/L4T releases: every
field is optional and unknown/missing fields yield ``None`` (spec §24).
When ``tegrastats`` is unavailable the backend degrades: the GPU reports
``UNKNOWN`` availability with ``None`` metrics while CPU and system-memory
telemetry keep flowing.

Lifecycle: ``start`` happens lazily on first ``sample()``; the owning
Agent closes the backend after inspection (one-shot) or at shutdown
(long-lived ``worker serve``, milestone P1G).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path

import psutil

from edgeshard.cluster.state import DeviceAvailability, DeviceState, MemoryPoolState
from edgeshard.control.worker.discovery.jetson import SYSTEM_MEMORY_POOL_ID
from edgeshard.control.worker.identity import (
    derive_cpu_device_id,
    derive_jetson_gpu_device_id,
)
from edgeshard.control.worker.telemetry.base import FreshDeviceTelemetry, StateFragment

logger = logging.getLogger("worker.telemetry.jetson")

# Lines that look like tegrastats output carry at least one of these.
_TEGRAMARKERS = ("RAM ", "GR3D_FREQ", "CPU [")

_GR3D_RE = re.compile(r"GR3D_FREQ\s+(\d+(?:\.\d+)?)\s*%")
_GR3D_CLOCK_RE = re.compile(
    r"GR3D_FREQ\s+\d+(?:\.\d+)?\s*%\s*@(?:\[(\d+(?:\.\d+)?)\]|(\d+(?:\.\d+)?))"
)
_EMC_CLOCK_RE = re.compile(
    r"EMC_FREQ\s+\d+(?:\.\d+)?\s*%\s*@(?:\[(\d+(?:\.\d+)?)\]|(\d+(?:\.\d+)?))"
)
_ZONE_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9_]*)@(\d+(?:\.\d+)?)C\b")
_RAM_RE = re.compile(r"RAM\s+(\d+)/(\d+)MB")
_POWER_RAIL_RE = re.compile(
    r"\b(VDD_[A-Za-z0-9_]+)\s+(\d+(?:\.\d+)?)(m?)W(?:\s*/\s*(\d+(?:\.\d+)?)(m?)W)?"
)
"""One INA rail reading: ``VDD_X 5.2W``, ``VDD_X 18792mW`` or the
current/average pair ``VDD_X 18792mW/5552mW`` (first value = current)."""

_GPU_POWER_RAILS = ("VDD_GPU_SOC", "VDD_GPU", "VDD_CPU_GPU_CV")
"""GPU power rail preference across JetPack releases. ``VDD_IN`` is the
whole-module input power and is deliberately *not* a GPU rail."""

_GPU_CURRENT_FREQUENCY_PATTERNS = (
    "sys/class/devfreq/*gpu*/cur_freq",
    "sys/devices/platform/*gpu*/devfreq/*/cur_freq",
    "sys/devices/platform/*/*gpu*/devfreq/*/cur_freq",
)
_EMC_CURRENT_FREQUENCY_PATTERNS = (
    "sys/kernel/debug/bpmp/debug/clk/emc/rate",
    "sys/class/devfreq/*memory-controller*/cur_freq",
    "sys/class/devfreq/*emc*/cur_freq",
)


@dataclass(frozen=True)
class TegrastatsSample:
    """Parsed dynamic facts from one tegrastats line; all fields optional."""

    gpu_utilization: float | None = None
    gpu_temperature_c: float | None = None
    gpu_power_w: float | None = None
    gpu_clock_mhz: float | None = None
    emc_clock_mhz: float | None = None
    cpu_temperature_c: float | None = None
    ram_used_bytes: int | None = None
    ram_total_bytes: int | None = None


class TegrastatsParser:
    """Release-tolerant parser for tegrastats stdout lines (spec §24)."""

    def parse(self, line: str) -> TegrastatsSample | None:
        stripped = line.strip()
        if not stripped or not any(marker in stripped for marker in _TEGRAMARKERS):
            return None

        zones: dict[str, float] = {}
        for name, value in _ZONE_RE.findall(stripped):
            # Zone-name case differs by release (r32: GPU@/BCPU@/MCPU@,
            # Orin r35/r36: gpu@/cpu@) — normalize to upper case.
            zones.setdefault(name.upper(), float(value))

        gr3d = _GR3D_RE.search(stripped)
        gr3d_clock = _GR3D_CLOCK_RE.search(stripped)
        emc_clock = _EMC_CLOCK_RE.search(stripped)
        ram = _RAM_RE.search(stripped)

        return TegrastatsSample(
            gpu_utilization=float(gr3d.group(1)) if gr3d else None,
            gpu_temperature_c=zones.get("GPU"),
            gpu_power_w=self._gpu_power(stripped),
            gpu_clock_mhz=self._clock(gr3d_clock),
            emc_clock_mhz=self._clock(emc_clock),
            cpu_temperature_c=self._cpu_temperature(zones),
            ram_used_bytes=int(ram.group(1)) * 1024 * 1024 if ram else None,
            ram_total_bytes=int(ram.group(2)) * 1024 * 1024 if ram else None,
        )

    @staticmethod
    def _clock(match: re.Match[str] | None) -> float | None:
        if match is None:
            return None
        value = match.group(1) or match.group(2)
        return float(value)

    @staticmethod
    def _gpu_power(line: str) -> float | None:
        """GPU power in watts from the INA power rails (spec §24).

        Supports every observed JetPack spelling: ``VDD_GPU 5.2W`` (r32
        Nano), ``VDD_GPU_SOC 310mW`` / ``VDD_CPU_GPU_CV 15123mW/4321mW``
        (Xavier/Orin, mW and current/average pairs — the first value is the
        current draw, the second the since-boot average). Rails are matched
        in :data:`_GPU_POWER_RAILS` preference order; ``VDD_IN`` is the
        whole-module input power and never stands in for GPU power.
        """
        rails: dict[str, float] = {}
        for name, value, milli, _average, _average_milli in _POWER_RAIL_RE.findall(line):
            watts = float(value) / 1000.0 if milli else float(value)
            rails.setdefault(name.upper(), watts)
        for rail in _GPU_POWER_RAILS:
            if rail in rails:
                return rails[rail]
        return None

    @staticmethod
    def _cpu_temperature(zones: dict[str, float]) -> float | None:
        # L4T r35 reports one CPU zone; r32 reports BCPU/MCPU clusters.
        if "CPU" in zones:
            return zones["CPU"]
        cluster_temps = [zones[name] for name in ("BCPU", "MCPU") if name in zones]
        return max(cluster_temps) if cluster_temps else None


class JetsonSysfsTelemetryReader:
    """Read unprivileged instantaneous Jetson GPU facts from sysfs.

    Permission errors and unsupported sensors are normal unknown values. The
    reader never invokes a privileged helper and never substitutes a SoC or
    junction temperature for the explicitly named ``gpu-thermal`` zone.
    """

    def __init__(self, root: Path = Path("/")) -> None:
        self._root = root

    def gpu_clock_mhz(self) -> float | None:
        return self._first_frequency_mhz(_GPU_CURRENT_FREQUENCY_PATTERNS)

    def emc_clock_mhz(self) -> float | None:
        return self._first_frequency_mhz(_EMC_CURRENT_FREQUENCY_PATTERNS)

    def gpu_temperature_c(self) -> float | None:
        for type_path in sorted(
            self._root.glob("sys/class/thermal/thermal_zone*/type")
        ):
            try:
                zone_type = type_path.read_text(encoding="ascii").strip()
            except (OSError, UnicodeError):
                continue
            if zone_type.casefold() != "gpu-thermal":
                continue
            try:
                millidegrees = int(
                    type_path.with_name("temp").read_text(encoding="ascii").strip()
                )
            except (OSError, ValueError):
                return None
            return millidegrees / 1000.0
        return None

    def _first_frequency_mhz(self, patterns: tuple[str, ...]) -> float | None:
        for pattern in patterns:
            for path in sorted(self._root.glob(pattern)):
                try:
                    hz = int(path.read_text(encoding="ascii").strip())
                except (OSError, ValueError):
                    continue
                if hz > 0:
                    return hz / 1_000_000.0
        return None


class TegrastatsProcess:
    """The single long-lived tegrastats reader (spec §24).

    ``start`` spawns the subprocess once; a background task consumes stdout,
    parses each line, and stores the latest sample. ``wait_first`` lets
    one-shot callers wait (bounded) for the first parsed sample.
    """

    def __init__(
        self,
        *,
        command: tuple[str, ...] = ("tegrastats",),
        parser: TegrastatsParser | None = None,
    ) -> None:
        self._command = command
        self._parser = parser or TegrastatsParser()
        self._process: asyncio.subprocess.Process | None = None
        self._task: asyncio.Task[None] | None = None
        self._latest: TegrastatsSample | None = None
        self._first = asyncio.Event()

    @property
    def latest(self) -> TegrastatsSample | None:
        return self._latest

    async def start(self) -> None:
        if self._process is not None or self._first.is_set():
            return
        try:
            self._process = await asyncio.create_subprocess_exec(
                *self._command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except OSError as exc:  # no tegrastats binary is non-fatal (spec §47)
            logger.warning("tegrastats unavailable; GPU telemetry will be unknown: %s", exc)
            self._first.set()
            return
        self._task = asyncio.create_task(self._read_loop())

    async def _read_loop(self) -> None:
        assert self._process is not None
        stdout = self._process.stdout
        assert stdout is not None
        try:
            while True:
                raw = await stdout.readline()
                if not raw:
                    break
                sample = self._parser.parse(raw.decode("utf-8", "replace"))
                if sample is not None:
                    self._latest = sample
                    self._first.set()
        finally:
            self._first.set()  # release waiters even without a sample

    async def wait_first(self, timeout_s: float) -> TegrastatsSample | None:
        """Latest sample, waiting up to ``timeout_s`` for the first one."""
        if not self._first.is_set():
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._first.wait(), timeout_s)
        if self._latest is None:
            logger.info("no tegrastats sample within %.1fs; GPU metrics None", timeout_s)
        return self._latest

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        process = self._process
        if process is not None and process.returncode is None:
            with contextlib.suppress(ProcessLookupError, OSError):
                process.terminate()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(process.wait(), timeout=5.0)
            if process.returncode is None:
                with contextlib.suppress(ProcessLookupError, OSError):
                    process.kill()
                await process.wait()
        self._process = None
        self._latest = None
        self._first.clear()


class JetsonTelemetryBackend:
    """Jetson CPU/GPU/memory telemetry (spec §23).

    CPU utilization and shared-memory availability come from psutil (the
    host-equivalent facts); the integrated GPU's utilization, temperature,
    and power come from the cached tegrastats sample.
    """

    def __init__(
        self,
        worker_id: str,
        *,
        tegrastats: TegrastatsProcess | None = None,
        sysfs_root: Path = Path("/"),
        cpu_sample_interval_s: float | None = None,
        first_sample_timeout_s: float = 3.0,
    ) -> None:
        self._cpu_device_id = derive_cpu_device_id(worker_id)
        self._gpu_device_id = derive_jetson_gpu_device_id(worker_id)
        self._tegrastats = tegrastats or TegrastatsProcess()
        self._sysfs = JetsonSysfsTelemetryReader(sysfs_root)
        self._cpu_sample_interval_s = cpu_sample_interval_s
        self._first_sample_timeout_s = first_sample_timeout_s
        self._sample_lock = threading.Lock()

    async def sample(self) -> StateFragment:
        await self._tegrastats.start()
        sample = self._tegrastats.latest
        if sample is None:
            await self._tegrastats.wait_first(self._first_sample_timeout_s)
        return await asyncio.to_thread(self._sample_blocking, self._cpu_sample_interval_s)

    def sample_fresh(self) -> StateFragment:
        """Read fresh physical state without restarting ``tegrastats``."""
        return self._sample_blocking(None)

    def sample_fresh_device(self, device_id: str) -> FreshDeviceTelemetry | None:
        """Fresh unprivileged sysfs facts plus cached long-lived tegrastats."""
        if device_id != self._gpu_device_id:
            return None
        with self._sample_lock:
            sample = self._tegrastats.latest
        return FreshDeviceTelemetry(
            device_id=device_id,
            utilization=sample.gpu_utilization if sample else None,
            temperature_c=self._sysfs.gpu_temperature_c(),
            power_w=sample.gpu_power_w if sample else None,
            clock_mhz=self._sysfs.gpu_clock_mhz(),
            emc_clock_mhz=self._sysfs.emc_clock_mhz(),
        )

    def _sample_blocking(self, cpu_interval_s: float | None) -> StateFragment:
        with self._sample_lock:
            sample = self._tegrastats.latest
            memory = psutil.virtual_memory()
            utilization = psutil.cpu_percent(interval=cpu_interval_s)
        gpu_temperature = self._sysfs.gpu_temperature_c()
        gpu_clock = self._sysfs.gpu_clock_mhz()
        utilization = max(0.0, min(100.0, float(utilization)))

        return StateFragment(
            device_states=(
                DeviceState(
                    device_id=self._cpu_device_id,
                    utilization=utilization,
                    temperature_c=sample.cpu_temperature_c if sample else None,
                    power_w=None,
                    availability=DeviceAvailability.AVAILABLE,
                    running_runtime_ids=(),
                ),
                DeviceState(
                    device_id=self._gpu_device_id,
                    utilization=sample.gpu_utilization if sample else None,
                    temperature_c=gpu_temperature,
                    power_w=sample.gpu_power_w if sample else None,
                    # A live tegrastats sample or readable GPU devfreq clock
                    # proves the device answers. Otherwise state is UNKNOWN.
                    availability=(
                        DeviceAvailability.AVAILABLE
                        if sample is not None or gpu_clock is not None
                        else DeviceAvailability.UNKNOWN
                    ),
                    running_runtime_ids=(),
                ),
            ),
            memory_states=(
                MemoryPoolState(
                    memory_pool_id=SYSTEM_MEMORY_POOL_ID,
                    available_bytes=int(memory.available),
                ),
            ),
        )

    async def close(self) -> None:
        """Stop the long-lived tegrastats reader (spec §24 lifecycle)."""
        await self._tegrastats.stop()
