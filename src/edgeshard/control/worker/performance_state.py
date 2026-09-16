"""Read-only Jetson operating-policy provenance for compute profiling."""

from __future__ import annotations

import logging
import re
import subprocess
from collections.abc import Callable, Iterable
from pathlib import Path

from edgeshard.profiling.domain.environment import PerformanceState

logger = logging.getLogger("worker.performance_state")

CommandRunner = Callable[[tuple[str, ...]], str | None]
_POWER_MODE_RE = re.compile(r"^NV Power Mode:\s*(\S.*?)\s*$", re.MULTILINE)
_JETSON_CLOCKS_EMC_RE = re.compile(
    r"\bEMC\s+MinFreq=(\d+)\s+MaxFreq=(\d+)"
    r"(?:\s+CurrentFreq=(\d+))?\b"
)


def normalize_performance_state(
    *,
    power_mode: str | None = None,
    cpu_range_mhz: tuple[int, int] | None = None,
    gpu_range_mhz: tuple[int, int] | None = None,
    emc_range_mhz: tuple[int, int] | None = None,
) -> PerformanceState:
    """Build a state whose lock flags come only from min/max equality."""

    def locked(value: tuple[int, int] | None) -> bool | None:
        return None if value is None else value[0] == value[1]

    return PerformanceState(
        power_mode=power_mode,
        cpu_min_mhz=cpu_range_mhz[0] if cpu_range_mhz else None,
        cpu_max_mhz=cpu_range_mhz[1] if cpu_range_mhz else None,
        gpu_min_mhz=gpu_range_mhz[0] if gpu_range_mhz else None,
        gpu_max_mhz=gpu_range_mhz[1] if gpu_range_mhz else None,
        emc_min_mhz=emc_range_mhz[0] if emc_range_mhz else None,
        emc_max_mhz=emc_range_mhz[1] if emc_range_mhz else None,
        cpu_locked=locked(cpu_range_mhz),
        gpu_locked=locked(gpu_range_mhz),
        emc_locked=locked(emc_range_mhz),
    )


class JetsonPerformanceStateReader:
    """Observe nvpmodel and sysfs/debugfs policy limits without changing them."""

    def __init__(
        self,
        *,
        root: Path = Path("/"),
        command_runner: CommandRunner | None = None,
    ) -> None:
        self._root = root
        self._command_runner = command_runner or self._run_command

    def read(self) -> PerformanceState:
        mode = self._power_mode()
        cpu = self._aggregate_cpu_range()
        gpu = self._first_range(
            (
                "sys/class/devfreq/*gpu*/min_freq",
                "sys/devices/platform/*gpu*/devfreq/*/min_freq",
                "sys/devices/platform/*/*gpu*/devfreq/*/min_freq",
            ),
            maximum_name="max_freq",
            divisor=1_000_000,
        )
        emc = self._first_range(
            (
                "sys/kernel/debug/bpmp/debug/clk/emc/min_rate",
                "sys/class/devfreq/*memory-controller*/min_freq",
                "sys/class/devfreq/*emc*/min_freq",
            ),
            maximum_name=None,
            divisor=1_000_000,
        )
        if emc is None:
            emc = self._jetson_clocks_emc_range()
        return normalize_performance_state(
            power_mode=mode,
            cpu_range_mhz=cpu,
            gpu_range_mhz=gpu,
            emc_range_mhz=emc,
        )

    def _power_mode(self) -> str | None:
        output = self._command_runner(("nvpmodel", "-q"))
        if not output:
            return None
        match = _POWER_MODE_RE.search(output)
        return match.group(1).strip() if match else None

    def _aggregate_cpu_range(self) -> tuple[int, int] | None:
        minima = self._read_many(
            "sys/devices/system/cpu/cpu*/cpufreq/scaling_min_freq", divisor=1_000
        )
        maxima = self._read_many(
            "sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq", divisor=1_000
        )
        if not minima or not maxima:
            return None
        return min(minima), max(maxima)

    def _jetson_clocks_emc_range(self) -> tuple[int, int] | None:
        """Read EMC policy limits exposed only by ``jetson_clocks --show``.

        Some JetPack releases do not expose a readable EMC min/max pair in
        sysfs/debugfs. ``CurrentFreq`` is deliberately ignored: it is live
        telemetry, not operating-policy compatibility state.
        """
        output = self._command_runner(("jetson_clocks", "--show"))
        if not output:
            return None
        match = _JETSON_CLOCKS_EMC_RE.search(output)
        if match is None:
            return None
        minimum = round(int(match.group(1)) / 1_000_000)
        maximum = round(int(match.group(2)) / 1_000_000)
        if minimum <= 0 or maximum <= 0 or minimum > maximum:
            return None
        return minimum, maximum

    def _first_range(
        self,
        minimum_patterns: Iterable[str],
        *,
        maximum_name: str | None,
        divisor: int,
    ) -> tuple[int, int] | None:
        for pattern in minimum_patterns:
            for minimum_path in sorted(self._root.glob(pattern)):
                if maximum_name is None:
                    maximum_path = minimum_path.with_name(
                        "max_rate" if minimum_path.name == "min_rate" else "max_freq"
                    )
                else:
                    maximum_path = minimum_path.with_name(maximum_name)
                minimum = self._read_frequency(minimum_path, divisor)
                maximum = self._read_frequency(maximum_path, divisor)
                if minimum is not None and maximum is not None:
                    return minimum, maximum
        return None

    def _read_many(self, pattern: str, *, divisor: int) -> tuple[int, ...]:
        values = (self._read_frequency(path, divisor) for path in sorted(self._root.glob(pattern)))
        return tuple(value for value in values if value is not None)

    @staticmethod
    def _read_frequency(path: Path, divisor: int) -> int | None:
        try:
            raw = int(path.read_text(encoding="ascii").strip())
        except (OSError, ValueError):
            return None
        value = round(raw / divisor)
        return value if value > 0 else None

    @staticmethod
    def _run_command(command: tuple[str, ...]) -> str | None:
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=2.0,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.warning("Jetson policy query %r failed: %s", command, exc)
            return None
        if completed.returncode != 0:
            logger.warning(
                "Jetson policy query %r failed with exit code %d",
                command,
                completed.returncode,
            )
            return None
        return completed.stdout


__all__ = [
    "JetsonPerformanceStateReader",
    "normalize_performance_state",
]
