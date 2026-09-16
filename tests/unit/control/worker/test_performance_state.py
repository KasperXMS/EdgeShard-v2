from __future__ import annotations

from pathlib import Path

import pytest

from edgeshard.control.worker.performance_state import (
    JetsonPerformanceStateReader,
    normalize_performance_state,
)
from edgeshard.profiling.domain.environment import PerformanceState


def test_dynamic_and_locked_state_normalization() -> None:
    dynamic = normalize_performance_state(
        power_mode="MODE_30W",
        cpu_range_mhz=(729, 1728),
        gpu_range_mhz=(306, 612),
        emc_range_mhz=(204, 3199),
    )
    assert dynamic == PerformanceState(
        power_mode="MODE_30W",
        cpu_min_mhz=729,
        cpu_max_mhz=1728,
        gpu_min_mhz=306,
        gpu_max_mhz=612,
        emc_min_mhz=204,
        emc_max_mhz=3199,
        cpu_locked=False,
        gpu_locked=False,
        emc_locked=False,
    )
    locked = normalize_performance_state(
        power_mode="MODE_30W",
        cpu_range_mhz=(1728, 1728),
        gpu_range_mhz=(612, 612),
        emc_range_mhz=(3199, 3199),
    )
    assert locked.cpu_locked is True
    assert locked.gpu_locked is True
    assert locked.emc_locked is True


def _write(root: Path, relative: str, value: int) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(value), encoding="ascii")


def test_reader_observes_nvpmodel_and_policy_ranges(tmp_path: Path) -> None:
    for cpu in range(2):
        base = f"sys/devices/system/cpu/cpu{cpu}/cpufreq"
        _write(tmp_path, f"{base}/scaling_min_freq", 729_000)
        _write(tmp_path, f"{base}/scaling_max_freq", 1_728_000)
    _write(tmp_path, "sys/class/devfreq/17000000.gpu/min_freq", 306_000_000)
    _write(tmp_path, "sys/class/devfreq/17000000.gpu/max_freq", 612_000_000)
    _write(
        tmp_path,
        "sys/kernel/debug/bpmp/debug/clk/emc/min_rate",
        204_000_000,
    )
    _write(
        tmp_path,
        "sys/kernel/debug/bpmp/debug/clk/emc/max_rate",
        3_199_000_000,
    )
    reader = JetsonPerformanceStateReader(
        root=tmp_path,
        command_runner=lambda _command: "NV Power Mode: MODE_30W\n8\n",
    )
    state = reader.read()
    assert state.power_mode == "MODE_30W"
    assert (state.cpu_min_mhz, state.cpu_max_mhz, state.cpu_locked) == (
        729,
        1728,
        False,
    )
    assert (state.gpu_min_mhz, state.gpu_max_mhz, state.gpu_locked) == (
        306,
        612,
        False,
    )
    assert (state.emc_min_mhz, state.emc_max_mhz, state.emc_locked) == (
        204,
        3199,
        False,
    )


def test_reader_derives_locked_gpu_from_nested_devfreq_policy(tmp_path: Path) -> None:
    gpu = "sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu"
    _write(tmp_path, f"{gpu}/min_freq", 612_000_000)
    _write(tmp_path, f"{gpu}/max_freq", 612_000_000)

    state = JetsonPerformanceStateReader(
        root=tmp_path,
        command_runner=lambda command: (
            "NV Power Mode: MODE_30W\n8\n"
            if command == ("nvpmodel", "-q")
            else pytest.fail(f"unexpected privileged query: {command!r}")
        ),
    ).read()

    assert (state.gpu_min_mhz, state.gpu_max_mhz, state.gpu_locked) == (
        612,
        612,
        True,
    )


def test_inaccessible_emc_is_supported_unknown_without_fallback_or_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    emc = "sys/kernel/debug/bpmp/debug/clk/emc"
    _write(tmp_path, f"{emc}/min_rate", 204_000_000)
    _write(tmp_path, f"{emc}/max_rate", 3_199_000_000)
    original_read_text = Path.read_text

    def read_text(path: Path, *args: object, **kwargs: object) -> str:
        if "clk/emc" in path.as_posix():
            raise PermissionError("permission denied")
        return original_read_text(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "read_text", read_text)
    commands: list[tuple[str, ...]] = []

    def run(command: tuple[str, ...]) -> str:
        commands.append(command)
        return "NV Power Mode: MODE_30W\n8\n"

    state = JetsonPerformanceStateReader(
        root=tmp_path, command_runner=run
    ).read()

    assert (state.emc_min_mhz, state.emc_max_mhz, state.emc_locked) == (
        None,
        None,
        None,
    )
    assert commands == [("nvpmodel", "-q")]
    assert not caplog.records
