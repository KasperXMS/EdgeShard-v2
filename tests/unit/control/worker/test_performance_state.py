from __future__ import annotations

from pathlib import Path

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


def _reader_for_jetson_clocks(
    tmp_path: Path, jetson_clocks_output: str
) -> JetsonPerformanceStateReader:
    def run(command: tuple[str, ...]) -> str:
        if command == ("nvpmodel", "-q"):
            return "NV Power Mode: MODE_30W\n8\n"
        assert command == ("jetson_clocks", "--show")
        return jetson_clocks_output

    return JetsonPerformanceStateReader(root=tmp_path, command_runner=run)


def test_reader_parses_dynamic_emc_range_from_jetson_clocks(tmp_path: Path) -> None:
    state = _reader_for_jetson_clocks(
        tmp_path,
        "GPU MinFreq=306000000 MaxFreq=612000000 CurrentFreq=408000000\n"
        "EMC MinFreq=204000000 MaxFreq=3199000000 CurrentFreq=2133000000\n",
    ).read()

    assert (state.emc_min_mhz, state.emc_max_mhz, state.emc_locked) == (
        204,
        3199,
        False,
    )


def test_reader_parses_locked_emc_range_from_jetson_clocks(tmp_path: Path) -> None:
    state = _reader_for_jetson_clocks(
        tmp_path,
        "EMC MinFreq=3199000000 MaxFreq=3199000000 CurrentFreq=3199000000\n",
    ).read()

    assert (state.emc_min_mhz, state.emc_max_mhz, state.emc_locked) == (
        3199,
        3199,
        True,
    )


def test_reader_leaves_emc_unknown_when_jetson_clocks_has_no_range(
    tmp_path: Path,
) -> None:
    state = _reader_for_jetson_clocks(tmp_path, "GPU MinFreq=306000000 MaxFreq=612000000\n").read()

    assert (state.emc_min_mhz, state.emc_max_mhz, state.emc_locked) == (
        None,
        None,
        None,
    )
