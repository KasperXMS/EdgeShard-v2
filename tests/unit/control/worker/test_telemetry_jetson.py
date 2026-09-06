"""Jetson telemetry backend tests (Phase 1 spec §23-24).

The long-lived tegrastats reader is exercised with a real subprocess (a
small Python script printing tegrastats-shaped lines), so no Jetson
hardware is needed (spec §50).
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

from edgeshard.cluster.state import DeviceAvailability
from edgeshard.control.worker.discovery.jetson import SYSTEM_MEMORY_POOL_ID
from edgeshard.control.worker.identity import (
    derive_cpu_device_id,
    derive_jetson_gpu_device_id,
)
from edgeshard.control.worker.telemetry.jetson import (
    JetsonTelemetryBackend,
    TegrastatsParser,
    TegrastatsProcess,
)

WORKER_ID = str(uuid.uuid4())

ORIN_R35_LINE = (
    "10-25-2024 12:00:00 RAM 3214/30536MB (lfb 4x4MB) SWAP 0/15268MB "
    "(lfb 15268MB) CPU [12%@1420,8%@1420,5%@1420,7%@1420] EMD1 0% "
    "GR3D_FREQ 34% PLL@45.5C CPU@46.5C PMIC@100C Tboard@42C "
    "soc_thermal@45.2C VDD_GPU_SOC 5.2W VDD_CPU_CV 3.1W VIN_SYS_5V0 12.0W"
)

XAVIER_R32_LINE = (
    "RAM 4567/15467MB (lfb 9x4MB) SWAP 0/7733MB CPU [5%@1420,3%@1420] "
    "GR3D_FREQ 12% PLL@38.5C AO@40C GPU@39C BCPU@41.5C MCPU@42.5C "
    "thermal@40.1C VDD_IN 4.5W VDD_CPU 1.2W VDD_GPU 0.9W VDD_SOC 1.1W"
)

# JetPack 6 (L4T r36) on Orin: lower-case thermal zones, milliwatt rails
# with current/average pairs, MHz suffixes on CPU clocks.
ORIN_R36_LINE = (
    "12-01-2025 09:15:00 RAM 3036/70612MB (lfb 15x4MB) SWAP 0/35306MB "
    "(lfb 35306MB) CPU [2%@1728MHz,0%@1728MHz,1%@1728MHz,0%@1728MHz] "
    "EMC_FREQ 0% GR3D_FREQ 7% pll@48C cpu@48.5C PMIC@100C gpu@45C "
    "AO@44C thermal@48.25C VDD_CPU_GPU_CV 15123mW/4321mW "
    "VDD_SOC 1234mW/1100mW VDD_IN 18792mW/5552mW"
)

# JetPack 5 (L4T r35) on Xavier NX / AGX: mW pairs and a dedicated
# VDD_GPU_SOC rail taking precedence over the whole-module VDD_IN.
XAVIER_R35_MW_LINE = (
    "RAM 2345/30536MB (lfb 4x4MB) CPU [12%@1728MHz,5%@1728MHz] "
    "GR3D_FREQ 34% PLL@20C CPU@22.5C PMIC@100C GPU@21.5C AO@25C "
    "thermal@22.25C VDD_GPU_SOC 5210mW/5183mW VDD_CPU_CORE 2464mW/2464mW "
    "VDD_IN 12345mW/12000mW"
)

MB = 1024 * 1024


def test_parser_maps_orin_r35_line() -> None:
    sample = TegrastatsParser().parse(ORIN_R35_LINE)
    assert sample is not None
    assert sample.gpu_utilization == 34.0
    # Orin r35 exposes no dedicated GPU@ thermal zone: None, not failure.
    assert sample.gpu_temperature_c is None
    assert sample.cpu_temperature_c == 46.5
    assert sample.gpu_power_w == 5.2
    assert sample.ram_used_bytes == 3214 * MB
    assert sample.ram_total_bytes == 30536 * MB


def test_parser_maps_xavier_r32_line() -> None:
    sample = TegrastatsParser().parse(XAVIER_R32_LINE)
    assert sample is not None
    assert sample.gpu_utilization == 12.0
    assert sample.gpu_temperature_c == 39.0
    # r32 has no plain CPU@ zone; the hottest CPU cluster stands in.
    assert sample.cpu_temperature_c == 42.5
    assert sample.gpu_power_w == 0.9  # VDD_GPU on r32
    assert sample.ram_total_bytes == 15467 * MB


def test_parser_maps_jetpack6_r36_lowercase_zones_and_mw_pairs() -> None:
    """JetPack 6: ``gpu@``/``cpu@`` lower-case zones, mW current/average pairs."""
    sample = TegrastatsParser().parse(ORIN_R36_LINE)
    assert sample is not None
    assert sample.gpu_utilization == 7.0
    assert sample.gpu_temperature_c == 45.0  # lower-case gpu@ zone
    assert sample.cpu_temperature_c == 48.5  # lower-case cpu@ zone
    # VDD_CPU_GPU_CV 15123mW/4321mW: mW converted, *current* (first) value.
    assert sample.gpu_power_w == 15.123
    assert sample.ram_used_bytes == 3036 * MB
    assert sample.ram_total_bytes == 70612 * MB


def test_parser_maps_jetpack5_r35_mw_pair_and_prefers_gpu_rail() -> None:
    sample = TegrastatsParser().parse(XAVIER_R35_MW_LINE)
    assert sample is not None
    assert sample.gpu_temperature_c == 21.5
    assert sample.cpu_temperature_c == 22.5
    # VDD_GPU_SOC outranks both VDD_IN and any later rail; 5210mW -> 5.21W.
    assert sample.gpu_power_w == 5.21


def test_parser_never_maps_vdd_in_to_gpu_power() -> None:
    """VDD_IN is whole-module input power, not a GPU rail (spec §23)."""
    sample = TegrastatsParser().parse(
        "RAM 1/2MB GR3D_FREQ 0% VDD_IN 18792mW/5552mW VDD_SOC 1234mW"
    )
    assert sample is not None
    assert sample.gpu_power_w is None


def test_parser_gpu_power_mw_without_average() -> None:
    sample = TegrastatsParser().parse("RAM 1/2MB VDD_GPU 310mW")
    assert sample is not None
    assert sample.gpu_power_w == 0.31


def test_parser_gpu_power_watts_without_average() -> None:
    sample = TegrastatsParser().parse("RAM 1/2MB VDD_GPU 5.2W")
    assert sample is not None
    assert sample.gpu_power_w == 5.2


def test_parser_tolerates_missing_gr3d() -> None:
    sample = TegrastatsParser().parse("RAM 100/200MB (lfb 2x4MB) CPU [1%@1420]")
    assert sample is not None
    assert sample.gpu_utilization is None
    assert sample.ram_used_bytes == 100 * MB


def test_parser_rejects_non_tegrastats_lines() -> None:
    assert TegrastatsParser().parse("") is None
    assert TegrastatsParser().parse("some stderr noise") is None
    assert TegrastatsParser().parse("Error: NVML shared library not found") is None


def test_parser_handles_gr3d_with_clock_suffix() -> None:
    sample = TegrastatsParser().parse("GR3D_FREQ 99%@306 RAM 1/2MB")
    assert sample is not None
    assert sample.gpu_utilization == 99.0


def _tegrastats_script(tmp_path: Path) -> Path:
    script = tmp_path / "fake_tegrastats.py"
    script.write_text(
        "import time\n"
        f"line = {ORIN_R35_LINE!r}\n"
        "while True:\n"
        "    print(line, flush=True)\n"
        "    time.sleep(0.05)\n",
        encoding="utf-8",
    )
    return script


async def test_process_reads_long_lived_stream(tmp_path: Path) -> None:
    process = TegrastatsProcess(command=(sys.executable, str(_tegrastats_script(tmp_path))))
    await process.start()
    try:
        sample = await process.wait_first(5.0)
        assert sample is not None
        assert sample.gpu_utilization == 34.0
        assert process.latest is not None
    finally:
        await process.stop()
    assert process.latest is None  # cache cleared on stop


async def test_process_stop_terminates_subprocess(tmp_path: Path) -> None:
    script = _tegrastats_script(tmp_path)
    process = TegrastatsProcess(command=(sys.executable, str(script)))
    await process.start()
    await process.wait_first(5.0)
    await process.stop()
    assert process.latest is None
    await process.stop()  # idempotent


async def test_process_missing_binary_is_non_fatal() -> None:
    process = TegrastatsProcess(command=("definitely-not-tegrastats",))
    await process.start()  # warns, does not raise (spec §47)
    assert await process.wait_first(0.1) is None
    await process.stop()


async def test_process_that_exits_without_samples_unblocks() -> None:
    process = TegrastatsProcess(command=(sys.executable, "-c", "pass"))
    await process.start()
    assert await process.wait_first(5.0) is None
    await process.stop()


def _backend_command(tmp_path: Path) -> tuple[str, ...]:
    return (sys.executable, str(_tegrastats_script(tmp_path)))


async def test_backend_maps_sample_into_state_fragment(tmp_path: Path) -> None:
    backend = JetsonTelemetryBackend(
        WORKER_ID, tegrastats=TegrastatsProcess(command=_backend_command(tmp_path))
    )
    fragment = await backend.sample()
    await backend.close()

    cpu_state, gpu_state = fragment.device_states
    assert cpu_state.device_id == derive_cpu_device_id(WORKER_ID)
    assert cpu_state.availability is DeviceAvailability.AVAILABLE
    assert cpu_state.utilization is not None
    assert 0.0 <= cpu_state.utilization <= 100.0
    assert cpu_state.temperature_c == 46.5

    assert gpu_state.device_id == derive_jetson_gpu_device_id(WORKER_ID)
    assert gpu_state.availability is DeviceAvailability.AVAILABLE
    assert gpu_state.utilization == 34.0
    assert gpu_state.temperature_c is None  # no GPU@ zone on r35
    assert gpu_state.power_w == 5.2

    (memory_state,) = fragment.memory_states
    assert memory_state.memory_pool_id == SYSTEM_MEMORY_POOL_ID
    assert memory_state.available_bytes is not None
    assert memory_state.available_bytes >= 0


async def test_backend_without_tegrastats_reports_unknown_gpu() -> None:
    backend = JetsonTelemetryBackend(
        WORKER_ID,
        tegrastats=TegrastatsProcess(command=("definitely-not-tegrastats",)),
        first_sample_timeout_s=0.1,
    )
    fragment = await backend.sample()
    await backend.close()

    cpu_state, gpu_state = fragment.device_states
    assert cpu_state.availability is DeviceAvailability.AVAILABLE  # psutil flows
    assert gpu_state.availability is DeviceAvailability.UNKNOWN
    assert gpu_state.utilization is None
    assert gpu_state.temperature_c is None
    assert gpu_state.power_w is None

    (memory_state,) = fragment.memory_states
    assert memory_state.memory_pool_id == SYSTEM_MEMORY_POOL_ID
    assert memory_state.available_bytes is not None


async def test_repeated_samples_reuse_the_single_process(tmp_path: Path) -> None:
    """Spec §24: one long-lived reader serves every heartbeat sample."""
    backend = JetsonTelemetryBackend(
        WORKER_ID, tegrastats=TegrastatsProcess(command=_backend_command(tmp_path))
    )
    try:
        first = await backend.sample()
        second = await backend.sample()
        _, first_gpu = first.device_states
        _, second_gpu = second.device_states
        assert first_gpu.utilization == second_gpu.utilization == 34.0
    finally:
        await backend.close()
