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
