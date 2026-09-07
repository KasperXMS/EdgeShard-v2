"""P2F iperf3 runner tests (spec §34-§35, P2F DoD "no runaway iperf servers").

Everything runs against fake async subprocess factories. Pinned: JSON-only
parsing with iperf3's documented section conventions (TCP receiver-side
throughput, sender-side retransmits, UDP aggregate), typed failures for
every documented case (error payload, missing binary, timeout, undecodable
output with raw stderr preserved), and the guaranteed server cleanup —
one-off servers plus terminate→kill escalation in a ``finally``.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from edgeshard.profiling.domain.experiment import ProfilingErrorCategory
from edgeshard.profiling.domain.network import NetworkDirection, NetworkTransport
from edgeshard.profiling.errors import ProfilingError
from edgeshard.profiling.network.iperf import (
    DEFAULT_IPERF3_DURATION_S,
    DEFAULT_IPERF3_PORT,
    Iperf3Observation,
    Iperf3Probe,
    Iperf3Runner,
    iperf3_client_command,
    iperf3_server_command,
    parse_iperf3_json,
)
from edgeshard.profiling.network.processes import terminate_process

TCP_JSON = {
    "start": {"connecting_to": "10.0.0.2"},
    "end": {
        "sum_sent": {"bits_per_second": 9.41e9, "retransmits": 12, "bytes": 5.9e9},
        "sum_received": {"bits_per_second": 9.30e9, "bytes": 5.8e9},
    },
}
UDP_JSON = {
    "end": {
        "sum": {
            "bits_per_second": 8.8e8,
            "bytes": 5.5e8,
            "jitter_ms": 0.043,
            "lost_packets": 17,
            "packets": 4000,
        }
    }
}


class FakeProcess:
    """Stand-in for ``asyncio.subprocess.Process``."""

    def __init__(
        self,
        stdout: bytes = b"",
        stderr: bytes = b"",
        returncode: int = 0,
        hang: bool = False,
        ignore_terminate: bool = False,
    ) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self._exit_code = returncode
        self._hang = hang
        self._ignore_terminate = ignore_terminate
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False

    async def communicate(self):
        if self._hang:
            await asyncio.sleep(3600)
        self.returncode = self._exit_code
        return self._stdout, self._stderr

    def terminate(self) -> None:
        self.terminated = True
        if not self._ignore_terminate:
            self.returncode = self._exit_code

    def kill(self) -> None:
        self.killed = True
        self.returncode = self._exit_code

    async def wait(self) -> int:
        if self._ignore_terminate and not self.killed:
            await asyncio.sleep(3600)  # refuses to die on SIGTERM
        if self.returncode is None:
            self.returncode = self._exit_code
        return self.returncode


class FakeFactory:
    """Hands out queued processes and records every command."""

    def __init__(self, *processes: FakeProcess) -> None:
        self._processes = list(processes)
        self.commands: list[tuple[str, ...]] = []

    async def __call__(self, command):
        self.commands.append(tuple(command))
        if not self._processes:
            raise AssertionError("unexpected extra spawn")
        if len(self._processes) == 1:
            return self._processes[0]
        return self._processes.pop(0)


class RaisingFactory:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def __call__(self, command):
        raise self._exc


def _runner(factory, **kwargs) -> Iperf3Runner:
    return Iperf3Runner(process_factory=factory, server_startup_s=0.0, **kwargs)


class TestIperf3Probe:
    def test_defaults(self) -> None:
        probe = Iperf3Probe(target="10.0.0.2")
        assert probe.transport is NetworkTransport.TCP
        assert probe.direction is NetworkDirection.FORWARD
        assert probe.duration_s == DEFAULT_IPERF3_DURATION_S
        assert probe.port == DEFAULT_IPERF3_PORT
        assert probe.payload_bytes is None

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="target"):
            Iperf3Probe(target="")
        with pytest.raises(ValueError, match="duration_s"):
            Iperf3Probe(target="t", duration_s=0.0)
        with pytest.raises(ValueError, match="payload_bytes"):
            Iperf3Probe(target="t", payload_bytes=0)
        with pytest.raises(ValueError, match="port"):
            Iperf3Probe(target="t", port=0)
        with pytest.raises(ValueError, match="port"):
            Iperf3Probe(target="t", port=65536)


class TestCommandBuilders:
    def test_server_is_one_off(self) -> None:
        """``-1``: the server exits after a single test (first cleanup layer)."""
        assert iperf3_server_command() == ("iperf3", "-s", "-1", "-p", "5201")

    def test_tcp_forward_client(self) -> None:
        command = iperf3_client_command(Iperf3Probe(target="10.0.0.2", duration_s=5.0))
        assert command == (
            "iperf3", "-c", "10.0.0.2", "-p", "5201", "--json", "-t", "5",
        )

    def test_json_flag_is_always_present(self) -> None:
        """§34: JSON output only, never human-readable text."""
        for probe in (
            Iperf3Probe(target="t"),
            Iperf3Probe(target="t", transport=NetworkTransport.UDP),
            Iperf3Probe(target="t", direction=NetworkDirection.REVERSE),
        ):
            assert "--json" in iperf3_client_command(probe)

    def test_udp_flags(self) -> None:
        command = iperf3_client_command(
            Iperf3Probe(target="t", transport=NetworkTransport.UDP, duration_s=2.5)
        )
        assert "-u" in command
        assert command[command.index("-b") + 1] == "0"  # best-effort baseline
        assert "-t" in command and command[command.index("-t") + 1] == "2.5"

    def test_reverse_flag(self) -> None:
        command = iperf3_client_command(
            Iperf3Probe(target="t", direction=NetworkDirection.REVERSE)
        )
        assert command[-1] == "-R"

    def test_payload_size_flag(self) -> None:
        """§36: workload-derived payload bytes map to the iperf3 buffer length."""
        command = iperf3_client_command(Iperf3Probe(target="t", payload_bytes=4_194_304))
        assert command[command.index("-l") + 1] == "4194304"

    def test_custom_binary_and_port(self) -> None:
        command = iperf3_client_command(
            Iperf3Probe(target="t", port=6000), binary="/usr/bin/iperf3"
        )
        assert command[0] == "/usr/bin/iperf3"
        assert command[command.index("-p") + 1] == "6000"

    def test_server_port_validation(self) -> None:
        with pytest.raises(ValueError, match="port"):
            iperf3_server_command(port=0)

    def test_client_and_server_bind_explicit_interfaces(self) -> None:
        server = iperf3_server_command(bind_address="100.64.0.20")
        client = iperf3_client_command(
            Iperf3Probe(target="100.64.0.20", bind_address="100.64.0.10")
        )
        assert server[-2:] == ("-B", "100.64.0.20")
        assert client[-2:] == ("-B", "100.64.0.10")


class TestParseIperf3Json:
    def test_tcp_prefers_receiver_side_throughput(self) -> None:
        observation = parse_iperf3_json(TCP_JSON)
        assert observation.bits_per_second == pytest.approx(9.30e9)
        assert observation.retransmits == 12  # sender-side fact

    def test_tcp_falls_back_to_sender_summary(self) -> None:
        payload = {"end": {"sum_sent": {"bits_per_second": 9.41e9, "retransmits": 0}}}
        observation = parse_iperf3_json(payload)
        assert observation.bits_per_second == pytest.approx(9.41e9)
        assert observation.retransmits == 0

    def test_udp_aggregate(self) -> None:
        observation = parse_iperf3_json(UDP_JSON)
        assert observation.bits_per_second == pytest.approx(8.8e8)
        assert observation.retransmits is None  # UDP has no retransmits

    def test_error_payload_fails_typed(self) -> None:
        payload = {"error": "unable to connect to server: Connection refused"}
        with pytest.raises(ProfilingError) as excinfo:
            parse_iperf3_json(payload)
        assert excinfo.value.category is ProfilingErrorCategory.NETWORK_UNREACHABLE
        assert "Connection refused" in str(excinfo.value)

    def test_missing_end_section_fails_typed(self) -> None:
        with pytest.raises(ProfilingError) as excinfo:
            parse_iperf3_json({"start": {}})
        assert excinfo.value.category is ProfilingErrorCategory.BENCHMARK_FAILED
        assert "'end'" in str(excinfo.value)

    def test_missing_throughput_fails_typed(self) -> None:
        """No value is ever substituted for a missing observation (§42)."""
        for payload in (
            {"end": {"sum_received": {"bytes": 100}}},
            {"end": {"sum_received": {"bits_per_second": "fast"}}},
            {"end": {"sum_received": {"bits_per_second": True}}},
            {"end": {}},
        ):
            with pytest.raises(ProfilingError) as excinfo:
                parse_iperf3_json(payload)
            assert excinfo.value.category is ProfilingErrorCategory.BENCHMARK_FAILED
            assert "bits_per_second" in str(excinfo.value)

    def test_non_finite_or_negative_throughput_fails_typed(self) -> None:
        for value in (float("nan"), float("inf"), -1.0):
            with pytest.raises(ProfilingError) as excinfo:
                parse_iperf3_json({"end": {"sum_received": {"bits_per_second": value}}})
            assert excinfo.value.category is ProfilingErrorCategory.BENCHMARK_FAILED

    def test_float_retransmits_are_not_invented_into_ints(self) -> None:
        payload = {"end": {"sum_sent": {"bits_per_second": 1.0, "retransmits": 1.5}}}
        assert parse_iperf3_json(payload).retransmits is None


class TestIperf3RunnerClient:
    async def test_success(self) -> None:
        factory = FakeFactory(FakeProcess(stdout=json.dumps(TCP_JSON).encode()))
        observation = await _runner(factory).run_client(
            Iperf3Probe(target="10.0.0.2", duration_s=2.0)
        )
        assert observation == Iperf3Observation(bits_per_second=9.30e9, retransmits=12)
        assert factory.commands == [
            ("iperf3", "-c", "10.0.0.2", "-p", "5201", "--json", "-t", "2")
        ]

    async def test_missing_binary_fails_typed(self) -> None:
        factory = RaisingFactory(FileNotFoundError(2, "No such file or directory", "iperf3"))
        with pytest.raises(ProfilingError) as excinfo:
            await _runner(factory).run_client(Iperf3Probe(target="10.0.0.2"))
        assert excinfo.value.category is ProfilingErrorCategory.IPERF_UNAVAILABLE
        assert "iperf3" in str(excinfo.value)

    async def test_connection_refused_error_payload_fails_typed(self) -> None:
        payload = {"error": "unable to connect to server: Connection refused"}
        factory = FakeFactory(
            FakeProcess(stdout=json.dumps(payload).encode(), returncode=1)
        )
        with pytest.raises(ProfilingError) as excinfo:
            await _runner(factory).run_client(Iperf3Probe(target="10.0.0.2"))
        assert excinfo.value.category is ProfilingErrorCategory.NETWORK_UNREACHABLE

    async def test_undecodable_output_preserves_raw_stderr(self) -> None:
        """§34 diagnostics rule: raw stderr survives in the typed failure."""
        factory = FakeFactory(
            FakeProcess(
                stdout=b"iperf3: interrupt - the server has terminated",
                stderr=b"iperf3: error - protocol mismatch\n",
                returncode=1,
            )
        )
        with pytest.raises(ProfilingError) as excinfo:
            await _runner(factory).run_client(Iperf3Probe(target="10.0.0.2"))
        assert excinfo.value.category is ProfilingErrorCategory.BENCHMARK_FAILED
        details = dict(excinfo.value.to_failure().details)
        assert details["stderr"] == "iperf3: error - protocol mismatch\n"
        assert details["target"] == "10.0.0.2"
        assert "exit code 1" in str(excinfo.value)

    async def test_non_object_json_fails_typed(self) -> None:
        factory = FakeFactory(FakeProcess(stdout=b"42", stderr=b""))
        with pytest.raises(ProfilingError) as excinfo:
            await _runner(factory).run_client(Iperf3Probe(target="10.0.0.2"))
        assert excinfo.value.category is ProfilingErrorCategory.BENCHMARK_FAILED
        assert "not an object" in str(excinfo.value)

    async def test_timeout_kills_and_fails_typed(self) -> None:
        process = FakeProcess(hang=True)
        factory = FakeFactory(process)
        runner = _runner(factory, timeout_grace_s=0.05)
        with pytest.raises(ProfilingError) as excinfo:
            await runner.run_client(Iperf3Probe(target="10.0.0.2", duration_s=0.05))
        assert excinfo.value.category is ProfilingErrorCategory.TIMEOUT
        assert process.terminated  # cleaned up, never left running


class TestIperf3RunnerServer:
    async def test_cancelled_startup_terminates_server(self) -> None:
        server = FakeProcess()
        runner = Iperf3Runner(
            process_factory=FakeFactory(server), server_startup_s=10.0
        )
        task = asyncio.create_task(runner.start_server())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert server.terminated

    async def test_measure_spawns_server_then_client_and_terminates(self) -> None:
        server = FakeProcess()
        client = FakeProcess(stdout=json.dumps(TCP_JSON).encode())
        factory = FakeFactory(server, client)
        observation = await _runner(factory).measure(Iperf3Probe(target="127.0.0.1"))
        assert observation.bits_per_second == pytest.approx(9.30e9)
        assert factory.commands[0] == ("iperf3", "-s", "-1", "-p", "5201")
        assert factory.commands[1][1] == "-c"
        assert server.terminated  # §34/P2F DoD: no runaway iperf servers

    async def test_server_terminated_even_when_client_fails(self) -> None:
        server = FakeProcess()
        client = FakeProcess(stdout=b"garbage", stderr=b"boom", returncode=1)
        factory = FakeFactory(server, client)
        with pytest.raises(ProfilingError):
            await _runner(factory).measure(Iperf3Probe(target="127.0.0.1"))
        assert server.terminated

    async def test_server_terminated_even_when_client_times_out(self) -> None:
        server = FakeProcess()
        client = FakeProcess(hang=True)
        factory = FakeFactory(server, client)
        runner = _runner(factory, timeout_grace_s=0.05)
        with pytest.raises(ProfilingError) as excinfo:
            await runner.measure(Iperf3Probe(target="127.0.0.1", duration_s=0.05))
        assert excinfo.value.category is ProfilingErrorCategory.TIMEOUT
        assert server.terminated
        assert client.terminated

    async def test_custom_port_is_shared_by_server_and_client(self) -> None:
        server = FakeProcess()
        client = FakeProcess(stdout=json.dumps(TCP_JSON).encode())
        factory = FakeFactory(server, client)
        await _runner(factory).measure(Iperf3Probe(target="127.0.0.1", port=6000))
        assert "6000" in factory.commands[0]
        assert "6000" in factory.commands[1]

    async def test_custom_binary(self) -> None:
        factory = FakeFactory(FakeProcess())
        runner = Iperf3Runner(
            process_factory=factory, binary="/opt/iperf3", server_startup_s=0.0
        )
        await runner.start_server()
        assert factory.commands[0][0] == "/opt/iperf3"


class TestIperf3RunnerConstruction:
    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="binary"):
            Iperf3Runner(binary="")
        with pytest.raises(ValueError, match="timeout_grace_s"):
            Iperf3Runner(timeout_grace_s=0.0)
        with pytest.raises(ValueError, match="server_startup_s"):
            Iperf3Runner(server_startup_s=-1.0)


class TestTerminateProcess:
    async def test_already_exited_process_is_untouched(self) -> None:
        process = FakeProcess()
        process.returncode = 0
        await terminate_process(process)
        assert not process.terminated and not process.killed

    async def test_graceful_termination(self) -> None:
        process = FakeProcess()
        await terminate_process(process)
        assert process.terminated and not process.killed

    async def test_stubborn_process_is_killed(self) -> None:
        """Escalation: SIGTERM ignored → SIGKILL, still reaped (§34)."""
        process = FakeProcess(ignore_terminate=True)
        await terminate_process(process, grace_s=0.05)
        assert process.terminated and process.killed
        assert process.returncode is not None  # reaped, no zombie
