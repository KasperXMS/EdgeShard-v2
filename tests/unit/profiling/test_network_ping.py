"""P2F system-ping RTT runner tests (spec §33, P2F DoD).

Everything runs against fake async subprocess factories — no host network
access. Pinned: platform-aware command construction, RTT line parsing
(including the Windows ``time<1ms`` resolution bound, never zero, and
localized zh-CN Windows replies via the ``TTL=`` fallback), typed failures
for total loss / missing binary / timeout, and the mandatory concurrency
bound across a fan-out.
"""

from __future__ import annotations

import asyncio

import pytest

from edgeshard.profiling.domain.experiment import ProfilingErrorCategory
from edgeshard.profiling.domain.measurement import summarize_samples
from edgeshard.profiling.errors import ProfilingError
from edgeshard.profiling.network.ping import (
    DEFAULT_PING_CONCURRENCY,
    DEFAULT_PING_PACKET_COUNT,
    PingObservation,
    PingRunner,
    parse_ping_output,
    ping_command,
)

LINUX_PING_OUTPUT = """\
PING 192.168.1.20 (192.168.1.20) 56(84) bytes of data.
64 bytes from 192.168.1.20: icmp_seq=1 ttl=64 time=0.451 ms
64 bytes from 192.168.1.20: icmp_seq=2 ttl=64 time=0.512 ms
64 bytes from 192.168.1.20: icmp_seq=3 ttl=64 time=1.207 ms

--- 192.168.1.20 ping statistics ---
5 packets transmitted, 3 received, 40% packet loss, time 4006ms
rtt min/avg/max/mdev = 0.451/0.723/1.207/0.347 ms
"""

WINDOWS_PING_OUTPUT = """\
Pinging 192.168.1.20 with 32 bytes of data:
Reply from 192.168.1.20: bytes=32 time<1ms TTL=128
Reply from 192.168.1.20: bytes=32 time=3ms TTL=128
Request timed out.
Reply from 192.168.1.20: bytes=32 time=2ms TTL=128

Ping statistics for 192.168.1.20:
    Packets: Sent = 4, Received = 3, Lost = 1 (25% loss),
Approximate round trip times in milli-seconds:
    Minimum = 0ms, Maximum = 3ms, Average = 1ms
"""

# Real zh-CN Windows output. The fullwidth commas in the statistics block
# are spelled <FW> and substituted with chr(0xFF0C) solely to keep ruff
# RUF001 quiet; the parsed string is byte-identical to the console output
# observed on the localized host. The reply lines — the ones that matter
# for the TTL fallback — need no substitution.
LOCALIZED_WINDOWS_PING_OUTPUT = """\
正在 Ping 127.0.0.1 具有 32 字节的数据:
来自 127.0.0.1 的回复: 字节=32 时间<1ms TTL=128
请求超时。
来自 127.0.0.1 的回复: 字节=32 时间=3ms TTL=128

127.0.0.1 的 Ping 统计信息:
    数据包: 已发送 = 3<FW>已接收 = 2<FW>丢失 = 1 (33% 丢失)<FW>
往返行程的估计时间(以毫秒为单位):
    最短 = 0ms<FW>最长 = 3ms<FW>平均 = 1ms
""".replace("<FW>", chr(0xFF0C))

TOTAL_LOSS_OUTPUT = """\
PING 10.9.9.9 (10.9.9.9) 56(84) bytes of data.

--- 10.9.9.9 ping statistics ---
5 packets transmitted, 0 received, 100% packet loss, time 4003ms
"""


class FakeProcess:
    """Stand-in for ``asyncio.subprocess.Process``."""

    def __init__(
        self,
        stdout: bytes = b"",
        stderr: bytes = b"",
        returncode: int = 0,
        delay: float = 0.0,
        hang: bool = False,
    ) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self._exit_code = returncode
        self._delay = delay
        self._hang = hang
        self.returncode: int | None = None  # running until communicate/wait finishes
        self.terminated = False
        self.killed = False

    async def communicate(self) -> tuple[bytes, bytes]:
        if self._hang:
            await asyncio.sleep(3600)
        if self._delay:
            await asyncio.sleep(self._delay)
        self.returncode = self._exit_code
        return self._stdout, self._stderr

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = self._exit_code

    def kill(self) -> None:
        self.killed = True
        self.returncode = self._exit_code

    async def wait(self) -> int:
        if self.returncode is None:
            self.returncode = self._exit_code
        return self.returncode


def _factory(processes: list[FakeProcess] | FakeProcess):
    """Fake process factory recording every command."""
    commands: list[tuple[str, ...]] = []
    queue = list(processes) if isinstance(processes, list) else None

    async def factory(command):
        commands.append(tuple(command))
        if queue is not None:
            return queue[len(commands) - 1]
        assert not isinstance(processes, list)
        return processes

    return factory, commands


class TestPingCommand:
    def test_linux_invocation(self) -> None:
        assert ping_command("10.0.0.1", platform="linux") == (
            "ping",
            "-c",
            str(DEFAULT_PING_PACKET_COUNT),
            "-W",
            "2",
            "10.0.0.1",
        )

    def test_windows_invocation_uses_millisecond_wait(self) -> None:
        assert ping_command("10.0.0.1", platform="win32") == (
            "ping",
            "-n",
            str(DEFAULT_PING_PACKET_COUNT),
            "-w",
            "2000",
            "10.0.0.1",
        )

    def test_knobs_flow_through(self) -> None:
        command = ping_command(
            "host.local", packet_count=5, packet_timeout_s=0.5, platform="linux"
        )
        assert command == ("ping", "-c", "5", "-W", "1", "host.local")  # ceil to whole seconds

    def test_explicit_source_address_is_bound_per_platform(self) -> None:
        linux = ping_command(
            "10.0.0.2", platform="linux", bind_address="10.0.0.1"
        )
        windows = ping_command(
            "10.0.0.2", platform="win32", bind_address="10.0.0.1"
        )
        assert linux[-3:] == ("-I", "10.0.0.1", "10.0.0.2")
        assert windows[-3:] == ("-S", "10.0.0.1", "10.0.0.2")

    def test_default_packet_count_within_spec_range(self) -> None:
        """§33 policy: 5-10 packets per pair."""
        assert 5 <= DEFAULT_PING_PACKET_COUNT <= 10

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            ping_command("")
        with pytest.raises(ValueError, match="packet_count"):
            ping_command("10.0.0.1", packet_count=0)
        with pytest.raises(ValueError, match="packet_timeout_s"):
            ping_command("10.0.0.1", packet_timeout_s=0.0)


class TestParsePingOutput:
    def test_linux_reply_lines(self) -> None:
        assert parse_ping_output(LINUX_PING_OUTPUT) == (0.451, 0.512, 1.207)

    def test_windows_sub_millisecond_bound_is_never_zero(self) -> None:
        """``time<1ms`` records the resolution bound 1.0 — not a fabricated 0 (§42)."""
        assert parse_ping_output(WINDOWS_PING_OUTPUT) == (1.0, 3.0, 2.0)

    def test_localized_windows_replies_use_ttl_fallback(self) -> None:
        """zh-CN Windows translates the ``time`` token (``时间<1ms``).

        Real output observed on a zh-CN Windows host; the locale-independent
        ``TTL=`` anchor must keep replies from reading as total loss, and the
        localized statistics block (``最短 = 0ms``, no TTL) must stay ignored.
        """
        assert parse_ping_output(LOCALIZED_WINDOWS_PING_OUTPUT) == (1.0, 3.0)

    def test_total_loss_yields_no_samples(self) -> None:
        assert parse_ping_output(TOTAL_LOSS_OUTPUT) == ()

    def test_summary_block_does_not_count_as_reply(self) -> None:
        output = "rtt min/avg/max/mdev = 0.451/0.723/1.207/0.347 ms\nMinimum = 0ms"
        assert parse_ping_output(output) == ()

    def test_empty_output(self) -> None:
        assert parse_ping_output("") == ()


class TestPingRunnerProbe:
    async def test_success_observation(self) -> None:
        process = FakeProcess(stdout=LINUX_PING_OUTPUT.encode())
        factory, commands = _factory(process)
        runner = PingRunner(process_factory=factory, platform="linux")
        observation = await runner.probe("192.168.1.20")
        assert commands == [("ping", "-c", "10", "-W", "2", "192.168.1.20")]
        assert observation.target == "192.168.1.20"
        assert observation.packets_sent == 10
        assert observation.packets_received == 3
        assert observation.samples_ms == (0.451, 0.512, 1.207)
        assert observation.summary.median == 0.512
        assert observation.summary.maximum == 1.207
        assert observation.summary.p95 is None  # fewer than 5 samples

    async def test_packet_count_override(self) -> None:
        process = FakeProcess(stdout=WINDOWS_PING_OUTPUT.encode())
        factory, commands = _factory(process)
        runner = PingRunner(process_factory=factory, platform="win32")
        observation = await runner.probe("192.168.1.20", packet_count=4)
        assert observation.packets_sent == 4
        assert observation.packets_received == 3
        assert commands[0] == ("ping", "-n", "4", "-w", "2000", "192.168.1.20")

    async def test_p95_present_with_enough_samples(self) -> None:
        lines = "\n".join(
            f"64 bytes from 10.0.0.9: icmp_seq={i} ttl=64 time={i / 10:.3f} ms"
            for i in range(1, 8)
        )
        factory, _ = _factory(FakeProcess(stdout=lines.encode()))
        observation = await PingRunner(process_factory=factory).probe("10.0.0.9", packet_count=7)
        assert observation.summary.p95 == 0.7  # nearest rank over 7 samples

    async def test_total_loss_fails_typed_with_stderr_preserved(self) -> None:
        factory, _ = _factory(
            FakeProcess(
                stdout=TOTAL_LOSS_OUTPUT.encode(),
                stderr=b"",
                returncode=1,
            )
        )
        with pytest.raises(ProfilingError) as excinfo:
            await PingRunner(process_factory=factory).probe("10.9.9.9", packet_count=5)
        assert excinfo.value.category is ProfilingErrorCategory.NETWORK_UNREACHABLE
        assert "10.9.9.9" in str(excinfo.value)
        details = dict(excinfo.value.to_failure().details)
        assert details["packets_sent"] == 5

    async def test_missing_binary_fails_typed(self) -> None:
        async def factory(command):
            raise FileNotFoundError(2, "No such file or directory", command[0])

        with pytest.raises(ProfilingError) as excinfo:
            await PingRunner(process_factory=factory).probe("10.0.0.1")
        assert excinfo.value.category is ProfilingErrorCategory.NETWORK_UNREACHABLE
        assert "ping executable" in str(excinfo.value)

    async def test_timeout_fails_typed_after_killing_the_process(self) -> None:
        process = FakeProcess(hang=True)
        factory, _ = _factory(process)
        runner = PingRunner(process_factory=factory, command_timeout_s=0.05)
        with pytest.raises(ProfilingError) as excinfo:
            await runner.probe("10.0.0.1")
        assert excinfo.value.category is ProfilingErrorCategory.TIMEOUT
        assert process.terminated  # cleaned up, never left running

    async def test_invalid_packet_count_override_rejected(self) -> None:
        factory, _ = _factory(FakeProcess())
        with pytest.raises(ValueError, match="packet_count"):
            await PingRunner(process_factory=factory).probe("10.0.0.1", packet_count=0)


class TestPingRunnerConstruction:
    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="packet_count"):
            PingRunner(packet_count=0)
        with pytest.raises(ValueError, match="packet_timeout_s"):
            PingRunner(packet_timeout_s=0.0)
        with pytest.raises(ValueError, match="concurrency"):
            PingRunner(concurrency=0)
        with pytest.raises(ValueError, match="command_timeout_s"):
            PingRunner(command_timeout_s=-1.0)

    def test_default_concurrency_is_bounded(self) -> None:
        assert DEFAULT_PING_CONCURRENCY >= 1


class TestPingRunnerFanOut:
    async def test_probe_many_preserves_input_order(self) -> None:
        processes = [
            FakeProcess(stdout=f"64 bytes from {i}: icmp_seq=1 ttl=64 time={i}.5 ms".encode())
            for i in range(4)
        ]
        factory, commands = _factory(processes)
        observations = await PingRunner(process_factory=factory).probe_many(
            ["t0", "t1", "t2", "t3"]
        )
        assert [observation.target for observation in observations] == ["t0", "t1", "t2", "t3"]
        assert [command[-1] for command in commands] == ["t0", "t1", "t2", "t3"]

    async def test_concurrency_is_bounded_by_semaphore(self) -> None:
        """§33: concurrent probes MUST be limited — pinned with a tracker."""

        class Tracker:
            def __init__(self) -> None:
                self.active = 0
                self.max_active = 0

        tracker = Tracker()

        class TrackingProcess(FakeProcess):
            async def communicate(self):
                tracker.active += 1
                tracker.max_active = max(tracker.max_active, tracker.active)
                await asyncio.sleep(0.02)
                tracker.active -= 1
                self.returncode = 0
                return b"64 bytes from x: icmp_seq=1 ttl=64 time=0.5 ms", b""

        async def factory(command):
            return TrackingProcess()

        runner = PingRunner(process_factory=factory, concurrency=2)
        observations = await runner.probe_many([f"t{i}" for i in range(6)])
        assert len(observations) == 6
        assert tracker.max_active <= 2

    async def test_empty_fan_out(self) -> None:
        factory, commands = _factory(FakeProcess())
        assert await PingRunner(process_factory=factory).probe_many([]) == ()
        assert commands == []

    async def test_typed_failure_propagates_from_fan_out(self) -> None:
        factory, _ = _factory(
            [
                FakeProcess(stdout=b"64 bytes from x: icmp_seq=1 ttl=64 time=0.5 ms"),
                FakeProcess(stdout=TOTAL_LOSS_OUTPUT.encode(), returncode=1),
            ]
        )
        with pytest.raises(ProfilingError) as excinfo:
            await PingRunner(process_factory=factory, concurrency=1).probe_many(
                ["good", "bad"]
            )
        assert excinfo.value.category is ProfilingErrorCategory.NETWORK_UNREACHABLE


class TestPingObservationValidation:
    def test_received_must_match_samples(self) -> None:
        with pytest.raises(ValueError, match="must equal"):
            PingObservation(
                target="t",
                packets_sent=5,
                packets_received=3,
                samples_ms=(1.0, 2.0),
                summary=summarize_samples((1.0, 2.0)),
            )
