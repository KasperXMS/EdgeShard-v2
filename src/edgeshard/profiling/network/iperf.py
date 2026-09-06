"""iperf3 single-flow bandwidth baselines (spec §34-§35).

Rules pinned here:

* JSON output only — the runner invokes ``iperf3 --json`` and parses the
  documented sections (TCP: ``end.sum_received`` with retransmits from
  ``end.sum_sent``; UDP: ``end.sum``), never human-readable text (§34);
* every failure is typed (§42): a JSON ``error`` payload →
  ``NETWORK_UNREACHABLE``, a missing/unusable binary →
  ``IPERF_UNAVAILABLE``, an over-budget run → ``TIMEOUT``, undecodable or
  structurally invalid output → ``BENCHMARK_FAILED`` with the *raw stderr
  preserved* in the failure details (§34 diagnostics rule);
* server cleanup is guaranteed (P2F DoD "no runaway iperf servers"):
  one-off servers (``-s -1``) as the first layer and terminate→kill
  escalation in a ``finally`` as the second;
* measurements are idle single-flow baselines — the regime is recorded by
  the case spec (§35) and never presented as guaranteed concurrent
  bandwidth (§52.7). Reverse runs use ``-R`` and report through the same
  client-side JSON fields.
"""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from edgeshard.profiling.domain.experiment import ProfilingErrorCategory
from edgeshard.profiling.domain.network import NetworkDirection, NetworkTransport
from edgeshard.profiling.errors import ProfilingError
from edgeshard.profiling.network.processes import (
    ProcessFactory,
    default_process_factory,
    terminate_process,
)

DEFAULT_IPERF3_BINARY = "iperf3"
DEFAULT_IPERF3_PORT = 5201
DEFAULT_IPERF3_DURATION_S = 5.0
DEFAULT_IPERF3_TIMEOUT_GRACE_S = 15.0
"""Client budget is ``duration_s + grace`` — connects, ramps down, prints JSON."""

DEFAULT_IPERF3_SERVER_STARTUP_S = 0.25
"""Settle time after spawning a server before a client may connect."""

DEFAULT_IPERF3_UDP_TARGET_BITS_PER_SECOND = 0
"""``-b 0``: send as fast as the path allows (UDP baseline, §34)."""

_MAX_PRESERVED_STDOUT_CHARS = 2048
"""stdout is capped in failure details; stderr is preserved in full (§34)."""


@dataclass(frozen=True)
class Iperf3Probe:
    """One directed single-flow bandwidth probe request (§32, §34)."""

    target: str
    transport: NetworkTransport = NetworkTransport.TCP
    direction: NetworkDirection = NetworkDirection.FORWARD
    duration_s: float = DEFAULT_IPERF3_DURATION_S
    payload_bytes: int | None = None
    port: int = DEFAULT_IPERF3_PORT

    def __post_init__(self) -> None:
        if not self.target:
            raise ValueError("target must not be empty")
        if self.duration_s <= 0.0:
            raise ValueError(f"duration_s must be positive, got {self.duration_s}")
        if self.payload_bytes is not None and self.payload_bytes <= 0:
            raise ValueError(f"payload_bytes must be positive, got {self.payload_bytes}")
        if not 1 <= self.port <= 65535:
            raise ValueError(f"port must be within [1, 65535], got {self.port}")


def iperf3_server_command(
    *, port: int = DEFAULT_IPERF3_PORT, binary: str = DEFAULT_IPERF3_BINARY
) -> tuple[str, ...]:
    """One-off server invocation (``-1``: exits after a single test)."""
    if not 1 <= port <= 65535:
        raise ValueError(f"port must be within [1, 65535], got {port}")
    return (binary, "-s", "-1", "-p", str(port))


def iperf3_client_command(
    probe: Iperf3Probe, *, binary: str = DEFAULT_IPERF3_BINARY
) -> tuple[str, ...]:
    """JSON-mode client invocation for one directed probe.

    ``-R`` reverses the flow (destination → source) while the JSON report
    stays on the client, so forward and reverse runs parse identically
    (§34: profile both directions). ``-l`` sets the write buffer to the
    workload-derived payload size (§36).
    """
    command = [
        binary,
        "-c",
        probe.target,
        "-p",
        str(probe.port),
        "--json",
        "-t",
        f"{probe.duration_s:g}",
    ]
    if probe.transport is NetworkTransport.UDP:
        command += ["-u", "-b", str(DEFAULT_IPERF3_UDP_TARGET_BITS_PER_SECOND)]
    if probe.payload_bytes is not None:
        command += ["-l", str(probe.payload_bytes)]
    if probe.direction is NetworkDirection.REVERSE:
        command += ["-R"]
    return tuple(command)


@dataclass(frozen=True)
class Iperf3Observation:
    """Parsed single-flow throughput fact (iperf3 JSON conventions)."""

    bits_per_second: float
    retransmits: int | None


def parse_iperf3_json(payload: Mapping[str, Any]) -> Iperf3Observation:
    """Throughput observation from a decoded iperf3 ``--json`` payload.

    Pure parsing (no I/O): an ``error`` key means the run never produced a
    measurement → ``NETWORK_UNREACHABLE`` with iperf3's own message; a
    missing ``end`` section or a missing/non-numeric throughput value →
    ``BENCHMARK_FAILED``. Values are never substituted (§42, §52.2).
    """
    if "error" in payload:
        raise ProfilingError(
            ProfilingErrorCategory.NETWORK_UNREACHABLE,
            f"iperf3 reported an error: {payload['error']}",
        )
    end = payload.get("end")
    if not isinstance(end, Mapping):
        raise ProfilingError(
            ProfilingErrorCategory.BENCHMARK_FAILED,
            "iperf3 JSON payload has no 'end' section",
        )
    sum_received = end.get("sum_received")
    sum_sent = end.get("sum_sent")
    udp_sum = end.get("sum")
    # iperf3 JSON conventions: TCP reports sender and receiver summaries;
    # the receiver side is the delivered throughput. UDP reports one
    # aggregate 'sum'. Reverse (-R) runs keep the same client-side fields.
    throughput_section: Any = None
    if isinstance(sum_received, Mapping):
        throughput_section = sum_received
    elif isinstance(udp_sum, Mapping):
        throughput_section = udp_sum
    elif isinstance(sum_sent, Mapping):
        throughput_section = sum_sent
    bits_per_second = (
        throughput_section.get("bits_per_second")
        if isinstance(throughput_section, Mapping)
        else None
    )
    if isinstance(bits_per_second, bool) or not isinstance(bits_per_second, (int, float)):
        raise ProfilingError(
            ProfilingErrorCategory.BENCHMARK_FAILED,
            "iperf3 JSON payload carries no numeric bits_per_second observation",
        )
    value = float(bits_per_second)
    if not math.isfinite(value) or value < 0.0:
        raise ProfilingError(
            ProfilingErrorCategory.BENCHMARK_FAILED,
            f"iperf3 reported a non-finite or negative throughput: {value}",
        )
    retransmits: int | None = None
    if isinstance(sum_sent, Mapping):
        raw_retransmits = sum_sent.get("retransmits")
        if isinstance(raw_retransmits, int) and not isinstance(raw_retransmits, bool):
            retransmits = raw_retransmits
    return Iperf3Observation(bits_per_second=value, retransmits=retransmits)


class Iperf3Runner:
    """Spawns iperf3 servers/clients and parses JSON results (§34)."""

    def __init__(
        self,
        *,
        process_factory: ProcessFactory | None = None,
        binary: str = DEFAULT_IPERF3_BINARY,
        timeout_grace_s: float = DEFAULT_IPERF3_TIMEOUT_GRACE_S,
        server_startup_s: float = DEFAULT_IPERF3_SERVER_STARTUP_S,
    ) -> None:
        if not binary:
            raise ValueError("binary must not be empty")
        if timeout_grace_s <= 0.0:
            raise ValueError(f"timeout_grace_s must be positive, got {timeout_grace_s}")
        if server_startup_s < 0.0:
            raise ValueError(f"server_startup_s must not be negative, got {server_startup_s}")
        self._process_factory = (
            process_factory if process_factory is not None else default_process_factory
        )
        self._binary = binary
        self._timeout_grace_s = timeout_grace_s
        self._server_startup_s = server_startup_s

    async def start_server(self, *, port: int = DEFAULT_IPERF3_PORT) -> asyncio.subprocess.Process:
        """Spawn a one-off iperf3 server; caller guarantees termination.

        In the real cluster the server runs on the *destination* worker —
        P2G orchestration places it there through the control plane. This
        method covers the local/loopback case; either way cleanup goes
        through :func:`terminate_process` (no runaway servers).
        """
        process = await self._spawn(iperf3_server_command(port=port, binary=self._binary))
        if self._server_startup_s > 0.0:
            await asyncio.sleep(self._server_startup_s)
        return process

    async def run_client(self, probe: Iperf3Probe) -> Iperf3Observation:
        """Run one probe against an already-listening server (§34)."""
        command = iperf3_client_command(probe, binary=self._binary)
        process = await self._spawn(command)
        timeout = probe.duration_s + self._timeout_grace_s
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except TimeoutError as exc:
            await terminate_process(process)
            raise ProfilingError(
                ProfilingErrorCategory.TIMEOUT,
                f"iperf3 probe of {probe.target!r} exceeded {timeout:.1f}s",
                {"target": probe.target, "command": " ".join(command)},
            ) from exc
        out_text = stdout.decode("utf-8", errors="replace")
        err_text = stderr.decode("utf-8", errors="replace")
        try:
            payload = json.loads(out_text)
        except json.JSONDecodeError as exc:
            # §34: preserve raw diagnostics — stderr is where iperf3
            # explains itself when the JSON report never materialized.
            raise ProfilingError(
                ProfilingErrorCategory.BENCHMARK_FAILED,
                f"iperf3 produced undecodable JSON output "
                f"(exit code {process.returncode}): {exc}",
                {
                    "target": probe.target,
                    "stderr": err_text,
                    "stdout_head": out_text[:_MAX_PRESERVED_STDOUT_CHARS],
                },
            ) from exc
        if not isinstance(payload, dict):
            raise ProfilingError(
                ProfilingErrorCategory.BENCHMARK_FAILED,
                "iperf3 JSON payload is not an object",
                {"target": probe.target, "stderr": err_text},
            )
        return parse_iperf3_json(payload)

    async def measure(self, probe: Iperf3Probe) -> Iperf3Observation:
        """Single-host convenience: local server + client, cleanup guaranteed.

        The server is terminated in a ``finally`` no matter how the client
        fails — timeouts, refused connections, and bad output all leave
        zero processes behind (P2F DoD).
        """
        server = await self.start_server(port=probe.port)
        try:
            return await self.run_client(probe)
        finally:
            await terminate_process(server)

    async def _spawn(self, command: Sequence[str]) -> asyncio.subprocess.Process:
        try:
            return await self._process_factory(command)
        except OSError as exc:
            raise ProfilingError(
                ProfilingErrorCategory.IPERF_UNAVAILABLE,
                f"iperf3 executable {self._binary!r} is not usable: {exc}",
                {"command": " ".join(command)},
            ) from exc


__all__ = [
    "DEFAULT_IPERF3_BINARY",
    "DEFAULT_IPERF3_DURATION_S",
    "DEFAULT_IPERF3_PORT",
    "Iperf3Observation",
    "Iperf3Probe",
    "Iperf3Runner",
    "iperf3_client_command",
    "iperf3_server_command",
    "parse_iperf3_json",
]
