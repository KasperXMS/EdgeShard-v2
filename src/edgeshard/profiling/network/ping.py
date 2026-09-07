"""ICMP round-trip probing via the system ``ping`` (spec §33).

RTT uses the platform ``ping`` binary — no raw sockets, no extra
dependencies. The runner is platform-aware (Linux ``-c``/``-W`` seconds,
Windows ``-n``/``-w`` milliseconds), bounds fan-out concurrency with a
semaphore (§33: concurrent probes MUST be limited so they do not
contaminate each other), and turns every failure into a typed
``ProfilingError`` (§42): all packets lost → ``NETWORK_UNREACHABLE``
(preserving the binary's stderr), command over budget → ``TIMEOUT``
(after killing the process).

Per-reply RTT samples are parsed from the ``time=Y ms`` reply lines. A
Windows sub-millisecond reply prints ``time<Yms``; the runner records the
documented resolution bound ``Y`` — an upper bound the tool itself
reported, never a fabricated zero (§42, §52.1). Localized Windows builds
translate the ``time`` token (zh-CN prints ``时间<1ms``), so a line that
misses the English pattern but carries the locale-independent ``TTL=``
reply marker falls back to its single millisecond value. Summary
statistics (median/p95-if-enough/jitter-as-stddev/loss) come from the
shared :func:`summarize_samples` and :class:`RttMetrics` conventions.
"""

from __future__ import annotations

import asyncio
import math
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass

from edgeshard.profiling.domain.experiment import ProfilingErrorCategory
from edgeshard.profiling.domain.measurement import SampleSummary, summarize_samples
from edgeshard.profiling.errors import ProfilingError
from edgeshard.profiling.network.processes import (
    ProcessFactory,
    default_process_factory,
    terminate_process,
)

DEFAULT_PING_PACKET_COUNT = 10
"""Packets per probe (§33 policy: 5-10; idle clusters take the upper end)."""

DEFAULT_PING_PACKET_TIMEOUT_S = 2.0
"""Per-packet reply wait (ping ``-W``/``-w``)."""

DEFAULT_PING_COMMAND_TIMEOUT_S_SLACK = 5.0
"""Extra seconds on top of ``packet_count * packet_timeout`` for the process."""

DEFAULT_PING_CONCURRENCY = 4
"""Simultaneous probes across a fan-out (§33: MUST stay bounded)."""

_TIME_PATTERN = re.compile(r"time[=<](\d+(?:\.\d+)?)\s*ms")
"""One reply RTT: ``time=0.045 ms`` (Unix) or ``time<1ms`` (Windows)."""

_REPLY_MARKER = re.compile(r"ttl=", re.IGNORECASE)
"""Locale-independent reply anchor: every Windows/Unix reply line carries
``TTL=``/``ttl=``, while localized Windows builds translate the ``time``
token itself (e.g. zh-CN prints ``时间<1ms``) and timeout/statistics lines
carry no TTL. Used only as the fallback gate for :data:`_ANY_MS_VALUE`."""

_ANY_MS_VALUE = re.compile(r"(\d+(?:\.\d+)?)\s*ms")
"""The RTT on a localized reply line: its only millisecond field."""


def ping_command(
    target: str,
    *,
    packet_count: int = DEFAULT_PING_PACKET_COUNT,
    packet_timeout_s: float = DEFAULT_PING_PACKET_TIMEOUT_S,
    platform: str = sys.platform,
) -> tuple[str, ...]:
    """Platform-correct ``ping`` invocation for one probe.

    Pure command construction (testable without spawning): Windows counts
    with ``-n`` and waits in milliseconds (``-w``); Unix counts with
    ``-c`` and waits in seconds (``-W``, rounded up to whole seconds for
    iputils compatibility).
    """
    if not target:
        raise ValueError("ping target must not be empty")
    if packet_count < 1:
        raise ValueError(f"packet_count must be positive, got {packet_count}")
    if packet_timeout_s <= 0.0:
        raise ValueError(f"packet_timeout_s must be positive, got {packet_timeout_s}")
    if platform == "win32":
        return (
            "ping",
            "-n",
            str(packet_count),
            "-w",
            str(int(packet_timeout_s * 1000)),
            target,
        )
    return (
        "ping",
        "-c",
        str(packet_count),
        "-W",
        str(math.ceil(packet_timeout_s)),
        target,
    )


def parse_ping_output(stdout: str) -> tuple[float, ...]:
    """Per-reply RTT samples in milliseconds, in reply order (§33).

    Only reply lines carrying a ``time`` field count as received; loss
    lines, headers, and the trailing statistics block contribute nothing.
    ``time<Yms`` is recorded as the reported resolution bound ``Y``. A
    localized reply line (``TTL=`` present, English ``time`` token
    translated away) falls back to its single millisecond value; timeout
    and statistics lines carry no ``TTL=``, so they still count as nothing.
    """
    samples: list[float] = []
    for line in stdout.splitlines():
        match = _TIME_PATTERN.search(line)
        if match is None and _REPLY_MARKER.search(line):
            match = _ANY_MS_VALUE.search(line)
        if match:
            samples.append(float(match.group(1)))
    return tuple(samples)


@dataclass(frozen=True)
class PingObservation:
    """One probe's empirical facts (§33).

    Packet loss is derived downstream from sent/received counts
    (``RttMetrics`` convention); jitter is the sample stddev.
    """

    target: str
    packets_sent: int
    packets_received: int
    samples_ms: tuple[float, ...]
    summary: SampleSummary

    def __post_init__(self) -> None:
        if self.packets_received != len(self.samples_ms):
            raise ValueError(
                f"packets_received ({self.packets_received}) must equal the "
                f"sample count ({len(self.samples_ms)})"
            )


class PingRunner:
    """Runs bounded, typed ICMP RTT probes (spec §33)."""

    def __init__(
        self,
        *,
        process_factory: ProcessFactory | None = None,
        packet_count: int = DEFAULT_PING_PACKET_COUNT,
        packet_timeout_s: float = DEFAULT_PING_PACKET_TIMEOUT_S,
        command_timeout_s: float | None = None,
        concurrency: int = DEFAULT_PING_CONCURRENCY,
        platform: str = sys.platform,
    ) -> None:
        if packet_count < 1:
            raise ValueError(f"packet_count must be positive, got {packet_count}")
        if packet_timeout_s <= 0.0:
            raise ValueError(f"packet_timeout_s must be positive, got {packet_timeout_s}")
        if concurrency < 1:
            raise ValueError(f"concurrency must be positive, got {concurrency}")
        self._process_factory = (
            process_factory if process_factory is not None else default_process_factory
        )
        self._packet_count = packet_count
        self._packet_timeout_s = packet_timeout_s
        self._platform = platform
        self._command_timeout_s = (
            command_timeout_s
            if command_timeout_s is not None
            else packet_count * packet_timeout_s + DEFAULT_PING_COMMAND_TIMEOUT_S_SLACK
        )
        if self._command_timeout_s <= 0.0:
            raise ValueError(
                f"command_timeout_s must be positive, got {self._command_timeout_s}"
            )
        self._concurrency = concurrency

    async def probe(self, target: str, *, packet_count: int | None = None) -> PingObservation:
        """One RTT probe against ``target``; failures stay typed (§42)."""
        count = packet_count if packet_count is not None else self._packet_count
        if count < 1:
            raise ValueError(f"packet_count must be positive, got {count}")
        command = ping_command(
            target,
            packet_count=count,
            packet_timeout_s=self._packet_timeout_s,
            platform=self._platform,
        )
        try:
            process = await self._process_factory(command)
        except OSError as exc:
            raise ProfilingError(
                ProfilingErrorCategory.NETWORK_UNREACHABLE,
                f"system ping executable is not usable: {exc}",
                {"target": target},
            ) from exc
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=self._command_timeout_s
            )
        except TimeoutError as exc:
            await terminate_process(process)
            raise ProfilingError(
                ProfilingErrorCategory.TIMEOUT,
                f"ping probe of {target!r} exceeded {self._command_timeout_s:.1f}s",
                {"target": target, "command": " ".join(command)},
            ) from exc
        samples = parse_ping_output(stdout.decode("utf-8", errors="replace"))
        if not samples:
            raise ProfilingError(
                ProfilingErrorCategory.NETWORK_UNREACHABLE,
                f"all {count} ping packets to {target!r} were lost",
                {
                    "target": target,
                    "packets_sent": count,
                    "stderr": stderr.decode("utf-8", errors="replace"),
                },
            )
        return PingObservation(
            target=target,
            packets_sent=count,
            packets_received=len(samples),
            samples_ms=samples,
            summary=summarize_samples(samples),
        )

    async def probe_many(
        self, targets: Iterable[str], *, packet_count: int | None = None
    ) -> tuple[PingObservation, ...]:
        """Semaphore-bounded fan-out over targets (§33 concurrency limit).

        Results keep the input order. The first typed failure propagates —
        callers needing per-target failure isolation (dense RTT matrices
        where one unreachable host must not sink the run) probe through
        the case-level runner instead, which records a failure per case.
        """
        semaphore = asyncio.Semaphore(self._concurrency)

        async def bounded(target: str) -> PingObservation:
            async with semaphore:
                return await self.probe(target, packet_count=packet_count)

        return tuple(await asyncio.gather(*(bounded(target) for target in targets)))


__all__ = [
    "DEFAULT_PING_CONCURRENCY",
    "DEFAULT_PING_PACKET_COUNT",
    "DEFAULT_PING_PACKET_TIMEOUT_S",
    "PingObservation",
    "PingRunner",
    "parse_ping_output",
    "ping_command",
]
