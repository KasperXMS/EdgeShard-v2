"""Shared subprocess plumbing for network probes (spec §33-§34).

Both probe runners spawn external binaries (``ping``, ``iperf3``) through
an injectable factory so tests never touch the host network stack, and
both guarantee process cleanup: a probe process — especially an iperf3
*server* — must never outlive its measurement ("no runaway iperf
servers", P2F DoD).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence

ProcessFactory = Callable[[Sequence[str]], Awaitable[asyncio.subprocess.Process]]
"""Injection point for subprocess creation (tests supply fakes)."""

DEFAULT_TERMINATE_GRACE_S = 5.0
"""Wait between ``terminate`` and ``kill`` escalation."""


async def default_process_factory(command: Sequence[str]) -> asyncio.subprocess.Process:
    """Spawn ``command`` with captured stdout/stderr (Phase 1 convention)."""
    return await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )


async def terminate_process(
    process: asyncio.subprocess.Process,
    *,
    grace_s: float = DEFAULT_TERMINATE_GRACE_S,
) -> None:
    """Guaranteed cleanup: terminate → wait → kill escalation (§34).

    Tolerates an already-exited process (``returncode`` set, or
    ``ProcessLookupError`` on a race) and never leaves a zombie: the
    ``kill`` path also reaps via ``wait``.
    """
    if process.returncode is not None:
        return
    try:
        process.terminate()
        await asyncio.wait_for(process.wait(), timeout=grace_s)
    except ProcessLookupError:
        return  # exited between the check and the signal
    except TimeoutError:
        try:
            process.kill()
        except ProcessLookupError:
            return  # exited between the grace period and the escalation
        await process.wait()


__all__ = [
    "DEFAULT_TERMINATE_GRACE_S",
    "ProcessFactory",
    "default_process_factory",
    "terminate_process",
]
