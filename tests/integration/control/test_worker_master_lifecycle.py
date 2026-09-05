"""Worker Agent ↔ Master lifecycle over real gRPC (spec §27, §44, §52).

The in-process counterpart of the subprocess Test A: a real
``MasterService`` (with injected fake monotonic/wall clocks so liveness
transitions cost zero real sleeping, spec §37) behind a real ``grpc.aio``
server, and a real ``WorkerAgent`` owning a real ``WorkerRegistryClient``
and running real local inspections on the dev host (no Docker daemon or
NVIDIA driver required — the probes report empty fragments, spec §47).

Covers: registration lands exactly one worker; heartbeats are accepted
with advancing sequences at the Master-dictated cadence (the Worker's
configured preference is deliberately absurd); the worker stays ONLINE;
after the agent is cancelled, fake-clock advancement degrades it to
SUSPECT then OFFLINE while the registry entry remains (§32); and an agent
started *before* the Master exists backs off and registers once the
Master binds (real-transport reconnect, §27).
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

from edgeshard.cluster.state import WorkerStatus
from edgeshard.control.master.config import MasterConfig
from edgeshard.control.master.service import MasterService
from edgeshard.control.worker.config import (
    MasterConfig as WorkerMasterEndpoint,
)
from edgeshard.control.worker.config import (
    ModelStoreSection,
    ReconnectConfig,
    RuntimeSection,
    WorkerConfig,
    WorkerSection,
)
from edgeshard.control.worker.master_client import WorkerAgent
from edgeshard.protocol.control.grpc_server import start_control_server

HOST = "127.0.0.1"

TEST_CONFIG = MasterConfig(
    heartbeat_interval_ms=100,
    suspect_after_ms=500,
    offline_after_ms=1_000,
    liveness_tick_ms=5,
)
"""Master-side cadence of 100 ms; the fake monotonic clock means real
inspection latency never advances Master time, so these thresholds are
crossed only when the test says so."""


class FakeClock:
    """Injected Master clocks: monotonic for liveness, wall for debug."""

    def __init__(self) -> None:
        self.now = 1_000.0
        self.wall_base = datetime(2026, 1, 1, tzinfo=UTC)

    def monotonic(self) -> float:
        return self.now

    def wall(self) -> datetime:
        return self.wall_base + timedelta(seconds=self.now - 1_000.0)

    def advance(self, seconds: float) -> None:
        self.now += seconds


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((HOST, 0))
        return int(sock.getsockname()[1])


def make_worker_config(tmp_path: Path, port: int) -> WorkerConfig:
    return WorkerConfig(
        worker=WorkerSection(
            identity_path=tmp_path / "worker-id",
            master=WorkerMasterEndpoint(endpoint=f"{HOST}:{port}"),
            # Absurd on purpose: the Master's 100 ms must win (§29/§30).
            heartbeat_interval_s=99.0,
            reconnect=ReconnectConfig(initial_delay_s=0.05, max_delay_s=0.2),
        ),
        model_store=ModelStoreSection(root=tmp_path / "models"),
        runtime=RuntimeSection(discover_managed_containers=False),
    )


async def wait_for(
    predicate: Callable[[], bool], timeout_s: float = 60.0, message: str = "condition"
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.05)
    raise TimeoutError(f"timed out waiting for {message}")


async def test_agent_registers_and_heartbeats_then_degrades_offline(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    service = MasterService(TEST_CONFIG, monotonic=clock.monotonic, wall=clock.wall)
    server, port = await start_control_server(service, host=HOST, port=0)
    agent = WorkerAgent(make_worker_config(tmp_path, port))
    task = asyncio.create_task(agent.run())
    try:
        async with service:
            # §52 Test A shape: exactly one worker, registered, ONLINE.
            await wait_for(lambda: len(service.registry) == 1, message="registration")
            (record,) = service.registry.list_workers()
            worker_id = record.identity.worker_id
            persisted = (tmp_path / "worker-id").read_text(encoding="utf-8").strip()
            assert worker_id == persisted

            # Heartbeats flow at the Master cadence with advancing sequences.
            await wait_for(
                lambda: service.sessions.current(worker_id).last_accepted_sequence
                >= 3,
                message="three accepted heartbeats",
            )
            assert service.worker_status(worker_id) == WorkerStatus.ONLINE
            stored = service.states.get(worker_id)
            assert stored is not None
            assert stored.state.worker_id == worker_id
            # Master-side receive timestamps come from the Master's own
            # clocks (spec §31), not from worker_reported_at.
            assert stored.received_monotonic == clock.monotonic()

            # Capability made it across: the dev host reports at least the
            # CPU device even without Docker or NVIDIA (spec §47).
            assert record.capability.devices
            assert record.capability.capability_revision
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await server.stop(grace=None)

    # §32 with zero real sleeping: once heartbeats stop, the fake clock
    # alone drives SUSPECT → OFFLINE, and the entry is never deleted.
    clock.advance(0.501)
    assert service.worker_status(worker_id) == WorkerStatus.SUSPECT
    clock.advance(0.5)
    assert service.worker_status(worker_id) == WorkerStatus.OFFLINE
    assert len(service.registry) == 1


async def test_agent_backs_off_until_master_appears(tmp_path: Path) -> None:
    """Real-transport reconnect (§27): worker up first, Master binds later."""
    port = free_port()
    agent = WorkerAgent(make_worker_config(tmp_path, port))
    task = asyncio.create_task(agent.run())
    try:
        # Let the agent fail against the empty port and enter backoff.
        await asyncio.sleep(0.3)

        clock = FakeClock()
        service = MasterService(TEST_CONFIG, monotonic=clock.monotonic, wall=clock.wall)
        server, bound = await start_control_server(service, host=HOST, port=port)
        assert bound == port
        try:
            async with service:
                await wait_for(
                    lambda: len(service.registry) == 1,
                    timeout_s=15.0,
                    message="registration after reconnect",
                )
                (record,) = service.registry.list_workers()
                await wait_for(
                    lambda: service.sessions.current(
                        record.identity.worker_id
                    ).last_accepted_sequence
                    >= 1,
                    timeout_s=15.0,
                    message="first heartbeat after reconnect",
                )
                assert (
                    service.worker_status(record.identity.worker_id)
                    == WorkerStatus.ONLINE
                )
        finally:
            await server.stop(grace=None)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
