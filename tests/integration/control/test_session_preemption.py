"""Session preemption over real gRPC: two Agents, one worker_id (spec §27, §35).

The regression these tests pin: an Agent whose heartbeat is rejected must
NOT blindly re-register. Before typed rejections, two processes claiming
the same ``worker_id`` would steal the session back and forth forever
(each registration invalidates the other's session — ping-pong). With the
§30/§46 rejection taxonomy the only stable outcome is: the *superseded*
Agent stops, the newer one keeps the session.

Both tests run a real ``MasterService`` behind a real ``grpc.aio`` server
and two real ``WorkerAgent``\\ s with real ``WorkerRegistryClient``\\ s; only
local inspection is canned (both Agents report the same identity and
capability — the situation a duplicated deployment or a restarted-but-
not-exited process creates). Timing is real: the Master dictates a 50 ms
heartbeat cadence, so preemption resolves within a few beats.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable

from edgeshard.cluster.state import WorkerStatus
from edgeshard.control.master.config import MasterConfig
from edgeshard.control.master.service import MasterService
from edgeshard.control.worker.agent import LocalInspection
from edgeshard.control.worker.config import (
    MasterConfig as WorkerMasterEndpoint,
)
from edgeshard.control.worker.config import (
    ReconnectConfig,
    WorkerConfig,
    WorkerSection,
)
from edgeshard.control.worker.master_client import WorkerAgent
from edgeshard.protocol.control.grpc_client import WorkerRegistryClient
from edgeshard.protocol.control.grpc_server import start_control_server
from edgeshard.protocol.control.mapper import (
    HeartbeatRequest,
    HeartbeatResponse,
    RegisterWorkerRequest,
    RegisterWorkerResponse,
    RejectionReason,
    UpdateCapabilityRequest,
    UpdateCapabilityResponse,
)
from factories import make_rtx_capability, make_worker_identity, make_worker_state

HOST = "127.0.0.1"

TEST_CONFIG = MasterConfig(
    heartbeat_interval_ms=50,
    suspect_after_ms=150,
    offline_after_ms=300,
    liveness_tick_ms=10,
)


class CannedInspector:
    """Both Agents report the *same* worker identity, capability and state."""

    def __init__(self, inspection: LocalInspection) -> None:
        self._inspection = inspection
        self.started = 0
        self.closed = 0
        self.inspects = 0

    async def start(self) -> None:
        self.started += 1

    async def inspect(self) -> LocalInspection:
        self.inspects += 1
        return self._inspection

    async def close(self) -> None:
        self.closed += 1


class RecordingClient:
    """Wraps the real gRPC client, counting calls and stale rejections."""

    def __init__(self, inner: WorkerRegistryClient) -> None:
        self._inner = inner
        self.registrations = 0
        self.heartbeats = 0
        self.updates = 0
        self.stale_rejections = 0

    async def register_worker(
        self, request: RegisterWorkerRequest, *, timeout: float | None = None
    ) -> RegisterWorkerResponse:
        self.registrations += 1
        return await self._inner.register_worker(request, timeout=timeout)

    async def heartbeat(
        self, request: HeartbeatRequest, *, timeout: float | None = None
    ) -> HeartbeatResponse:
        self.heartbeats += 1
        response = await self._inner.heartbeat(request, timeout=timeout)
        if not response.accepted and response.reason is RejectionReason.STALE_SESSION:
            self.stale_rejections += 1
        return response

    async def update_capability(
        self, request: UpdateCapabilityRequest, *, timeout: float | None = None
    ) -> UpdateCapabilityResponse:
        self.updates += 1
        return await self._inner.update_capability(request, timeout=timeout)

    async def close(self) -> None:
        await self._inner.close()


async def wait_until(
    predicate: Callable[[], bool], *, timeout: float = 10.0, what: str
) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"timed out after {timeout}s waiting for {what}")


class Fixture:
    """Server + two same-worker_id Agents with recording clients."""

    def __init__(self) -> None:
        self.worker_id = ""
        self.clients: list[RecordingClient] = []
        self.inspectors: list[CannedInspector] = []
        self.agents: list[WorkerAgent] = []
        self.tasks: list[asyncio.Task[None]] = []

    async def setup(self, port: int) -> None:
        identity = make_worker_identity()
        self.worker_id = identity.worker_id
        inspection = LocalInspection(
            identity=identity,
            capability=make_rtx_capability(),
            state=make_worker_state(identity.worker_id),
        )
        endpoint = f"{HOST}:{port}"
        config = WorkerConfig(
            worker=WorkerSection(
                master=WorkerMasterEndpoint(endpoint=endpoint),
                heartbeat_interval_s=0.05,
                reconnect=ReconnectConfig(initial_delay_s=0.05, max_delay_s=0.2),
            ),
        )
        self.inspectors = [CannedInspector(inspection), CannedInspector(inspection)]
        self.clients = [
            RecordingClient(WorkerRegistryClient(endpoint)) for _ in range(2)
        ]
        self.agents = [
            WorkerAgent(config, client=self.clients[i], inspector=self.inspectors[i])
            for i in range(2)
        ]

    def spawn(self, index: int) -> asyncio.Task[None]:
        task = asyncio.create_task(self.agents[index].run(), name=f"agent-{index}")
        self.tasks.append(task)
        return task

    async def aclose(self) -> None:
        for agent in self.agents:
            agent.request_stop()
        for task in self.tasks:
            if not task.done():
                task.cancel()
        for task in self.tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        for client in self.clients:
            await client.close()


async def test_superseded_agent_stops_and_never_steals_the_session_back() -> None:
    """§27/§35: STALE_SESSION means *this* Agent exits — no re-registration.

    Agent A is heartbeating; Agent B registers with the same worker_id and
    preempts it. A's next heartbeat is rejected STALE_SESSION, and A must
    stop on its own: exactly one registration for its whole life, while B
    keeps heartbeating on the one surviving session.
    """
    service = MasterService(TEST_CONFIG)
    server, port = await start_control_server(service, host=HOST, port=0)
    fixture = Fixture()
    async with service:
        try:
            await fixture.setup(port)
            agent_b = fixture.agents[1]
            client_a, client_b = fixture.clients

            task_a = fixture.spawn(0)
            await wait_until(
                lambda: client_a.registrations == 1 and client_a.heartbeats >= 1,
                what="agent A registered and heartbeating",
            )

            # B preempts A's session with the same worker_id.
            fixture.spawn(1)
            await wait_until(
                lambda: client_b.registrations == 1, what="agent B registration"
            )

            # A stops by itself on the next rejected heartbeat — the typed
            # STALE_SESSION reason is the exit signal, not a retry signal.
            await asyncio.wait_for(task_a, timeout=5.0)
            assert task_a.exception() is None  # a clean stop, not a crash
            assert client_a.registrations == 1  # never re-registered
            assert client_a.stale_rejections == 1  # exactly one rejection, then exit
            assert fixture.inspectors[0].closed == 1  # lifecycle closed on stop

            # The Master's one session now belongs to B, and B keeps beating.
            session = service.sessions.current(fixture.worker_id)
            assert session is not None
            assert session.instance_id == agent_b.instance_id
            beats = client_b.heartbeats
            await wait_until(
                lambda: client_b.heartbeats >= beats + 3,
                what="agent B continuing to heartbeat",
            )
            assert client_b.registrations == 1
            assert client_b.stale_rejections == 0
            assert service.worker_status(fixture.worker_id) is WorkerStatus.ONLINE
        finally:
            await fixture.aclose()
            await server.stop(grace=None)


async def test_two_agents_do_not_ping_pong_sessions() -> None:
    """The ping-pong regression: total registrations stay exactly 2.

    Before typed rejections, A would re-register after every stale-session
    rejection, invalidating B's session, which would make B re-register,
    and so on forever. Here both Agents run against the real Master for
    many heartbeat intervals: A registers once and exits when superseded,
    B registers once and holds the session for the rest of the test — the
    registration counter across both clients never exceeds 2, and the
    Master's session instance never flips back to A.
    """
    service = MasterService(TEST_CONFIG)
    server, port = await start_control_server(service, host=HOST, port=0)
    fixture = Fixture()
    async with service:
        try:
            await fixture.setup(port)
            agent_b = fixture.agents[1]
            client_a, client_b = fixture.clients

            task_a = fixture.spawn(0)
            # A must be fully registered *and* heartbeating before B starts:
            # registrations are counted at send time, and only a sent
            # heartbeat proves A's registration was committed at the Master.
            # Otherwise B's registration could be processed first and the
            # roles would flip (A preempts B), making the test nondeterministic.
            await wait_until(
                lambda: client_a.registrations == 1 and client_a.heartbeats >= 1,
                what="agent A registered and heartbeating",
            )
            fixture.spawn(1)

            # Let the survivor heartbeat for a long stretch (>= 20 beats at
            # the Master's 50 ms cadence) — ample time for a ping-pong war
            # to have racked up registrations had the bug still existed.
            await wait_until(
                lambda: client_b.heartbeats >= 20,
                timeout=15.0,
                what="20 uninterrupted heartbeats from agent B",
            )

            assert task_a.done()  # the superseded agent exited on its own
            assert task_a.exception() is None
            assert client_a.registrations + client_b.registrations == 2
            assert client_a.registrations == 1
            assert client_b.registrations == 1
            assert client_b.stale_rejections == 0

            # The session never flipped back: B's instance still owns it and
            # its heartbeats kept being accepted (sequence advancing proves
            # no rejection/re-registration cycle reset it).
            session = service.sessions.current(fixture.worker_id)
            assert session is not None
            assert session.instance_id == agent_b.instance_id
            assert session.last_accepted_sequence >= 15
            assert len(service.registry) == 1  # one worker_id, one record
            assert service.worker_status(fixture.worker_id) is WorkerStatus.ONLINE
        finally:
            await fixture.aclose()
            await server.stop(grace=None)
