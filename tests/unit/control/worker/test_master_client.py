"""WorkerAgent loop tests (Phase 1 spec §27, §30, §44, §56 P1G).

Everything is faked except the Agent itself: a scripted control client, a
canned-inspection inspector, a recording sleeper and a fake clock. The
tests pin the §27 outer loop contract — exponential backoff with reset on
success, fresh inspection state on every registration, per-session
sequences restarting at 1, Master-dictated cadence, and capability-revision
drift answered with a full UpdateCapability carrying the atomically sampled
state (§16).

Rejections are classified by their typed ``RejectionReason`` (§30):
UNKNOWN_WORKER/REREGISTER_REQUIRED re-register immediately (never restoring
a stale session), STALE_SESSION stops the superseded Agent without
re-registering (no session ping-pong), and INSTANCE_MISMATCH/OUT_OF_ORDER
fail loudly instead of silently retrying.
"""

from __future__ import annotations

import dataclasses
import logging
import uuid
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import grpc
import pytest

from edgeshard.control.worker.agent import LocalInspection
from edgeshard.control.worker.config import (
    MasterConfig as WorkerMasterEndpoint,
)
from edgeshard.control.worker.config import (
    ReconnectConfig,
    WorkerConfig,
    WorkerSection,
)
from edgeshard.control.worker.master_client import WorkerAgent, WorkerAgentError
from edgeshard.protocol.control.mapper import (
    CONTROL_PROTOCOL_VERSION,
    HeartbeatRequest,
    HeartbeatResponse,
    RegisterWorkerRequest,
    RegisterWorkerResponse,
    RejectionReason,
    UpdateCapabilityRequest,
    UpdateCapabilityResponse,
)
from factories import finalize, make_rtx_capability, make_worker_identity, make_worker_state

HB_INTERVAL_S = 0.5  # FakeClient default: 500 ms Master-dictated cadence


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeClient:
    """Scripted control client: queues of effects, canned defaults beyond."""

    def __init__(self, heartbeat_interval_ms: int = 500) -> None:
        self.registrations: list[RegisterWorkerRequest] = []
        self.heartbeats: list[HeartbeatRequest] = []
        self.updates: list[UpdateCapabilityRequest] = []
        self.register_effects: deque[BaseException | None] = deque()
        self.heartbeat_effects: deque[BaseException | HeartbeatResponse | None] = deque()
        self.update_effects: deque[BaseException | UpdateCapabilityResponse | None] = deque()
        self.closed = False
        self.heartbeat_interval_ms = heartbeat_interval_ms
        self._sessions = 0

    async def register_worker(
        self, request: RegisterWorkerRequest, *, timeout: float | None = None
    ) -> RegisterWorkerResponse:
        self.registrations.append(request)
        effect = self.register_effects.popleft() if self.register_effects else None
        if isinstance(effect, BaseException):
            raise effect
        self._sessions += 1
        return RegisterWorkerResponse(
            session_id=f"session-{self._sessions}",
            heartbeat_interval_ms=self.heartbeat_interval_ms,
            server_protocol_version=CONTROL_PROTOCOL_VERSION,
        )

    async def heartbeat(
        self, request: HeartbeatRequest, *, timeout: float | None = None
    ) -> HeartbeatResponse:
        self.heartbeats.append(request)
        effect = self.heartbeat_effects.popleft() if self.heartbeat_effects else None
        if isinstance(effect, BaseException):
            raise effect
        return effect if effect is not None else HeartbeatResponse(accepted=True)

    async def update_capability(
        self, request: UpdateCapabilityRequest, *, timeout: float | None = None
    ) -> UpdateCapabilityResponse:
        self.updates.append(request)
        effect = self.update_effects.popleft() if self.update_effects else None
        if isinstance(effect, BaseException):
            raise effect
        return effect if effect is not None else UpdateCapabilityResponse(accepted=True)

    async def close(self) -> None:
        self.closed = True


class FakeInspector:
    """Returns canned inspections; sticks on the last once exhausted.

    Implements the ``Inspector`` lifecycle protocol: ``run`` must start the
    inspector once before registering and close it once when the Agent
    stops, whatever the exit path (stop request, superseded, fatal error).
    """

    def __init__(self, results: Sequence[LocalInspection]) -> None:
        self._results = list(results)
        self.returned: list[LocalInspection] = []
        self.started = 0
        self.closed = 0

    async def start(self) -> None:
        self.started += 1

    async def inspect(self) -> LocalInspection:
        inspection = self._results[min(len(self.returned), len(self._results) - 1)]
        self.returned.append(inspection)
        return inspection

    async def close(self) -> None:
        self.closed += 1


class FakeSleeper:
    """Records delays; optionally stops the agent after N sleeps."""

    def __init__(self, stop_after: int | None = None) -> None:
        self.delays: list[float] = []
        self.agent: WorkerAgent | None = None
        self._stop_after = stop_after

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)
        if self._stop_after is not None and len(self.delays) >= self._stop_after:
            assert self.agent is not None
            self.agent.request_stop()


class FakeClock:
    """Successive tz-aware timestamps for worker_reported_at."""

    def __init__(self) -> None:
        self._now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        self._step = timedelta(seconds=5)
        self.values: list[datetime] = []

    def __call__(self) -> datetime:
        value = self._now
        self.values.append(value)
        self._now += self._step
        return value


@dataclass
class Rig:
    agent: WorkerAgent
    client: FakeClient
    inspector: FakeInspector
    sleeper: FakeSleeper
    clock: FakeClock


def make_inspection(
    capability=None, worker_id: str = "w-1"
) -> LocalInspection:
    capability = capability if capability is not None else make_rtx_capability()
    return LocalInspection(
        identity=make_worker_identity(worker_id),
        capability=capability,
        state=make_worker_state(worker_id),
    )


def make_rig(
    *,
    inspections: Sequence[LocalInspection] | None = None,
    register_effects: Sequence[BaseException | None] = (),
    heartbeat_effects: Sequence[BaseException | HeartbeatResponse | None] = (),
    update_effects: Sequence[BaseException | UpdateCapabilityResponse | None] = (),
    stop_after: int | None = 4,
    interval_ms: int = 500,
    initial_delay_s: float = 1.0,
    max_delay_s: float = 4.0,
    configured_interval_s: float = 99.0,
    profiling_endpoint: str | None = None,
) -> Rig:
    client = FakeClient(heartbeat_interval_ms=interval_ms)
    client.register_effects = deque(register_effects)
    client.heartbeat_effects = deque(heartbeat_effects)
    client.update_effects = deque(update_effects)
    inspector = FakeInspector(inspections or [make_inspection()])
    sleeper = FakeSleeper(stop_after=stop_after)
    clock = FakeClock()
    config = WorkerConfig(
        worker=WorkerSection(
            identity_path=Path("worker-id"),
            master=WorkerMasterEndpoint(endpoint="master.test:51000"),
            heartbeat_interval_s=configured_interval_s,
            reconnect=ReconnectConfig(
                initial_delay_s=initial_delay_s, max_delay_s=max_delay_s
            ),
        )
    )
    agent = WorkerAgent(
        config,
        client=client,
        inspector=inspector,
        sleeper=sleeper,
        clock=clock,
        profiling_endpoint=profiling_endpoint,
    )
    sleeper.agent = agent
    return Rig(agent=agent, client=client, inspector=inspector, sleeper=sleeper, clock=clock)


def evolved_capability():
    """A plausible rare capability change (spec §16): one new platform tag."""
    capability = make_rtx_capability()
    gpu = capability.devices[1]
    devices = (
        capability.devices[0],
        dataclasses.replace(gpu, platform_tags=(*gpu.platform_tags, "nvlink")),
    )
    return finalize(dataclasses.replace(capability, devices=devices))


def unavailable(detail: str = "connect failed") -> grpc.aio.AioRpcError:
    return grpc.aio.AioRpcError(grpc.StatusCode.UNAVAILABLE, details=detail)


def invalid_argument(detail: str = "protocol violation") -> grpc.aio.AioRpcError:
    return grpc.aio.AioRpcError(grpc.StatusCode.INVALID_ARGUMENT, details=detail)


# ---------------------------------------------------------------------------
# Startup contract
# ---------------------------------------------------------------------------


def test_missing_master_endpoint_fails_loudly() -> None:
    config = WorkerConfig()  # worker.master defaults to None (spec §26)
    with pytest.raises(WorkerAgentError, match=r"worker\.master\.endpoint"):
        WorkerAgent(config)


def test_instance_id_unique_per_agent_and_uuid() -> None:
    first = make_rig().agent
    second = make_rig().agent
    assert first.instance_id != second.instance_id
    uuid.UUID(first.instance_id)  # raises if malformed


async def test_stopped_before_run_registers_nothing() -> None:
    rig = make_rig()
    rig.agent.request_stop()

    await rig.agent.run()

    assert rig.client.registrations == []
    assert rig.sleeper.delays == []
    assert rig.client.closed is False  # injected clients are caller-owned
    # The inspector lifecycle still ran to completion (start + close).
    assert rig.inspector.started == 1
    assert rig.inspector.closed == 1


# ---------------------------------------------------------------------------
# Happy path: registration then heartbeats
# ---------------------------------------------------------------------------


async def test_register_then_heartbeats_with_monotonic_sequences() -> None:
    rig = make_rig(stop_after=4)  # sleeps: hb, hb, hb, then stop

    await rig.agent.run()

    (registration,) = rig.client.registrations
    assert registration.protocol_version == CONTROL_PROTOCOL_VERSION
    assert registration.instance_id == rig.agent.instance_id
    assert registration.identity.worker_id == "w-1"
    assert registration.initial_state == rig.inspector.returned[0].state

    assert len(rig.client.heartbeats) == 3
    for index, heartbeat in enumerate(rig.client.heartbeats, start=1):
        assert heartbeat.sequence_number == index  # §30: starts at 1
        assert heartbeat.session_id == "session-1"
        assert (
            heartbeat.instance_id
            == registration.instance_id
            == rig.agent.instance_id
        )
        assert heartbeat.capability_revision == registration.capability.capability_revision
        assert heartbeat.worker_reported_at == rig.clock.values[index - 1]


async def test_master_cadence_overrides_configured_interval(caplog) -> None:
    # Configured preference is a nonsensical 99 s; the Master says 250 ms.
    rig = make_rig(interval_ms=250, stop_after=1)

    with caplog.at_level(logging.INFO, logger="worker.master_client"):
        await rig.agent.run()

    assert rig.sleeper.delays == [0.25]
    assert any("overrides" in record.message for record in caplog.records)


async def test_matching_cadence_logs_no_override(caplog) -> None:
    rig = make_rig(interval_ms=500, configured_interval_s=0.5, stop_after=1)

    with caplog.at_level(logging.INFO, logger="worker.master_client"):
        await rig.agent.run()

    assert rig.sleeper.delays == [0.5]
    assert not any("overrides" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# Reconnect with exponential backoff (§27)
# ---------------------------------------------------------------------------


async def test_registration_backoff_doubles_then_resets_on_success() -> None:
    rig = make_rig(
        register_effects=[OSError("refused"), OSError("refused")],
        stop_after=3,
    )

    await rig.agent.run()

    # Backoff 1 s -> 2 s, then the *heartbeat cadence* (0.5 s), proving the
    # delay reset on successful registration instead of continuing to 4 s.
    assert rig.sleeper.delays == [1.0, 2.0, HB_INTERVAL_S]
    assert len(rig.client.registrations) == 3


async def test_backoff_caps_at_max_delay() -> None:
    rig = make_rig(
        register_effects=[OSError("refused")] * 5,
        stop_after=5,
        initial_delay_s=1.0,
        max_delay_s=4.0,
    )

    await rig.agent.run()

    assert rig.sleeper.delays == [1.0, 2.0, 4.0, 4.0, 4.0]
    assert len(rig.client.registrations) == 5


async def test_unavailable_rpc_is_transient_and_retried() -> None:
    rig = make_rig(register_effects=[unavailable()], stop_after=2)

    await rig.agent.run()

    assert rig.sleeper.delays == [1.0, HB_INTERVAL_S]  # backoff, then cadence
    assert len(rig.client.registrations) == 2


async def test_invalid_argument_abort_is_fatal_without_retry() -> None:
    rig = make_rig(register_effects=[invalid_argument("protocol_version mismatch")])

    with pytest.raises(WorkerAgentError, match="protocol violation"):
        await rig.agent.run()

    assert len(rig.client.registrations) == 1
    assert rig.sleeper.delays == []  # a protocol bug must never be retried


async def test_invalid_argument_on_heartbeat_is_fatal() -> None:
    rig = make_rig(
        heartbeat_effects=[None, invalid_argument("malformed state")],
        stop_after=10,
    )

    with pytest.raises(WorkerAgentError, match="heartbeat"):
        await rig.agent.run()

    assert len(rig.client.heartbeats) == 2


async def test_connection_loss_backs_off_and_re_registers_fresh_state() -> None:
    rig = make_rig(
        heartbeat_effects=[None, OSError("connection reset")],
        stop_after=5,
    )

    await rig.agent.run()

    # Cadence, cadence, backoff, cadence, cadence — the loss slept the
    # *initial* delay again because the first registration had reset it.
    assert rig.sleeper.delays == [HB_INTERVAL_S, HB_INTERVAL_S, 1.0, HB_INTERVAL_S, HB_INTERVAL_S]

    # §27: re-register with a fresh inspection, never a cached one.
    assert len(rig.client.registrations) == 2
    second_registration = rig.client.registrations[1]
    assert second_registration.instance_id == rig.agent.instance_id  # stable (§10.2)
    assert second_registration.initial_state == rig.inspector.returned[3].state

    # Sequences restart at 1 under the new session (§30).
    assert [h.sequence_number for h in rig.client.heartbeats] == [1, 2, 1]
    assert rig.client.heartbeats[2].session_id == "session-2"


# ---------------------------------------------------------------------------
# Typed rejection classification (§27/§30: recover, stop, or fail loudly)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reason",
    [RejectionReason.UNKNOWN_WORKER, RejectionReason.REREGISTER_REQUIRED],
)
async def test_lost_registration_re_registers_immediately(reason, caplog) -> None:
    rig = make_rig(
        heartbeat_effects=[
            HeartbeatResponse(accepted=False, detail=reason.value, reason=reason)
        ],
        stop_after=3,
    )

    with caplog.at_level(logging.INFO, logger="worker.master_client"):
        await rig.agent.run()

    # No backoff delay anywhere: rejection goes straight to re-registration.
    assert rig.sleeper.delays == [HB_INTERVAL_S] * 3
    assert len(rig.client.registrations) == 2
    assert any("immediately" in record.message for record in caplog.records)

    fresh = rig.client.heartbeats[1]
    assert fresh.session_id == "session-2"
    assert fresh.sequence_number == 1
    assert fresh.instance_id == rig.agent.instance_id


async def test_stale_session_heartbeat_stops_agent_without_reregistering(caplog) -> None:
    """§27: the superseded Agent exits — it never steals the session back."""
    rig = make_rig(
        heartbeat_effects=[
            HeartbeatResponse(
                accepted=False,
                detail="stale session",
                reason=RejectionReason.STALE_SESSION,
            )
        ],
        stop_after=10,
    )

    with caplog.at_level(logging.ERROR, logger="worker.master_client"):
        await rig.agent.run()

    assert len(rig.client.registrations) == 1  # no ping-pong re-registration
    assert len(rig.client.heartbeats) == 1
    assert any("superseded" in record.message for record in caplog.records)
    assert rig.inspector.closed == 1  # resources released on the way out


@pytest.mark.parametrize(
    "reason",
    [RejectionReason.INSTANCE_MISMATCH, RejectionReason.OUT_OF_ORDER],
)
async def test_protocol_violation_rejection_is_fatal(reason) -> None:
    """§47: an Agent-side protocol violation fails loudly, never retries."""
    rig = make_rig(
        heartbeat_effects=[
            HeartbeatResponse(accepted=False, detail=reason.value, reason=reason)
        ],
        stop_after=10,
    )

    with pytest.raises(WorkerAgentError, match="non-recoverable"):
        await rig.agent.run()

    assert len(rig.client.registrations) == 1
    assert len(rig.client.heartbeats) == 1
    assert rig.inspector.closed == 1  # cleanup even on the fatal path


async def test_rejection_without_reason_is_fatal() -> None:
    """A rejection whose reason never arrived must not be retried blindly."""
    response = HeartbeatResponse.__new__(HeartbeatResponse)
    object.__setattr__(response, "accepted", False)
    object.__setattr__(response, "detail", "master said no")
    object.__setattr__(response, "reason", None)
    rig = make_rig(heartbeat_effects=[response], stop_after=10)

    with pytest.raises(WorkerAgentError, match="unspecified"):
        await rig.agent.run()

    assert len(rig.client.registrations) == 1


# ---------------------------------------------------------------------------
# Capability revision handling (§16)
# ---------------------------------------------------------------------------


async def test_revision_change_sends_update_before_next_heartbeat(caplog) -> None:
    base = make_inspection()
    evolved = make_inspection(capability=evolved_capability())
    rig = make_rig(inspections=[base, base, evolved], stop_after=4)

    with caplog.at_level(logging.INFO, logger="worker.master_client"):
        await rig.agent.run()

    (update,) = rig.client.updates
    assert update.capability == evolved.capability
    # §16: the update carries the state sampled alongside the new capability.
    assert update.state == evolved.state
    assert update.session_id == "session-1"
    assert update.instance_id == rig.agent.instance_id
    assert any("revision changed" in record.message for record in caplog.records)

    revisions = [h.capability_revision for h in rig.client.heartbeats]
    assert revisions == [
        base.capability.capability_revision,
        evolved.capability.capability_revision,
        evolved.capability.capability_revision,  # no repeat update
    ]
    assert len(rig.client.updates) == 1


async def test_rejected_update_re_registers_with_new_capability() -> None:
    base = make_inspection()
    evolved = make_inspection(capability=evolved_capability())
    rig = make_rig(
        inspections=[base, evolved],
        update_effects=[
            UpdateCapabilityResponse(
                accepted=False,
                detail="unknown worker",
                reason=RejectionReason.UNKNOWN_WORKER,
            )
        ],
        stop_after=4,
    )

    await rig.agent.run()

    assert len(rig.client.registrations) == 2
    # The re-registration itself carries the evolved capability, and its
    # acknowledged revision matches — no further updates are needed.
    assert (
        rig.client.registrations[1].capability.capability_revision
        == evolved.capability.capability_revision
    )
    assert len(rig.client.updates) == 1
    assert all(
        h.session_id == "session-2"
        and h.capability_revision == evolved.capability.capability_revision
        for h in rig.client.heartbeats
    )


async def test_stale_session_capability_update_stops_agent() -> None:
    """§27: a superseded Agent stops mid-update too — no re-registration."""
    base = make_inspection()
    evolved = make_inspection(capability=evolved_capability())
    rig = make_rig(
        inspections=[base, evolved],
        update_effects=[
            UpdateCapabilityResponse(
                accepted=False,
                detail="stale session",
                reason=RejectionReason.STALE_SESSION,
            )
        ],
        stop_after=10,
    )

    await rig.agent.run()

    assert len(rig.client.registrations) == 1
    assert len(rig.client.updates) == 1
    assert len(rig.client.heartbeats) == 0  # stopped before the next beat
    assert rig.inspector.closed == 1


# ---------------------------------------------------------------------------
# Profiling endpoint advertisement (Phase 2 spec §41, additive)
# ---------------------------------------------------------------------------


async def test_registration_advertises_profiling_endpoint() -> None:
    rig = make_rig(stop_after=1, profiling_endpoint="10.0.0.5:51100")

    await rig.agent.run()

    (registration,) = rig.client.registrations
    assert registration.profiling_endpoint == "10.0.0.5:51100"
    assert rig.agent.profiling_endpoint == "10.0.0.5:51100"


async def test_registration_without_profiling_endpoint_stays_absent() -> None:
    """Phase 1 behavior is untouched: no endpoint configured → field absent."""
    rig = make_rig(stop_after=1)

    await rig.agent.run()

    (registration,) = rig.client.registrations
    assert registration.profiling_endpoint is None
    assert rig.agent.profiling_endpoint is None


async def test_reregistration_keeps_advertising_profiling_endpoint() -> None:
    """Every reconnect performs a full registration (§27) — the endpoint
    must be re-advertised on the second registration too, never just the first."""
    rig = make_rig(
        stop_after=2,
        register_effects=[unavailable("reset"), None],
        profiling_endpoint="10.0.0.5:51100",
    )

    await rig.agent.run()

    assert len(rig.client.registrations) == 2
    assert all(
        request.profiling_endpoint == "10.0.0.5:51100"
        for request in rig.client.registrations
    )


def test_empty_profiling_endpoint_fails_loudly() -> None:
    config = WorkerConfig(
        worker=WorkerSection(master=WorkerMasterEndpoint(endpoint="master.test:51000"))
    )
    with pytest.raises(WorkerAgentError, match="profiling_endpoint"):
        WorkerAgent(config, profiling_endpoint="")
