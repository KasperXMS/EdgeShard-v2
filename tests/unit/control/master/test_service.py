"""MasterService tests — the §29-30 acceptance rules on real Master state.

All timing uses an injected fake clock (§31, §37): registration, heartbeat
ordering, stale-session rejection, capability updates, liveness transitions,
and snapshot immutability (§51) evaluate without real-time sleeping.
"""

from __future__ import annotations

import dataclasses
import uuid
from datetime import UTC, datetime, timedelta

import pytest

from edgeshard.cluster.capability import compute_capability_revision
from edgeshard.cluster.snapshot import WorkerSnapshot
from edgeshard.cluster.state import WorkerStatus
from edgeshard.control.master.config import MasterConfig
from edgeshard.control.master.service import MasterService
from edgeshard.protocol.control.mapper import (
    CONTROL_PROTOCOL_VERSION,
    ControlProtocolError,
    HeartbeatRequest,
    RegisterWorkerRequest,
    UpdateCapabilityRequest,
)
from factories import (
    finalize,
    make_rtx_capability,
    make_worker_identity,
    make_worker_state,
)


class FakeClock:
    """One source for both Master clocks; wall advances with monotonic."""

    def __init__(self) -> None:
        self.now = 1_000.0
        self.wall_base = datetime(2026, 9, 5, 12, 0, 0, tzinfo=UTC)

    def monotonic(self) -> float:
        return self.now

    def wall(self) -> datetime:
        return self.wall_base + timedelta(seconds=self.now - 1_000.0)

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_service(config: MasterConfig | None = None) -> tuple[MasterService, FakeClock]:
    clock = FakeClock()
    service = MasterService(config, monotonic=clock.monotonic, wall=clock.wall)
    return service, clock


def make_register_request(
    *, identity=None, capability=None, state=None, instance_id=None
) -> RegisterWorkerRequest:
    identity = identity or make_worker_identity()
    capability = capability or make_rtx_capability()
    state = state or make_worker_state(identity.worker_id)
    return RegisterWorkerRequest(
        protocol_version=CONTROL_PROTOCOL_VERSION,
        instance_id=instance_id or str(uuid.uuid4()),
        identity=identity,
        capability=capability,
        initial_state=state,
    )


def make_heartbeat(
    request: RegisterWorkerRequest, session_id: str, sequence: int, *, state=None
) -> HeartbeatRequest:
    return HeartbeatRequest(
        worker_id=request.identity.worker_id,
        instance_id=request.instance_id,
        session_id=session_id,
        sequence_number=sequence,
        capability_revision=request.capability.capability_revision,
        state=state or request.initial_state,
    )


async def register(service: MasterService, **kwargs) -> tuple[RegisterWorkerRequest, str]:
    request = make_register_request(**kwargs)
    response = await service.register_worker(request)
    return request, response.session_id


def snapshot_of(service: MasterService, worker_id: str) -> WorkerSnapshot:
    """Assemble the Master's knowledge as a WorkerSnapshot (§38; builder lands in P1H)."""
    record = service.registry.get(worker_id)
    session = service.sessions.current(worker_id)
    stored = service.states.get(worker_id)
    assert stored is not None
    return WorkerSnapshot(
        identity=record.identity,
        capability=record.capability,
        state=stored.state,
        status=service.worker_status(worker_id),
        session_id=session.session_id if session else None,
        last_seen_at=stored.received_wall,
    )


# -- registration (§29) -----------------------------------------------------


async def test_registration_stores_everything_and_marks_online() -> None:
    service, _clock = make_service()
    request = make_register_request()

    response = await service.register_worker(request)

    assert response.session_id
    assert response.heartbeat_interval_ms == MasterConfig().heartbeat_interval_ms
    assert response.server_protocol_version == CONTROL_PROTOCOL_VERSION

    worker_id = request.identity.worker_id
    assert len(service.registry) == 1
    record = service.registry.get(worker_id)
    assert record.identity == request.identity
    assert record.capability == request.capability

    session = service.sessions.current(worker_id)
    assert session is not None
    assert session.session_id == response.session_id
    assert session.instance_id == request.instance_id
    assert session.last_accepted_sequence == 0

    stored = service.states.get(worker_id)
    assert stored is not None
    assert stored.state == request.initial_state

    # §29: registration marks the Worker ONLINE.
    assert service.worker_status(worker_id) is WorkerStatus.ONLINE


async def test_registration_response_interval_follows_config() -> None:
    service, _clock = make_service(
        MasterConfig(heartbeat_interval_ms=250, suspect_after_ms=500, offline_after_ms=900)
    )
    response = await service.register_worker(make_register_request())
    assert response.heartbeat_interval_ms == 250


async def test_registration_rejects_tampered_capability_revision() -> None:
    """§16/§47: the reported revision must fingerprint the reported capability."""
    service, _clock = make_service()
    tampered = dataclasses.replace(make_rtx_capability(), capability_revision="bogus")
    request = make_register_request(capability=tampered)

    with pytest.raises(ControlProtocolError, match="capability_revision mismatch"):
        await service.register_worker(request)
    assert len(service.registry) == 0


# -- sessions (§35, §51 Master sessions) ------------------------------------


async def test_second_registration_invalidates_old_session() -> None:
    """§51: registration creates session; second registration invalidates old."""
    service, _clock = make_service()
    identity = make_worker_identity()
    first_request, first_session = await register(service, identity=identity)
    second_request, second_session = await register(
        service, identity=identity, state=make_worker_state(identity.worker_id)
    )

    assert second_session != first_session
    # §52 Test C: still the same registry entry.
    assert len(service.registry) == 1
    assert second_request.instance_id != first_request.instance_id

    old = await service.heartbeat(make_heartbeat(first_request, first_session, 1))
    assert old.accepted is False
    assert old.detail == "stale session"

    fresh = await service.heartbeat(make_heartbeat(second_request, second_session, 1))
    assert fresh.accepted


# -- heartbeats (§30, §51 Heartbeat order) ----------------------------------


async def test_heartbeat_order_per_spec_51() -> None:
    service, _clock = make_service()
    request, session_id = await register(service)

    async def beat(sequence: int):
        return await service.heartbeat(make_heartbeat(request, session_id, sequence))

    assert (await beat(5)).accepted  # sequences may jump forward
    rejected_old = await beat(4)
    assert rejected_old.accepted is False
    assert rejected_old.detail == "out-of-order heartbeat"
    rejected_dup = await beat(5)
    assert rejected_dup.accepted is False  # duplicate of the accepted 5
    assert (await beat(6)).accepted

    session = service.sessions.current(request.identity.worker_id)
    assert session is not None
    assert session.last_accepted_sequence == 6


async def test_rejected_heartbeat_never_overwrites_state() -> None:
    """§30/§52 Test D: send 1, 2, 4, late 3 — state must remain sequence 4's."""
    service, _clock = make_service()
    request, session_id = await register(service)
    worker_id = request.identity.worker_id

    def state_with(utilization: float):
        return dataclasses.replace(
            request.initial_state,
            device_states=tuple(
                dataclasses.replace(s, utilization=utilization)
                for s in request.initial_state.device_states
            ),
        )

    for sequence, utilization in ((1, 10.0), (2, 20.0), (4, 40.0)):
        response = await service.heartbeat(
            make_heartbeat(request, session_id, sequence, state=state_with(utilization))
        )
        assert response.accepted

    late = await service.heartbeat(
        make_heartbeat(request, session_id, 3, state=state_with(30.0))
    )
    assert late.accepted is False
    assert late.detail == "out-of-order heartbeat"

    stored = service.states.get(worker_id)
    assert stored is not None
    assert all(s.utilization == 40.0 for s in stored.state.device_states)


async def test_heartbeat_unknown_worker_rejected() -> None:
    service, _clock = make_service()
    request = make_register_request()

    response = await service.heartbeat(make_heartbeat(request, "some-session", 1))

    assert response.accepted is False
    assert response.detail == "unknown worker"


async def test_heartbeat_instance_mismatch_rejected() -> None:
    service, _clock = make_service()
    request, session_id = await register(service)
    spoofed = dataclasses.replace(request, instance_id=str(uuid.uuid4()))

    response = await service.heartbeat(make_heartbeat(spoofed, session_id, 1))

    assert response.accepted is False
    assert response.detail == "instance mismatch"


async def test_heartbeat_updates_last_seen_timestamps() -> None:
    """§31: Master keeps its own monotonic + wall receive timestamps."""
    service, clock = make_service()
    request, session_id = await register(service)
    worker_id = request.identity.worker_id

    clock.advance(3.0)
    await service.heartbeat(make_heartbeat(request, session_id, 1))

    stored = service.states.get(worker_id)
    assert stored is not None
    assert stored.received_monotonic == clock.monotonic()
    assert stored.received_wall == clock.wall()
    assert service.worker_status(worker_id) is WorkerStatus.ONLINE


async def test_capability_drift_warns_but_accepts(caplog) -> None:
    """A heartbeat revision differing from the stored capability is drift, not rejection."""
    service, _clock = make_service()
    request, session_id = await register(service)

    evolved = finalize(
        dataclasses.replace(
            request.capability,
            runtime_platforms=request.capability.runtime_platforms[:1],
        )
    )
    await service.update_capability(
        UpdateCapabilityRequest(
            worker_id=request.identity.worker_id,
            instance_id=request.instance_id,
            session_id=session_id,
            capability=evolved,
        )
    )

    with caplog.at_level("WARNING", logger="master.service"):
        # The heartbeat still reports the pre-update revision.
        response = await service.heartbeat(make_heartbeat(request, session_id, 1))

    assert response.accepted
    assert any("capability drift" in record.getMessage() for record in caplog.records)


# -- capability updates (§16) -----------------------------------------------


async def test_update_capability_replaces_stored_capability() -> None:
    service, _clock = make_service()
    request, session_id = await register(service)
    worker_id = request.identity.worker_id
    old_revision = request.capability.capability_revision

    gpu = request.capability.devices[1]
    evolved = finalize(
        dataclasses.replace(
            request.capability,
            devices=(
                request.capability.devices[0],
                dataclasses.replace(gpu, platform_tags=(*gpu.platform_tags, "nvlink")),
            ),
        )
    )
    assert compute_capability_revision(evolved) != old_revision

    response = await service.update_capability(
        UpdateCapabilityRequest(
            worker_id=worker_id,
            instance_id=request.instance_id,
            session_id=session_id,
            capability=evolved,
        )
    )

    assert response.accepted
    record = service.registry.get(worker_id)
    assert record.capability == evolved
    assert record.capability_revision != old_revision
    assert record.identity == request.identity  # identity untouched


async def test_update_capability_stale_session_rejected() -> None:
    service, _clock = make_service()
    identity = make_worker_identity()
    first_request, first_session = await register(service, identity=identity)
    _second, second_session = await register(
        service, identity=identity, state=make_worker_state(identity.worker_id)
    )

    response = await service.update_capability(
        UpdateCapabilityRequest(
            worker_id=identity.worker_id,
            instance_id=first_request.instance_id,
            session_id=first_session,
            capability=make_rtx_capability(),
        )
    )

    assert response.accepted is False
    assert response.detail == "stale session"
    assert second_session != first_session


async def test_update_capability_unknown_worker_rejected() -> None:
    service, _clock = make_service()
    response = await service.update_capability(
        UpdateCapabilityRequest(
            worker_id=str(uuid.uuid4()),
            instance_id=str(uuid.uuid4()),
            session_id="s",
            capability=make_rtx_capability(),
        )
    )
    assert response.accepted is False
    assert response.detail == "unknown worker"


async def test_update_capability_rejects_tampered_revision() -> None:
    service, _clock = make_service()
    request, session_id = await register(service)

    with pytest.raises(ControlProtocolError, match="capability_revision mismatch"):
        await service.update_capability(
            UpdateCapabilityRequest(
                worker_id=request.identity.worker_id,
                instance_id=request.instance_id,
                session_id=session_id,
                capability=dataclasses.replace(make_rtx_capability(), capability_revision="x"),
            )
        )


# -- liveness interplay (§32) ------------------------------------------------


async def test_offline_worker_remains_in_registry() -> None:
    """§32: OFFLINE Workers are never deleted automatically."""
    service, clock = make_service()
    request, session_id = await register(service)
    worker_id = request.identity.worker_id

    clock.advance(25.0)  # beyond the default 20 s OFFLINE threshold
    assert service.worker_status(worker_id) is WorkerStatus.OFFLINE
    assert [record.worker_id for record in service.registry.list_workers()] == [worker_id]

    # The Worker returns: a heartbeat on the still-current session revives it.
    response = await service.heartbeat(make_heartbeat(request, session_id, 1))
    assert response.accepted
    assert service.worker_status(worker_id) is WorkerStatus.ONLINE


async def test_worker_status_unknown_raises() -> None:
    service, _clock = make_service()
    with pytest.raises(KeyError):
        service.worker_status("ghost")


# -- snapshot immutability (§51) ---------------------------------------------


async def test_snapshot_immutability_across_heartbeats() -> None:
    """§51: snapshot A, heartbeat B → A unchanged, B sees the new state."""
    service, clock = make_service()
    request, session_id = await register(service)
    worker_id = request.identity.worker_id

    snapshot_a = snapshot_of(service, worker_id)
    assert all(s.utilization == 21.0 for s in snapshot_a.state.device_states)

    busier = dataclasses.replace(
        request.initial_state,
        device_states=tuple(
            dataclasses.replace(s, utilization=77.0)
            for s in request.initial_state.device_states
        ),
    )
    clock.advance(1.0)
    response = await service.heartbeat(
        make_heartbeat(request, session_id, 1, state=busier)
    )
    assert response.accepted

    # Snapshot A is a frozen point-in-time copy: the heartbeat did not touch it.
    assert all(s.utilization == 21.0 for s in snapshot_a.state.device_states)
    assert snapshot_a.last_seen_at == datetime(2026, 9, 5, 12, 0, 0, tzinfo=UTC)

    snapshot_b = snapshot_of(service, worker_id)
    assert all(s.utilization == 77.0 for s in snapshot_b.state.device_states)
    assert snapshot_b.last_seen_at == clock.wall()
    assert snapshot_b.status is WorkerStatus.ONLINE


# -- lifecycle ---------------------------------------------------------------


async def test_service_start_stop_context_manager() -> None:
    service, _clock = make_service(MasterConfig(liveness_tick_ms=5))
    async with service:
        request, session_id = await register(service)
        response = await service.heartbeat(make_heartbeat(request, session_id, 1))
        assert response.accepted
