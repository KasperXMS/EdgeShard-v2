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
from edgeshard.cluster.state import (
    DeviceAvailability,
    DeviceState,
    MemoryPoolState,
    WorkerState,
    WorkerStatus,
)
from edgeshard.control.master.config import MasterConfig
from edgeshard.control.master.service import MasterService
from edgeshard.protocol.control.mapper import (
    CONTROL_PROTOCOL_VERSION,
    ControlProtocolError,
    HeartbeatRequest,
    RegisterWorkerRequest,
    RejectionReason,
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
    *,
    identity=None,
    capability=None,
    state=None,
    instance_id=None,
    profiling_endpoint: str | None = None,
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
        profiling_endpoint=profiling_endpoint,
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
    """One Worker's slice of the Master's real ClusterSnapshot (§38, P1H)."""
    for worker in service.build_snapshot().workers:
        if worker.identity.worker_id == worker_id:
            return worker
    raise KeyError(worker_id)


# -- state/capability consistency mutators (§38 pre-write gate) --------------


def with_ghost_device(state: WorkerState) -> WorkerState:
    ghost = DeviceState(
        device_id="ghost-device",
        utilization=None,
        temperature_c=None,
        power_w=None,
        availability=DeviceAvailability.UNKNOWN,
        running_runtime_ids=(),
    )
    return dataclasses.replace(state, device_states=(*state.device_states, ghost))


def with_ghost_pool(state: WorkerState) -> WorkerState:
    ghost = MemoryPoolState(memory_pool_id="ghost-pool", available_bytes=None)
    return dataclasses.replace(state, memory_states=(*state.memory_states, ghost))


def with_runtime_on_ghost_device(state: WorkerState) -> WorkerState:
    broken = dataclasses.replace(
        state.runtime_instances[0], device_ids=("ghost-device",)
    )
    return dataclasses.replace(state, runtime_instances=(broken,))


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


@pytest.mark.parametrize(
    "break_state",
    [with_ghost_device, with_ghost_pool, with_runtime_on_ghost_device],
    ids=["ghost-device", "ghost-pool", "runtime-on-ghost-device"],
)
async def test_registration_rejects_state_the_capability_cannot_validate(
    break_state,
) -> None:
    """§38/§47: consistency is checked *before* any Master mutation.

    A state referencing a device/pool the reported capability lacks is a
    clear protocol error at the boundary — nothing is written, so a later
    snapshot can never be the first place the inconsistency surfaces.
    """
    service, _clock = make_service()
    identity = make_worker_identity()
    request = make_register_request(
        identity=identity,
        state=break_state(make_worker_state(identity.worker_id)),
    )

    with pytest.raises(ControlProtocolError, match="state/capability mismatch"):
        await service.register_worker(request)

    assert len(service.registry) == 0
    assert service.sessions.current(identity.worker_id) is None
    assert service.states.get(identity.worker_id) is None


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


async def test_heartbeat_revision_mismatch_rejected_reregister_required() -> None:
    """§30/§38: a heartbeat whose revision the Master does not store is rejected.

    Its state was sampled against a capability the Master cannot
    cross-validate, so it must never be written; the typed reason tells the
    Worker to send UpdateCapability or re-register. The sequence number is
    *not* consumed, so the Worker recovers without a session loss.
    """
    service, _clock = make_service()
    request, session_id = await register(service)
    worker_id = request.identity.worker_id

    evolved = finalize(
        dataclasses.replace(
            request.capability,
            runtime_platforms=request.capability.runtime_platforms[:1],
        )
    )
    evolved_state = make_worker_state(worker_id)
    update = await service.update_capability(
        UpdateCapabilityRequest(
            worker_id=worker_id,
            instance_id=request.instance_id,
            session_id=session_id,
            capability=evolved,
            state=evolved_state,
        )
    )
    assert update.accepted

    # A lagging heartbeat still reporting the pre-update revision.
    response = await service.heartbeat(make_heartbeat(request, session_id, 1))
    assert response.accepted is False
    assert response.reason is RejectionReason.REREGISTER_REQUIRED
    assert "capability revision mismatch" in response.detail

    # The rejected heartbeat wrote nothing: the stored state is still the one
    # the capability update carried atomically.
    stored = service.states.get(worker_id)
    assert stored is not None
    assert stored.state == evolved_state

    # The sequence was not consumed: the Worker catches up and seq 1 is accepted.
    caught_up = HeartbeatRequest(
        worker_id=worker_id,
        instance_id=request.instance_id,
        session_id=session_id,
        sequence_number=1,
        capability_revision=evolved.capability_revision,
        state=evolved_state,
    )
    retry = await service.heartbeat(caught_up)
    assert retry.accepted


async def test_heartbeat_rejects_inconsistent_state_before_writing() -> None:
    """§38: a heartbeat state the stored capability cannot validate is refused.

    The rejection happens before *any* mutation: the stored state keeps the
    previous value and the sequence number is not consumed, so the Worker
    can recover by resending a consistent state under the same sequence.
    """
    service, _clock = make_service()
    request, session_id = await register(service)
    worker_id = request.identity.worker_id

    broken = with_ghost_device(request.initial_state)
    with pytest.raises(ControlProtocolError, match="state/capability mismatch"):
        await service.heartbeat(make_heartbeat(request, session_id, 1, state=broken))

    stored = service.states.get(worker_id)
    assert stored is not None
    assert stored.state == request.initial_state  # nothing was overwritten
    session = service.sessions.current(worker_id)
    assert session is not None
    assert session.last_accepted_sequence == 0  # sequence not consumed

    retry = await service.heartbeat(make_heartbeat(request, session_id, 1))
    assert retry.accepted


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

    new_state = make_worker_state(worker_id)
    response = await service.update_capability(
        UpdateCapabilityRequest(
            worker_id=worker_id,
            instance_id=request.instance_id,
            session_id=session_id,
            capability=evolved,
            state=new_state,
        )
    )

    assert response.accepted
    record = service.registry.get(worker_id)
    assert record.capability == evolved
    assert record.capability_revision != old_revision
    assert record.identity == request.identity  # identity untouched

    # §16/§38: capability and state were replaced atomically — the stored
    # state is the one carried with the update, and any snapshot built now
    # pairs the new capability with a state that cross-validates against it.
    stored = service.states.get(worker_id)
    assert stored is not None
    assert stored.state == new_state
    snapshot = snapshot_of(service, worker_id)
    assert snapshot.capability == evolved
    assert snapshot.state == new_state


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
            state=make_worker_state(identity.worker_id),
        )
    )

    assert response.accepted is False
    assert response.detail == "stale session"
    assert response.reason is RejectionReason.STALE_SESSION
    assert second_session != first_session


async def test_update_capability_unknown_worker_rejected() -> None:
    service, _clock = make_service()
    worker_id = str(uuid.uuid4())
    response = await service.update_capability(
        UpdateCapabilityRequest(
            worker_id=worker_id,
            instance_id=str(uuid.uuid4()),
            session_id="s",
            capability=make_rtx_capability(),
            state=make_worker_state(worker_id),
        )
    )
    assert response.accepted is False
    assert response.detail == "unknown worker"
    assert response.reason is RejectionReason.UNKNOWN_WORKER


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
                state=make_worker_state(request.identity.worker_id),
            )
        )


async def test_update_capability_rejects_inconsistent_state_before_writing() -> None:
    """§38: the atomic pair never lands when its state cannot cross-validate.

    Neither the new capability nor the inconsistent state may reach storage:
    the previously stored, mutually-consistent pair stays untouched.
    """
    service, _clock = make_service()
    request, session_id = await register(service)
    worker_id = request.identity.worker_id

    evolved = finalize(
        dataclasses.replace(
            request.capability,
            runtime_platforms=request.capability.runtime_platforms[:1],
        )
    )
    broken_state = with_ghost_device(make_worker_state(worker_id))
    with pytest.raises(ControlProtocolError, match="state/capability mismatch"):
        await service.update_capability(
            UpdateCapabilityRequest(
                worker_id=worker_id,
                instance_id=request.instance_id,
                session_id=session_id,
                capability=evolved,
                state=broken_state,
            )
        )

    record = service.registry.get(worker_id)
    assert record.capability == request.capability  # capability untouched
    stored = service.states.get(worker_id)
    assert stored is not None
    assert stored.state == request.initial_state  # state untouched
    # The stored pair still cross-validates: snapshots keep building cleanly.
    snapshot = snapshot_of(service, worker_id)
    assert snapshot.capability == request.capability
    assert snapshot.state == request.initial_state


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


# -- profiling endpoint advertisement (Phase 2 spec §41, additive) -----------


async def test_registration_stores_profiling_endpoint() -> None:
    service, _clock = make_service()
    request = make_register_request(profiling_endpoint="10.0.0.5:51100")

    await service.register_worker(request)

    record = service.registry.get(request.identity.worker_id)
    assert record.profiling_endpoint == "10.0.0.5:51100"


async def test_registration_without_profiling_endpoint_stores_none() -> None:
    """Phase 1 registrations are untouched by the additive field."""
    service, _clock = make_service()
    request, _session = await register(service)
    record = service.registry.get(request.identity.worker_id)
    assert record.profiling_endpoint is None


async def test_reregistration_replaces_profiling_endpoint() -> None:
    """A restarted Worker that stopped hosting profiling clears the stale
    address: registration is the endpoint's only source of truth (§41)."""
    service, _clock = make_service()
    identity = make_worker_identity()
    hosting = make_register_request(
        identity=identity, profiling_endpoint="10.0.0.5:51100"
    )
    await service.register_worker(hosting)

    plain = make_register_request(
        identity=identity, state=make_worker_state(identity.worker_id)
    )
    await service.register_worker(plain)

    assert service.registry.get(identity.worker_id).profiling_endpoint is None


async def test_capability_update_preserves_profiling_endpoint() -> None:
    """Only registration changes the endpoint; capability updates carry no
    endpoint field and must keep the advertised one (§41)."""
    service, _clock = make_service()
    request = make_register_request(profiling_endpoint="10.0.0.5:51100")
    response = await service.register_worker(request)
    worker_id = request.identity.worker_id

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
    update = await service.update_capability(
        UpdateCapabilityRequest(
            worker_id=worker_id,
            instance_id=request.instance_id,
            session_id=response.session_id,
            capability=evolved,
            state=make_worker_state(worker_id),
        )
    )

    assert update.accepted
    record = service.registry.get(worker_id)
    assert record.capability == evolved
    assert record.profiling_endpoint == "10.0.0.5:51100"
