"""SnapshotBuilder tests (Phase 1 spec §38-39, §45, §52 Test B, §56 P1H).

The builder folds the four Master state components into one immutable
``ClusterSnapshot``. These tests pin the §38 contract through the real
``MasterService`` facade (injected fake clock, zero real sleeping):

* a snapshot is a true point-in-time copy — a heartbeat that lands after
  ``build`` returns never mutates it (§38, §51);
* ``build`` is synchronous, so against a stream of heartbeats every
  snapshot reflects exactly its instant and stays frozen afterward
  (the "concurrent heartbeat vs snapshot" case, §56 P1H);
* heterogeneous RTX + Jetson Workers coexist in one snapshot, both ONLINE,
  with independently-valid topologies (§52 Test B);
* OFFLINE Workers remain present (§32); facts only (§39); and a debug
  rendering exists for ``master serve`` (§45).
"""

from __future__ import annotations

import dataclasses
import uuid
from datetime import UTC, datetime, timedelta

import pytest

from edgeshard.cluster.snapshot import ClusterSnapshot, WorkerSnapshot
from edgeshard.cluster.state import WorkerState, WorkerStatus
from edgeshard.control.master.config import MasterConfig
from edgeshard.control.master.liveness import LivenessManager
from edgeshard.control.master.registry import WorkerRegistry
from edgeshard.control.master.service import MasterService
from edgeshard.control.master.sessions import SessionManager
from edgeshard.control.master.snapshot import SnapshotBuilder, format_snapshot
from edgeshard.control.master.state_store import StateStore
from edgeshard.protocol.control.mapper import (
    CONTROL_PROTOCOL_VERSION,
    HeartbeatRequest,
    RegisterWorkerRequest,
)
from factories import (
    RTX_GPU_DEVICE_ID,
    RTX_GPU_POOL_ID,
    RTX_HOST_POOL_ID,
    make_jetson_capability,
    make_rtx_capability,
    make_worker_identity,
    make_worker_state,
)

JETSON_CPU_DEVICE_ID = "cpu-system"
JETSON_GPU_DEVICE_ID = "gpu-system"
JETSON_POOL_ID = "system-memory"

WALL_BASE = datetime(2026, 9, 5, 12, 0, 0, tzinfo=UTC)


class FakeClock:
    """One source for both Master clocks; wall advances with monotonic (§31)."""

    def __init__(self) -> None:
        self.now = 1_000.0

    def monotonic(self) -> float:
        return self.now

    def wall(self) -> datetime:
        return WALL_BASE + timedelta(seconds=self.now - 1_000.0)

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


def with_utilization(state: WorkerState, utilization: float) -> WorkerState:
    return dataclasses.replace(
        state,
        device_states=tuple(
            dataclasses.replace(device, utilization=utilization)
            for device in state.device_states
        ),
    )


def jetson_register_request() -> RegisterWorkerRequest:
    identity = make_worker_identity()
    return make_register_request(
        identity=identity,
        capability=make_jetson_capability(),
        state=make_worker_state(
            identity.worker_id,
            device_ids=(JETSON_CPU_DEVICE_ID, JETSON_GPU_DEVICE_ID),
            pool_ids=(JETSON_POOL_ID,),
        ),
    )


def only_worker(snapshot: ClusterSnapshot) -> WorkerSnapshot:
    (worker,) = snapshot.workers
    return worker


# ---------------------------------------------------------------------------
# Basic shape
# ---------------------------------------------------------------------------


def test_empty_registry_builds_empty_snapshot() -> None:
    service, clock = make_service()

    snapshot = service.build_snapshot()

    assert snapshot.workers == ()
    assert snapshot.snapshot_id
    assert snapshot.created_at == clock.wall()


def test_created_at_and_id_come_from_injected_factories() -> None:
    clock = FakeClock()
    service = MasterService(
        None,
        monotonic=clock.monotonic,
        wall=clock.wall,
        snapshot_factory=lambda: "snapshot-fixed",
    )

    snapshot = service.build_snapshot()

    assert snapshot.snapshot_id == "snapshot-fixed"
    assert snapshot.created_at == WALL_BASE


def test_successive_snapshots_have_distinct_ids() -> None:
    service, _clock = make_service()
    first = service.build_snapshot()
    second = service.build_snapshot()
    assert first.snapshot_id != second.snapshot_id


async def test_single_worker_snapshot_fields() -> None:
    service, clock = make_service()
    request, session_id = await register(service)
    worker_id = request.identity.worker_id

    worker = only_worker(service.build_snapshot())

    assert worker.identity.worker_id == worker_id
    assert worker.capability.capability_revision == request.capability.capability_revision
    assert worker.state == request.initial_state
    assert worker.status is WorkerStatus.ONLINE
    assert worker.session_id == session_id
    assert worker.last_seen_at == clock.wall()


async def test_snapshot_workers_sorted_by_worker_id() -> None:
    service, _clock = make_service()
    ids = []
    for _ in range(3):
        request, _ = await register(service)
        ids.append(request.identity.worker_id)

    snapshot = service.build_snapshot()

    snapshot_ids = [worker.identity.worker_id for worker in snapshot.workers]
    assert snapshot_ids == sorted(ids)
    assert len(set(snapshot_ids)) == 3


# ---------------------------------------------------------------------------
# Heterogeneous cluster (§52 Test B)
# ---------------------------------------------------------------------------


async def test_rtx_and_jetson_coexist_in_one_snapshot() -> None:
    service, _clock = make_service()
    rtx_request, rtx_session = await register(service)
    jetson_request = jetson_register_request()
    jetson_response = await service.register_worker(jetson_request)
    # Both heartbeat independently.
    assert (
        await service.heartbeat(make_heartbeat(rtx_request, rtx_session, 1))
    ).accepted
    assert (
        await service.heartbeat(
            make_heartbeat(jetson_request, jetson_response.session_id, 1)
        )
    ).accepted

    snapshot = service.build_snapshot()
    assert len(snapshot.workers) == 2
    by_id = {worker.identity.worker_id: worker for worker in snapshot.workers}
    rtx = by_id[rtx_request.identity.worker_id]
    jetson = by_id[jetson_request.identity.worker_id]

    # Both ONLINE at the same instant.
    assert rtx.status is WorkerStatus.ONLINE
    assert jetson.status is WorkerStatus.ONLINE

    # Independent topologies cross-validate against each snapshot's own state.
    rtx_pools = {pool.memory_pool_id for pool in rtx.capability.memory_pools}
    assert rtx_pools == {RTX_HOST_POOL_ID, RTX_GPU_POOL_ID}
    assert any(device.identity.device_id == RTX_GPU_DEVICE_ID for device in rtx.capability.devices)
    jetson_pools = {pool.memory_pool_id for pool in jetson.capability.memory_pools}
    assert jetson_pools == {JETSON_POOL_ID}
    assert {device.identity.device_id for device in jetson.capability.devices} == {
        JETSON_CPU_DEVICE_ID,
        JETSON_GPU_DEVICE_ID,
    }
    # Each snapshot state references only its own capability's devices/pools
    # (WorkerSnapshot.__post_init__ already cross-validates; assert counts too).
    assert len(rtx.state.memory_states) == 1
    assert len(jetson.state.memory_states) == 1


# ---------------------------------------------------------------------------
# Immutability and concurrency (§38, §51, §56 P1H)
# ---------------------------------------------------------------------------


async def test_snapshot_immutable_across_a_later_heartbeat() -> None:
    service, clock = make_service()
    request, session_id = await register(service)

    snapshot_a = service.build_snapshot()
    assert all(device.utilization == 21.0 for device in only_worker(snapshot_a).state.device_states)
    assert snapshot_a.created_at == WALL_BASE

    clock.advance(1.0)
    busier = with_utilization(request.initial_state, 77.0)
    assert (
        await service.heartbeat(make_heartbeat(request, session_id, 1, state=busier))
    ).accepted

    # Snapshot A is a frozen point-in-time copy: the heartbeat did not touch it.
    assert all(device.utilization == 21.0 for device in only_worker(snapshot_a).state.device_states)
    assert snapshot_a.created_at == WALL_BASE

    snapshot_b = service.build_snapshot()
    assert all(device.utilization == 77.0 for device in only_worker(snapshot_b).state.device_states)
    assert snapshot_b.created_at == clock.wall()


async def test_snapshot_atomic_against_a_stream_of_heartbeats() -> None:
    """The §56 P1H concurrency case: build() never interleaves with a heartbeat."""
    service, clock = make_service()
    request, session_id = await register(service)

    captured: list[tuple[float, ClusterSnapshot]] = []
    for sequence in range(1, 6):
        utilization = float(sequence * 10)
        clock.advance(0.5)
        response = await service.heartbeat(
            make_heartbeat(
                request,
                session_id,
                sequence,
                state=with_utilization(request.initial_state, utilization),
            )
        )
        assert response.accepted
        snapshot = service.build_snapshot()
        captured.append((utilization, snapshot))
        # The snapshot taken right after this heartbeat reflects exactly it.
        worker = only_worker(snapshot)
        assert all(device.utilization == utilization for device in worker.state.device_states)

    # Every earlier snapshot still holds its own value: later heartbeats never
    # reached back into an already-built snapshot (§38).
    for utilization, snapshot in captured:
        worker = only_worker(snapshot)
        assert all(device.utilization == utilization for device in worker.state.device_states)
    ids = [snapshot.snapshot_id for _, snapshot in captured]
    assert len(set(ids)) == len(ids)


# ---------------------------------------------------------------------------
# Liveness and re-registration reflected in snapshots (§32, §35)
# ---------------------------------------------------------------------------


async def test_snapshot_reflects_liveness_and_keeps_offline_workers() -> None:
    config = MasterConfig(
        heartbeat_interval_ms=100,
        suspect_after_ms=200,
        offline_after_ms=400,
        liveness_tick_ms=5,
    )
    service, clock = make_service(config)
    await register(service)

    assert only_worker(service.build_snapshot()).status is WorkerStatus.ONLINE

    clock.advance(0.201)  # beyond suspect_after, within offline_after
    assert only_worker(service.build_snapshot()).status is WorkerStatus.SUSPECT

    clock.advance(0.2)  # beyond offline_after
    snapshot = service.build_snapshot()
    assert only_worker(snapshot).status is WorkerStatus.OFFLINE
    # §32: OFFLINE is a status, not an eviction — the Worker stays present.
    assert len(snapshot.workers) == 1


def test_liveness_classified_against_a_single_clock_read() -> None:
    """§38: one snapshot never mixes two monotonic instants.

    The injected clock advances 5 s on *every* read and both Workers were
    last seen at its first reading. With a single ``now`` for the whole
    build both are ONLINE; a per-Worker clock read would classify the
    second Worker 5 s older — past ``suspect_after`` — and mix two instants
    in one supposedly point-in-time snapshot.
    """

    class AdvancingClock:
        def __init__(self, start: float, step: float) -> None:
            self._next = start
            self._step = step
            self.reads = 0

        def __call__(self) -> float:
            self.reads += 1
            value = self._next
            self._next += self._step
            return value

    config = MasterConfig(
        heartbeat_interval_ms=1_000,
        suspect_after_ms=4_000,
        offline_after_ms=8_000,
        liveness_tick_ms=5,
    )
    registry = WorkerRegistry()
    sessions = SessionManager()
    states = StateStore()
    liveness = LivenessManager(states, config)

    clock = AdvancingClock(start=1_000.0, step=5.0)
    for _ in range(2):
        identity = make_worker_identity()
        registry.upsert(identity, make_rtx_capability())
        states.record(
            identity.worker_id,
            make_worker_state(identity.worker_id),
            monotonic=1_000.0,  # both last seen at the clock's first reading
            wall=WALL_BASE,
        )

    builder = SnapshotBuilder(
        registry, sessions, states, liveness, monotonic=clock, wall=lambda: WALL_BASE
    )
    snapshot = builder.build()

    assert clock.reads == 1  # exactly one monotonic read for the whole snapshot
    assert len(snapshot.workers) == 2
    assert all(worker.status is WorkerStatus.ONLINE for worker in snapshot.workers)


async def test_snapshot_reflects_new_session_after_re_registration() -> None:
    service, _clock = make_service()
    request, first_session = await register(service)
    # Same worker_id, new instance/process re-registers.
    re_request = make_register_request(
        identity=request.identity,
        capability=request.capability,
        state=request.initial_state,
    )
    re_response = await service.register_worker(re_request)

    worker = only_worker(service.build_snapshot())

    assert re_response.session_id != first_session
    assert worker.session_id == re_response.session_id


# ---------------------------------------------------------------------------
# Facts only (§39) and debug rendering (§45)
# ---------------------------------------------------------------------------


def test_worker_snapshot_carries_facts_only() -> None:
    fields = {field.name for field in dataclasses.fields(WorkerSnapshot)}
    assert fields == {
        "identity",
        "capability",
        "state",
        "status",
        "session_id",
        "last_seen_at",
    }
    # No profiling/scheduling outputs may ever appear on a snapshot (§39).
    forbidden = {"fits", "estimate", "prediction", "recommendation", "placement", "score"}
    assert not (fields & forbidden)


async def test_format_snapshot_debug_representation() -> None:
    service, _clock = make_service()
    await register(service)
    await service.register_worker(jetson_register_request())

    snapshot = service.build_snapshot()
    text = format_snapshot(snapshot)

    assert snapshot.snapshot_id in text
    assert f"workers={len(snapshot.workers)}" in text
    assert "devices=" in text and "pools=" in text and "session_id=" in text
    for worker in snapshot.workers:
        assert worker.identity.worker_id in text
        assert worker.status.value in text


async def test_service_facade_delegates_to_builder() -> None:
    service, _clock = make_service()
    await register(service)

    via_facade = service.build_snapshot()
    via_builder = service.snapshot_builder.build()

    assert via_facade.workers == via_builder.workers
    assert via_facade.snapshot_id != via_builder.snapshot_id  # each build is fresh


# ---------------------------------------------------------------------------
# Defensive invariant (§47): registry entry without stored state
# ---------------------------------------------------------------------------


def test_builder_rejects_registry_worker_without_state() -> None:
    registry = WorkerRegistry()
    sessions = SessionManager()
    states = StateStore()
    liveness = LivenessManager(states, MasterConfig())
    registry.upsert(make_worker_identity(), make_rtx_capability())
    # No states.record: registration would always have recorded one, so this
    # is an internal inconsistency the builder must refuse to paper over.
    builder = SnapshotBuilder(registry, sessions, states, liveness)

    with pytest.raises(ValueError, match="no stored state"):
        builder.build()
