"""End-to-end control transport against the real MasterService (spec §52).

RTX-like and Jetson-like Workers (the fixture platforms of spec §52 Test B)
register, heartbeat, and update capabilities over a real ``grpc.aio``
channel served by the production ``MasterService`` — the P1E fake-Master
scaffolding is gone. Every payload the Master decodes is cross-validated
as a ``WorkerSnapshot`` (spec §38), so the wire provably preserves
referential integrity of both memory topologies.

Covers the §52 shapes end to end: Test A (Master sees exactly the
registered Workers), Test B (independent heterogeneous Workers), Test C
(re-registration, stale-session rejection), Test D (out-of-order heartbeat
never overwrites newer state), and Test E (ONLINE → SUSPECT → OFFLINE with
shortened test-config thresholds and an injected fake clock — no real-time
sleeping, spec §37).
"""

from __future__ import annotations

import dataclasses
import uuid
from datetime import UTC, datetime, timedelta

from edgeshard.cluster.capability import WorkerCapability
from edgeshard.cluster.snapshot import WorkerSnapshot
from edgeshard.cluster.state import WorkerStatus
from edgeshard.control.master.config import MasterConfig
from edgeshard.control.master.service import MasterService
from edgeshard.protocol.control.grpc_client import WorkerRegistryClient
from edgeshard.protocol.control.grpc_server import start_control_server
from edgeshard.protocol.control.mapper import (
    CONTROL_PROTOCOL_VERSION,
    HeartbeatRequest,
    RegisterWorkerRequest,
    UpdateCapabilityRequest,
)
from factories import (
    RTX_GPU_DEVICE_ID,
    finalize,
    make_jetson_capability,
    make_rtx_capability,
    make_worker_identity,
    make_worker_state,
)

HOST = "127.0.0.1"

JETSON_CPU_DEVICE_ID = "cpu-system"
JETSON_GPU_DEVICE_ID = "gpu-system"
JETSON_POOL_ID = "system-memory"

# Shortened thresholds so the offline lifecycle evaluates instantly (§52 Test E).
TEST_CONFIG = MasterConfig(
    heartbeat_interval_ms=100,
    suspect_after_ms=200,
    offline_after_ms=400,
    liveness_tick_ms=5,
)


class FakeClock:
    """Injected Master clock (§31): wall time advances with monotonic time."""

    def __init__(self) -> None:
        self.now = 1_000.0
        self.wall_base = datetime(2026, 9, 5, 12, 0, 0, tzinfo=UTC)

    def monotonic(self) -> float:
        return self.now

    def wall(self) -> datetime:
        return self.wall_base + timedelta(seconds=self.now - 1_000.0)

    def advance(self, seconds: float) -> None:
        self.now += seconds


def snapshot_of(service: MasterService, worker_id: str) -> WorkerSnapshot:
    """One Worker's slice of the Master's real ClusterSnapshot (§38, P1H)."""
    for worker in service.build_snapshot().workers:
        if worker.identity.worker_id == worker_id:
            return worker
    raise KeyError(worker_id)


def evolve_rtx_capability() -> WorkerCapability:
    """A plausible rare capability change (spec §16): one new platform tag."""
    capability = make_rtx_capability()
    gpu = capability.devices[1]
    devices = (
        capability.devices[0],
        dataclasses.replace(gpu, platform_tags=(*gpu.platform_tags, "nvlink")),
    )
    return finalize(dataclasses.replace(capability, devices=devices))


async def test_rtx_and_jetson_workers_share_one_master() -> None:
    """§52 Tests A-E against the production Master state."""
    clock = FakeClock()
    service = MasterService(TEST_CONFIG, monotonic=clock.monotonic, wall=clock.wall)
    server, port = await start_control_server(service, host=HOST, port=0)
    async with service, WorkerRegistryClient(f"{HOST}:{port}") as client:
        try:
            # --- Registration (§52 Test A/B shapes) --------------------------
            rtx_identity = make_worker_identity()
            rtx_state = make_worker_state(rtx_identity.worker_id)
            rtx_response = await client.register_worker(
                RegisterWorkerRequest(
                    protocol_version=CONTROL_PROTOCOL_VERSION,
                    instance_id=str(uuid.uuid4()),
                    identity=rtx_identity,
                    capability=make_rtx_capability(),
                    initial_state=rtx_state,
                )
            )
            # Test A: after one registration the Master sees exactly one Worker.
            assert len(service.registry) == 1
            assert rtx_response.heartbeat_interval_ms == TEST_CONFIG.heartbeat_interval_ms
            assert rtx_response.server_protocol_version == CONTROL_PROTOCOL_VERSION

            jetson_identity = make_worker_identity()
            jetson_state = make_worker_state(
                jetson_identity.worker_id,
                device_ids=(JETSON_CPU_DEVICE_ID, JETSON_GPU_DEVICE_ID),
                pool_ids=(JETSON_POOL_ID,),
            )
            jetson_response = await client.register_worker(
                RegisterWorkerRequest(
                    protocol_version=CONTROL_PROTOCOL_VERSION,
                    instance_id=str(uuid.uuid4()),
                    identity=jetson_identity,
                    capability=make_jetson_capability(),
                    initial_state=jetson_state,
                )
            )

            # Test B: the Master sees both Workers, independently.
            assert {record.worker_id for record in service.registry.list_workers()} == {
                rtx_identity.worker_id,
                jetson_identity.worker_id,
            }
            assert rtx_response.session_id != jetson_response.session_id
            assert service.worker_status(rtx_identity.worker_id) is WorkerStatus.ONLINE
            assert service.worker_status(jetson_identity.worker_id) is WorkerStatus.ONLINE

            # The decoded payloads cross-validate as Master snapshots (§38):
            # discrete VRAM pools on RTX, shared system-memory on Jetson.
            rtx_snapshot = snapshot_of(service, rtx_identity.worker_id)
            assert rtx_snapshot.identity == rtx_identity
            assert rtx_snapshot.capability == make_rtx_capability()
            assert rtx_snapshot.state == rtx_state
            assert {
                pool.memory_pool_id for pool in rtx_snapshot.capability.memory_pools
            } == {"host-memory", f"gpu-{RTX_GPU_DEVICE_ID}-vram"}

            jetson_snapshot = snapshot_of(service, jetson_identity.worker_id)
            assert jetson_snapshot.identity == jetson_identity
            assert jetson_snapshot.capability == make_jetson_capability()
            assert jetson_snapshot.state == jetson_state
            assert {
                pool.memory_pool_id for pool in jetson_snapshot.capability.memory_pools
            } == {JETSON_POOL_ID}

            # --- Heartbeats (§52 Test D shape: 1, 2, 4, then late 3) --------
            session_id = rtx_response.session_id
            instance_id = service.sessions.current(rtx_identity.worker_id).instance_id
            capability_revision = make_rtx_capability().capability_revision

            def heartbeat(
                sequence: int,
                utilization: float,
                *,
                reported_at: datetime | None = None,
            ) -> HeartbeatRequest:
                state = dataclasses.replace(
                    rtx_state,
                    device_states=tuple(
                        dataclasses.replace(s, utilization=utilization)
                        for s in rtx_state.device_states
                    ),
                )
                return HeartbeatRequest(
                    worker_id=rtx_identity.worker_id,
                    instance_id=instance_id,
                    session_id=session_id,
                    sequence_number=sequence,
                    capability_revision=capability_revision,
                    state=state,
                    worker_reported_at=reported_at,
                )

            for sequence, utilization in ((1, 10.0), (2, 20.0), (4, 40.0)):
                clock.advance(0.05)
                response = await client.heartbeat(heartbeat(sequence, utilization))
                assert response.accepted, response.detail
            late = await client.heartbeat(heartbeat(3, 30.0))
            assert late.accepted is False
            assert "out-of-order" in late.detail

            # Final stored state remains the one from sequence 4 (§52 D).
            stored = service.states.get(rtx_identity.worker_id)
            assert stored is not None
            assert all(s.utilization == 40.0 for s in stored.state.device_states)

            # §31: the Worker's wall clock is carried for debugging only —
            # the Master stored its own receive timestamps instead.
            worker_claim = datetime(1999, 1, 1, tzinfo=UTC)
            reported = await client.heartbeat(
                heartbeat(5, 41.0, reported_at=worker_claim)
            )
            assert reported.accepted
            stored = service.states.get(rtx_identity.worker_id)
            assert stored is not None
            assert stored.received_wall != worker_claim
            assert stored.received_wall == clock.wall()
            assert stored.received_monotonic == clock.monotonic()

            # --- Capability update (spec §16) --------------------------------
            evolved = evolve_rtx_capability()
            assert evolved.capability_revision != capability_revision
            update = await client.update_capability(
                UpdateCapabilityRequest(
                    worker_id=rtx_identity.worker_id,
                    instance_id=instance_id,
                    session_id=session_id,
                    capability=evolved,
                )
            )
            assert update.accepted
            assert service.registry.get(rtx_identity.worker_id).capability == evolved
            snapshot_of(service, rtx_identity.worker_id)  # still cross-validates

            # --- Re-registration (§52 Test C shape) ---------------------------
            new_instance_id = str(uuid.uuid4())
            reregistration = await client.register_worker(
                RegisterWorkerRequest(
                    protocol_version=CONTROL_PROTOCOL_VERSION,
                    instance_id=new_instance_id,
                    identity=rtx_identity,
                    capability=make_rtx_capability(),
                    initial_state=rtx_state,
                )
            )
            assert reregistration.session_id != session_id
            # Still the same registry entry (§52 C), old session invalid (§35).
            assert len(service.registry) == 2

            stale = await client.heartbeat(heartbeat(6, 50.0))
            assert stale.accepted is False
            assert "stale session" in stale.detail
            fresh = await client.heartbeat(
                dataclasses.replace(
                    heartbeat(1, 15.0),
                    instance_id=new_instance_id,
                    session_id=reregistration.session_id,
                )
            )
            assert fresh.accepted

            # --- Offline lifecycle (§52 Test E shape, fake clock) --------------
            worker_id = rtx_identity.worker_id
            clock.advance(0.201)  # beyond suspect_after (200 ms)
            assert service.worker_status(worker_id) is WorkerStatus.SUSPECT
            clock.advance(0.2)  # beyond offline_after (400 ms)
            assert service.worker_status(worker_id) is WorkerStatus.OFFLINE
            # The Jetson Worker never heartbeat at all, so its registration-time
            # last_seen aged past both thresholds too — and per §32 both OFFLINE
            # Workers remain in the registry; nothing is deleted automatically.
            assert service.worker_status(jetson_identity.worker_id) is WorkerStatus.OFFLINE
            assert len(service.registry) == 2

            # A returning Worker revives without re-registration (§52 E end).
            revived = await client.heartbeat(
                dataclasses.replace(
                    heartbeat(2, 16.0),
                    instance_id=new_instance_id,
                    session_id=reregistration.session_id,
                )
            )
            assert revived.accepted
            assert service.worker_status(worker_id) is WorkerStatus.ONLINE
        finally:
            await server.stop(grace=None)
