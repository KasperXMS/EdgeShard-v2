"""End-to-end control transport with fake Worker fixtures (spec §52 B/C/D).

RTX-like and Jetson-like Workers (the fixture platforms of spec §52 Test B)
register, heartbeat, and update capabilities against a fake Master over a
real ``grpc.aio`` channel. Every payload the fake Master decodes is
cross-validated as a ``WorkerSnapshot`` (spec §38), so the wire provably
preserves referential integrity of both memory topologies. The fake Master's
session/sequence logic is test scaffolding only — the production
implementation is milestone P1F.
"""

from __future__ import annotations

import dataclasses
import uuid

from edgeshard.cluster.capability import WorkerCapability
from edgeshard.cluster.identity import WorkerIdentity
from edgeshard.cluster.snapshot import WorkerSnapshot
from edgeshard.cluster.state import WorkerStatus
from edgeshard.protocol.control import mapper
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


class WorkerRecord:
    """Mutable fake-Master bookkeeping (the P1F StateStore preview)."""

    def __init__(self, request: RegisterWorkerRequest, session_id: str) -> None:
        self.instance_id = request.instance_id
        self.session_id = session_id
        self.capability = request.capability
        self.state = request.initial_state
        self.last_sequence = 0


class FakeMaster:
    """Handler implementing spec §29-30 acceptance rules for the test."""

    def __init__(self) -> None:
        self.workers: dict[str, WorkerRecord] = {}

    async def register_worker(
        self, request: RegisterWorkerRequest
    ) -> mapper.RegisterWorkerResponse:
        session_id = str(uuid.uuid4())
        self.workers[request.identity.worker_id] = WorkerRecord(request, session_id)
        return mapper.RegisterWorkerResponse(
            session_id=session_id,
            heartbeat_interval_ms=500,
            server_protocol_version=CONTROL_PROTOCOL_VERSION,
        )

    async def heartbeat(self, request: HeartbeatRequest) -> mapper.HeartbeatResponse:
        record = self.workers.get(request.worker_id)
        if record is None:
            return mapper.HeartbeatResponse(False, "unknown worker")
        if request.session_id != record.session_id:
            return mapper.HeartbeatResponse(False, "stale session")
        if request.instance_id != record.instance_id:
            return mapper.HeartbeatResponse(False, "instance mismatch")
        if request.sequence_number <= record.last_sequence:
            return mapper.HeartbeatResponse(False, "out-of-order heartbeat")
        record.last_sequence = request.sequence_number
        record.state = request.state
        return mapper.HeartbeatResponse(True)

    async def update_capability(
        self, request: UpdateCapabilityRequest
    ) -> mapper.UpdateCapabilityResponse:
        record = self.workers.get(request.worker_id)
        if record is None:
            return mapper.UpdateCapabilityResponse(False, "unknown worker")
        if request.session_id != record.session_id:
            return mapper.UpdateCapabilityResponse(False, "stale session")
        record.capability = request.capability
        return mapper.UpdateCapabilityResponse(True)

    def snapshot(self, worker_id: str, identity: WorkerIdentity) -> WorkerSnapshot:
        record = self.workers[worker_id]
        return WorkerSnapshot(
            identity=identity,
            capability=record.capability,
            state=record.state,
            status=WorkerStatus.ONLINE,
            session_id=record.session_id,
            last_seen_at=None,
        )


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
    """§52 Test B shapes: both platforms register and stay independent."""
    master = FakeMaster()
    server, port = await start_control_server(master, host=HOST, port=0)
    async with WorkerRegistryClient(f"{HOST}:{port}") as client:
        try:
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

            assert set(master.workers) == {
                rtx_identity.worker_id,
                jetson_identity.worker_id,
            }
            assert rtx_response.session_id != jetson_response.session_id

            # The decoded payloads cross-validate as Master snapshots (§38):
            # discrete VRAM pools on RTX, shared system-memory on Jetson.
            rtx_snapshot = master.snapshot(rtx_identity.worker_id, rtx_identity)
            assert rtx_snapshot.capability == make_rtx_capability()
            assert rtx_snapshot.state == rtx_state
            assert {
                pool.memory_pool_id for pool in rtx_snapshot.capability.memory_pools
            } == {"host-memory", f"gpu-{RTX_GPU_DEVICE_ID}-vram"}

            jetson_snapshot = master.snapshot(jetson_identity.worker_id, jetson_identity)
            assert jetson_snapshot.capability == make_jetson_capability()
            assert jetson_snapshot.state == jetson_state
            assert {
                pool.memory_pool_id for pool in jetson_snapshot.capability.memory_pools
            } == {JETSON_POOL_ID}

            # --- Heartbeats (§52 Test D shape: 1, 2, 4, then late 3) --------
            session_id = rtx_response.session_id
            instance_id = master.workers[rtx_identity.worker_id].instance_id
            capability_revision = make_rtx_capability().capability_revision

            def heartbeat(sequence: int, utilization: float) -> HeartbeatRequest:
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
                )

            for sequence, utilization in ((1, 10.0), (2, 20.0), (4, 40.0)):
                response = await client.heartbeat(heartbeat(sequence, utilization))
                assert response.accepted, response.detail
            late = await client.heartbeat(heartbeat(3, 30.0))
            assert late.accepted is False
            assert "out-of-order" in late.detail

            # Final stored state remains the one from sequence 4 (§52 D).
            final = master.workers[rtx_identity.worker_id].state
            assert all(s.utilization == 40.0 for s in final.device_states)
            master.snapshot(rtx_identity.worker_id, rtx_identity)

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
            assert master.workers[rtx_identity.worker_id].capability == evolved
            master.snapshot(rtx_identity.worker_id, rtx_identity)

            # --- Re-registration (spec §52 Test C shape) ---------------------
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
            stale = await client.heartbeat(heartbeat(5, 50.0))
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
        finally:
            await server.stop(grace=None)
