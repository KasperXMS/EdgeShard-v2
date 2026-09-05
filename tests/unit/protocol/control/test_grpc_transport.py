"""Control-plane gRPC transport tests (Phase 1 spec §28, §40).

An in-process ``grpc.aio`` server with a recording handler stands in for the
Master (real Master state lands in P1F): domain objects must cross the wire
and back unchanged, and protocol violations must surface as explicit gRPC
errors instead of degraded responses (spec §47).
"""

from __future__ import annotations

import uuid

import grpc
import pytest

from edgeshard.protocol.control.grpc_client import (
    WorkerRegistryClient,
    control_channel_options,
)
from edgeshard.protocol.control.grpc_server import start_control_server
from edgeshard.protocol.control.mapper import (
    CONTROL_PROTOCOL_VERSION,
    ControlProtocolError,
    HeartbeatRequest,
    HeartbeatResponse,
    RegisterWorkerRequest,
    RegisterWorkerResponse,
    UpdateCapabilityRequest,
    UpdateCapabilityResponse,
)
from edgeshard.protocol.control.pb import worker_control_pb2 as pb
from edgeshard.protocol.control.pb import worker_control_pb2_grpc as pb_grpc
from factories import make_rtx_capability, make_worker_identity, make_worker_state

HOST = "127.0.0.1"


class RecordingHandler:
    """Fake Master: records domain requests, returns canned responses."""

    def __init__(self) -> None:
        self.registrations: list[RegisterWorkerRequest] = []
        self.heartbeats: list[HeartbeatRequest] = []
        self.updates: list[UpdateCapabilityRequest] = []
        self.session_id = str(uuid.uuid4())
        self.heartbeat_response = HeartbeatResponse(accepted=True)
        self.update_response = UpdateCapabilityResponse(accepted=True)
        self.register_error: Exception | None = None

    async def register_worker(
        self, request: RegisterWorkerRequest
    ) -> RegisterWorkerResponse:
        if self.register_error is not None:
            raise self.register_error
        self.registrations.append(request)
        return RegisterWorkerResponse(
            session_id=self.session_id,
            heartbeat_interval_ms=5000,
            server_protocol_version=CONTROL_PROTOCOL_VERSION,
        )

    async def heartbeat(self, request: HeartbeatRequest) -> HeartbeatResponse:
        self.heartbeats.append(request)
        return self.heartbeat_response

    async def update_capability(
        self, request: UpdateCapabilityRequest
    ) -> UpdateCapabilityResponse:
        self.updates.append(request)
        return self.update_response


@pytest.fixture
async def transport():
    handler = RecordingHandler()
    server, port = await start_control_server(handler, host=HOST, port=0)
    client = WorkerRegistryClient(f"{HOST}:{port}")
    try:
        yield handler, client, port
    finally:
        await client.close()
        await server.stop(grace=None)


def make_register_request(worker_id: str | None = None) -> RegisterWorkerRequest:
    identity = make_worker_identity(worker_id)
    return RegisterWorkerRequest(
        protocol_version=CONTROL_PROTOCOL_VERSION,
        instance_id=str(uuid.uuid4()),
        identity=identity,
        capability=make_rtx_capability(),
        initial_state=make_worker_state(identity.worker_id),
    )


def make_heartbeat_request(worker_id: str, *, sequence_number: int) -> HeartbeatRequest:
    capability = make_rtx_capability()
    return HeartbeatRequest(
        worker_id=worker_id,
        instance_id=str(uuid.uuid4()),
        session_id="session-1",
        sequence_number=sequence_number,
        capability_revision=capability.capability_revision,
        state=make_worker_state(worker_id),
    )


async def test_register_roundtrip_over_grpc(transport) -> None:
    handler, client, _port = transport
    request = make_register_request()

    response = await client.register_worker(request)

    assert response.session_id == handler.session_id
    assert response.heartbeat_interval_ms == 5000
    assert response.server_protocol_version == CONTROL_PROTOCOL_VERSION
    (received,) = handler.registrations
    assert received == request


async def test_heartbeat_accepted_roundtrip(transport) -> None:
    handler, client, _port = transport
    request = make_heartbeat_request(str(uuid.uuid4()), sequence_number=1)

    response = await client.heartbeat(request)

    assert response == HeartbeatResponse(accepted=True)
    assert handler.heartbeats == [request]


async def test_heartbeat_rejection_carries_detail(transport) -> None:
    handler, client, _port = transport
    handler.heartbeat_response = HeartbeatResponse(
        accepted=False, detail="stale session"
    )
    request = make_heartbeat_request(str(uuid.uuid4()), sequence_number=1)

    response = await client.heartbeat(request)

    assert response.accepted is False
    assert response.detail == "stale session"


async def test_update_capability_roundtrip(transport) -> None:
    handler, client, _port = transport
    worker_id = str(uuid.uuid4())
    request = UpdateCapabilityRequest(
        worker_id=worker_id,
        instance_id=str(uuid.uuid4()),
        session_id="session-1",
        capability=make_rtx_capability(),
    )

    response = await client.update_capability(request)

    assert response == UpdateCapabilityResponse(accepted=True)
    assert handler.updates == [request]


async def test_handler_protocol_error_aborts_rpc(transport) -> None:
    handler, client, _port = transport
    handler.register_error = ControlProtocolError("boom")

    with pytest.raises(grpc.aio.AioRpcError) as exc_info:
        await client.register_worker(make_register_request())

    assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT


async def test_handler_value_error_aborts_rpc(transport) -> None:
    handler, client, _port = transport
    handler.register_error = ValueError("bad request shape")

    with pytest.raises(grpc.aio.AioRpcError) as exc_info:
        await client.register_worker(make_register_request())

    assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT


async def test_invalid_wire_request_aborts_rpc(transport) -> None:
    """Crafted wire garbage never reaches the handler (spec §47)."""
    handler, _client, port = transport
    channel = grpc.aio.insecure_channel(f"{HOST}:{port}", options=control_channel_options())
    stub = pb_grpc.WorkerRegistryServiceStub(channel)  # type: ignore[no-untyped-call]
    try:
        with pytest.raises(grpc.aio.AioRpcError) as exc_info:
            await stub.RegisterWorker(
                pb.RegisterWorkerRequest(
                    protocol_version="999",
                    worker_id="w",
                    instance_id="i",
                )
            )
        assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    finally:
        await channel.close()
    assert handler.registrations == []


async def test_invalid_wire_heartbeat_aborts_rpc(transport) -> None:
    """Sequence 0 violates spec 30 and must abort, not be accepted."""
    handler, _client, port = transport
    channel = grpc.aio.insecure_channel(f"{HOST}:{port}", options=control_channel_options())
    stub = pb_grpc.WorkerRegistryServiceStub(channel)  # type: ignore[no-untyped-call]
    try:
        with pytest.raises(grpc.aio.AioRpcError) as exc_info:
            await stub.Heartbeat(
                pb.HeartbeatRequest(
                    worker_id="w",
                    instance_id="i",
                    session_id="s",
                    sequence_number=0,
                    capability_revision="rev",
                )
            )
        assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    finally:
        await channel.close()
    assert handler.heartbeats == []


async def test_two_workers_register_independently(transport) -> None:
    handler, client, _port = transport
    first = make_register_request()
    second = make_register_request()

    await client.register_worker(first)
    await client.register_worker(second)

    assert [r.identity.worker_id for r in handler.registrations] == [
        first.identity.worker_id,
        second.identity.worker_id,
    ]
    assert first.identity.worker_id != second.identity.worker_id
