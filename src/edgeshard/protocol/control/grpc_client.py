"""Async gRPC client for the Worker control plane (Phase 1 spec §28).

Workers speak only domain objects across this API: requests and responses are
the mapper's frozen dataclasses, never protobuf DTOs (spec §40). The channel
is insecure today (trusted LAN / overlay, spec §48); construction funnels
through one place so TLS credentials can be added later without touching
callers.
"""

from __future__ import annotations

import grpc

from edgeshard.protocol.control import mapper
from edgeshard.protocol.control.pb import worker_control_pb2_grpc as pb_grpc

MAX_CONTROL_MESSAGE_BYTES: int = 8 * 1024 * 1024
"""Control-plane send/receive ceiling: capabilities are kilobyte-scale, so a
generous but explicit bound fails loudly on pathological messages."""


def control_channel_options() -> list[tuple[str, int]]:
    """gRPC channel/server options shared by both sides of the control plane."""
    return [
        ("grpc.max_send_message_length", MAX_CONTROL_MESSAGE_BYTES),
        ("grpc.max_receive_message_length", MAX_CONTROL_MESSAGE_BYTES),
    ]


class WorkerRegistryClient:
    """Client for the Master's ``WorkerRegistryService`` endpoint."""

    def __init__(self, endpoint: str) -> None:
        self._endpoint = endpoint
        # TLS: this insecure channel is the single transport chokepoint
        # (spec §48); swap for a secure channel when credentials land.
        self._channel = grpc.aio.insecure_channel(
            endpoint, options=control_channel_options()
        )
        # Generated grpc stub code is untyped (only pb2 ships .pyi stubs).
        self._stub = pb_grpc.WorkerRegistryServiceStub(self._channel)  # type: ignore[no-untyped-call]

    @property
    def endpoint(self) -> str:
        return self._endpoint

    async def register_worker(
        self, request: mapper.RegisterWorkerRequest, *, timeout: float | None = None
    ) -> mapper.RegisterWorkerResponse:
        wire = await self._stub.RegisterWorker(
            mapper.register_request_to_wire(request), timeout=timeout
        )
        return mapper.register_response_from_wire(wire)

    async def heartbeat(
        self, request: mapper.HeartbeatRequest, *, timeout: float | None = None
    ) -> mapper.HeartbeatResponse:
        wire = await self._stub.Heartbeat(
            mapper.heartbeat_request_to_wire(request), timeout=timeout
        )
        return mapper.heartbeat_response_from_wire(wire)

    async def update_capability(
        self, request: mapper.UpdateCapabilityRequest, *, timeout: float | None = None
    ) -> mapper.UpdateCapabilityResponse:
        wire = await self._stub.UpdateCapability(
            mapper.update_capability_request_to_wire(request), timeout=timeout
        )
        return mapper.update_capability_response_from_wire(wire)

    async def close(self) -> None:
        await self._channel.close()

    async def __aenter__(self) -> WorkerRegistryClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()
