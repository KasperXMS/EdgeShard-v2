"""Async gRPC servicer for the Worker control plane (Phase 1 spec §28).

The servicer owns only the wire side: mapping request DTOs to domain
objects, delegating to a :class:`WorkerRegistryHandler` (implemented by
``edgeshard.control.master.service.MasterService``), and mapping responses
back. Domain-level protocol
violations abort the RPC explicitly with ``INVALID_ARGUMENT`` instead of
returning silently degraded results (spec §47); semantic rejections (stale
session, out-of-order heartbeat) are *not* transport errors — they travel as
``accepted=False`` responses the handler produces (spec §30).
"""

from __future__ import annotations

from typing import Protocol

import grpc

from edgeshard.protocol.control import mapper
from edgeshard.protocol.control.grpc_client import control_channel_options
from edgeshard.protocol.control.pb import worker_control_pb2 as pb
from edgeshard.protocol.control.pb import worker_control_pb2_grpc as pb_grpc


class WorkerRegistryHandler(Protocol):
    """Master-side behavior the servicer delegates to (spec §29-30)."""

    async def register_worker(
        self, request: mapper.RegisterWorkerRequest
    ) -> mapper.RegisterWorkerResponse: ...

    async def heartbeat(
        self, request: mapper.HeartbeatRequest
    ) -> mapper.HeartbeatResponse: ...

    async def update_capability(
        self, request: mapper.UpdateCapabilityRequest
    ) -> mapper.UpdateCapabilityResponse: ...


class WorkerRegistryServicer(pb_grpc.WorkerRegistryServiceServicer):
    """WorkerRegistryService implementation over a WorkerRegistryHandler."""

    def __init__(self, handler: WorkerRegistryHandler) -> None:
        self._handler = handler

    async def RegisterWorker(
        self,
        request: pb.RegisterWorkerRequest,
        context: grpc.aio.ServicerContext,
    ) -> pb.RegisterWorkerResponse:
        try:
            domain_request = mapper.register_request_from_wire(request)
            response = await self._handler.register_worker(domain_request)
            return mapper.register_response_to_wire(response)
        except (mapper.ControlProtocolError, ValueError) as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            raise  # unreachable: grpc.aio abort raises AbortError

    async def Heartbeat(
        self,
        request: pb.HeartbeatRequest,
        context: grpc.aio.ServicerContext,
    ) -> pb.HeartbeatResponse:
        try:
            domain_request = mapper.heartbeat_request_from_wire(request)
            response = await self._handler.heartbeat(domain_request)
            return mapper.heartbeat_response_to_wire(response)
        except (mapper.ControlProtocolError, ValueError) as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            raise  # unreachable: grpc.aio abort raises AbortError

    async def UpdateCapability(
        self,
        request: pb.UpdateCapabilityRequest,
        context: grpc.aio.ServicerContext,
    ) -> pb.UpdateCapabilityResponse:
        try:
            domain_request = mapper.update_capability_request_from_wire(request)
            response = await self._handler.update_capability(domain_request)
            return mapper.update_capability_response_to_wire(response)
        except (mapper.ControlProtocolError, ValueError) as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            raise  # unreachable: grpc.aio abort raises AbortError


async def start_control_server(
    handler: WorkerRegistryHandler,
    *,
    host: str,
    port: int,
) -> tuple[grpc.aio.Server, int]:
    """Start the control-plane gRPC server; returns the server and bound port.

    ``port=0`` lets the OS choose, and the bound port is reported back —
    useful for tests that run a Master alongside Workers on one host.
    """
    server = grpc.aio.server(options=control_channel_options())
    pb_grpc.add_WorkerRegistryServiceServicer_to_server(  # type: ignore[no-untyped-call]
        WorkerRegistryServicer(handler), server
    )
    bound_port: int = server.add_insecure_port(f"{host}:{port}")
    if bound_port == 0:
        raise RuntimeError(f"could not bind {host}:{port}")
    await server.start()
    return server, bound_port
