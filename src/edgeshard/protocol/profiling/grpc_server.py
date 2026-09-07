"""Async gRPC servicer for the Worker profiling plane (Phase 2 spec §41).

The servicer owns only the wire side: mapping request messages to domain
DTOs, delegating to a :class:`WorkerProfilingHandler` (implemented by the
Worker's profiling runner, spec §38-39), and mapping responses back.
Domain-level protocol violations — malformed payloads, unknown enums,
mismatched redundant fields — abort the RPC explicitly with
``INVALID_ARGUMENT`` instead of returning silently degraded results
(spec §47); semantic refusals (unknown/stale session, busy device) are
*not* transport errors — they travel as ``accepted=False`` responses with
a typed :class:`~edgeshard.protocol.profiling.mapper.ProfilingRejection`
that the handler produces (spec §41).
"""

from __future__ import annotations

from typing import Protocol

import grpc

from edgeshard.protocol.profiling import mapper
from edgeshard.protocol.profiling.grpc_client import profiling_channel_options
from edgeshard.protocol.profiling.pb import profiling_pb2 as pb
from edgeshard.protocol.profiling.pb import profiling_pb2_grpc as pb_grpc


class WorkerProfilingHandler(Protocol):
    """Worker-side behavior the servicer delegates to (spec §38-39)."""

    async def prepare_profiling_session(
        self, request: mapper.PrepareProfilingSessionRequest
    ) -> mapper.PrepareProfilingSessionResponse: ...

    async def run_profiling_case(
        self, request: mapper.RunProfilingCaseRequest
    ) -> mapper.RunProfilingCaseResponse: ...

    async def get_profiling_case(
        self, request: mapper.GetProfilingCaseRequest
    ) -> mapper.GetProfilingCaseResponse: ...

    async def cancel_profiling_case(
        self, request: mapper.CancelProfilingCaseRequest
    ) -> mapper.CancelProfilingCaseResponse: ...

    async def close_profiling_session(
        self, request: mapper.CloseProfilingSessionRequest
    ) -> mapper.CloseProfilingSessionResponse: ...


class WorkerProfilingServicer(pb_grpc.WorkerProfilingServiceServicer):
    """WorkerProfilingService implementation over a WorkerProfilingHandler."""

    def __init__(self, handler: WorkerProfilingHandler) -> None:
        self._handler = handler

    async def PrepareProfilingSession(
        self,
        request: pb.PrepareProfilingSessionRequest,
        context: grpc.aio.ServicerContext,
    ) -> pb.PrepareProfilingSessionResponse:
        try:
            domain_request = mapper.prepare_request_from_wire(request)
            response = await self._handler.prepare_profiling_session(domain_request)
            return mapper.prepare_response_to_wire(response)
        except (mapper.ProfilingProtocolError, ValueError) as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            raise  # unreachable: grpc.aio abort raises AbortError

    async def RunProfilingCase(
        self,
        request: pb.RunProfilingCaseRequest,
        context: grpc.aio.ServicerContext,
    ) -> pb.RunProfilingCaseResponse:
        try:
            domain_request = mapper.run_case_request_from_wire(request)
            response = await self._handler.run_profiling_case(domain_request)
            return mapper.run_case_response_to_wire(response)
        except (mapper.ProfilingProtocolError, ValueError) as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            raise  # unreachable: grpc.aio abort raises AbortError

    async def GetProfilingCase(
        self,
        request: pb.GetProfilingCaseRequest,
        context: grpc.aio.ServicerContext,
    ) -> pb.GetProfilingCaseResponse:
        try:
            domain_request = mapper.get_case_request_from_wire(request)
            response = await self._handler.get_profiling_case(domain_request)
            return mapper.get_case_response_to_wire(response)
        except (mapper.ProfilingProtocolError, ValueError) as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            raise  # unreachable: grpc.aio abort raises AbortError

    async def CancelProfilingCase(
        self,
        request: pb.CancelProfilingCaseRequest,
        context: grpc.aio.ServicerContext,
    ) -> pb.CancelProfilingCaseResponse:
        try:
            domain_request = mapper.cancel_case_request_from_wire(request)
            response = await self._handler.cancel_profiling_case(domain_request)
            return mapper.cancel_case_response_to_wire(response)
        except (mapper.ProfilingProtocolError, ValueError) as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            raise  # unreachable: grpc.aio abort raises AbortError

    async def CloseProfilingSession(
        self,
        request: pb.CloseProfilingSessionRequest,
        context: grpc.aio.ServicerContext,
    ) -> pb.CloseProfilingSessionResponse:
        try:
            domain_request = mapper.close_session_request_from_wire(request)
            response = await self._handler.close_profiling_session(domain_request)
            return mapper.close_session_response_to_wire(response)
        except (mapper.ProfilingProtocolError, ValueError) as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            raise  # unreachable: grpc.aio abort raises AbortError


async def start_profiling_server(
    handler: WorkerProfilingHandler,
    *,
    host: str,
    port: int,
) -> tuple[grpc.aio.Server, int]:
    """Start the profiling-plane gRPC server; returns server and bound port.

    ``port=0`` lets the OS choose, and the bound port is reported back —
    ``worker serve`` advertises ``host:bound_port`` at registration
    (spec §41), and tests run Workers alongside a Master on one host.
    """
    server = grpc.aio.server(options=profiling_channel_options())
    pb_grpc.add_WorkerProfilingServiceServicer_to_server(  # type: ignore[no-untyped-call]
        WorkerProfilingServicer(handler), server
    )
    bound_port: int = server.add_insecure_port(f"{host}:{port}")
    if bound_port == 0:
        raise RuntimeError(f"could not bind {host}:{port}")
    await server.start()
    return server, bound_port
