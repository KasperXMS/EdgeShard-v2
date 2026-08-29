"""gRPC servicer bridging wire DTOs and the runtime handler (spec 17).

The servicer owns only the wire side: mapping request DTOs to domain
messages, delegating to a handler (implemented by the runtime layer), and
mapping replies back. Domain violations abort the RPC explicitly instead of
returning silently degraded results (spec 16.2).
"""

from __future__ import annotations

from typing import Protocol

import grpc

from edgeshard.inference.session import SessionError
from edgeshard.model.errors import EdgeShardError
from edgeshard.protocol import protobuf_mapper as mapper
from edgeshard.protocol.domain import ProtocolError, ShardMessage
from edgeshard.protocol.grpc_client import channel_options
from edgeshard.protocol.pb import shard_runtime_pb2 as pb
from edgeshard.protocol.pb import shard_runtime_pb2_grpc as pb_grpc


class ShardRuntimeHandler(Protocol):
    """Runtime-side behavior the servicer delegates to."""

    async def create_session(self, execution_id: str, session_id: str) -> None:
        ...

    async def close_session(self, execution_id: str, session_id: str) -> None:
        ...

    async def prefill(self, message: ShardMessage) -> ShardMessage:
        ...

    async def decode(self, message: ShardMessage) -> ShardMessage:
        ...


class ShardRuntimeServicer(pb_grpc.ShardRuntimeServicer):
    """ShardRuntime service implementation over a ShardRuntimeHandler."""

    def __init__(self, handler: ShardRuntimeHandler, runtime_info: pb.RuntimeInfoReply) -> None:
        self._handler = handler
        self._runtime_info = runtime_info

    async def GetRuntimeInfo(
        self, request: pb.RuntimeInfoRequest, context: grpc.aio.ServicerContext
    ) -> pb.RuntimeInfoReply:
        return self._runtime_info

    async def CreateSession(
        self, request: pb.CreateSessionRequest, context: grpc.aio.ServicerContext
    ) -> pb.CreateSessionReply:
        try:
            await self._handler.create_session(request.execution_id, request.session_id)
        except (EdgeShardError, ValueError) as exc:
            return pb.CreateSessionReply(ok=False, detail=str(exc))
        return pb.CreateSessionReply(ok=True)

    async def CloseSession(
        self, request: pb.CloseSessionRequest, context: grpc.aio.ServicerContext
    ) -> pb.CloseSessionReply:
        try:
            await self._handler.close_session(request.execution_id, request.session_id)
        except (EdgeShardError, ValueError) as exc:
            return pb.CloseSessionReply(ok=False, detail=str(exc))
        return pb.CloseSessionReply(ok=True)

    async def Prefill(
        self, request: pb.ForwardRequest, context: grpc.aio.ServicerContext
    ) -> pb.ForwardReply:
        return await self._forward(request, context, decode=False)

    async def Decode(
        self, request: pb.ForwardRequest, context: grpc.aio.ServicerContext
    ) -> pb.ForwardReply:
        return await self._forward(request, context, decode=True)

    async def _forward(
        self,
        request: pb.ForwardRequest,
        context: grpc.aio.ServicerContext,
        *,
        decode: bool,
    ) -> pb.ForwardReply:
        try:
            message = mapper.forward_request_to_message(request)
            if decode:
                reply = await self._handler.decode(message)
            else:
                reply = await self._handler.prefill(message)
            return mapper.message_to_forward_reply(reply)
        except (ProtocolError, SessionError, ValueError) as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            raise  # unreachable: grpc.aio abort raises AbortError


async def start_shard_runtime_server(
    handler: ShardRuntimeHandler,
    runtime_info: pb.RuntimeInfoReply,
    *,
    host: str,
    port: int,
) -> tuple[grpc.aio.Server, int]:
    """Start the gRPC server; returns the server and the bound port.

    ``port=0`` lets the OS choose, and the bound port is reported back —
    useful for tests that run several runtimes on one host.
    """
    server = grpc.aio.server(options=channel_options())
    pb_grpc.add_ShardRuntimeServicer_to_server(  # type: ignore[no-untyped-call]
        ShardRuntimeServicer(handler, runtime_info), server
    )
    bound_port: int = server.add_insecure_port(f"{host}:{port}")
    if bound_port == 0:
        raise RuntimeError(f"could not bind {host}:{port}")
    await server.start()
    return server, bound_port
