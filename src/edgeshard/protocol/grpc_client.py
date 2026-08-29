"""Async gRPC client for the ShardRuntime service (spec 5.4, 17).

Phase 0 gRPC is a correctness/reference transport: message limits are raised
to an explicit ceiling above the default (spec 17.4); nothing is optimized
for large activation transfer. The same options helper configures both
client channels and servers so send/receive limits always match.
"""

from __future__ import annotations

import grpc

from edgeshard.protocol import protobuf_mapper as mapper
from edgeshard.protocol.domain import ProtocolError, ShardMessage
from edgeshard.protocol.pb import shard_runtime_pb2 as pb
from edgeshard.protocol.pb import shard_runtime_pb2_grpc as pb_grpc

MAX_MESSAGE_BYTES: int = 512 * 1024 * 1024
"""Phase 0 send/receive ceiling (spec 17.4): correctness baseline only."""


def channel_options() -> list[tuple[str, int]]:
    """gRPC channel/server options enforcing the Phase 0 message ceiling."""
    return [
        ("grpc.max_send_message_length", MAX_MESSAGE_BYTES),
        ("grpc.max_receive_message_length", MAX_MESSAGE_BYTES),
    ]


class ShardRuntimeClient:
    """Client for one shard runtime endpoint."""

    def __init__(self, endpoint: str) -> None:
        self._endpoint = endpoint
        self._channel = grpc.aio.insecure_channel(endpoint, options=channel_options())
        # Generated grpc stub code is untyped (only pb2 ships .pyi stubs).
        self._stub = pb_grpc.ShardRuntimeStub(self._channel)  # type: ignore[no-untyped-call]

    @property
    def endpoint(self) -> str:
        return self._endpoint

    async def get_runtime_info(self) -> pb.RuntimeInfoReply:
        reply: pb.RuntimeInfoReply = await self._stub.GetRuntimeInfo(pb.RuntimeInfoRequest())
        return reply

    async def create_session(self, execution_id: str, session_id: str) -> None:
        reply = await self._stub.CreateSession(
            pb.CreateSessionRequest(execution_id=execution_id, session_id=session_id)
        )
        if not reply.ok:
            raise ProtocolError(f"create session failed: {reply.detail or 'no detail'}")

    async def close_session(self, execution_id: str, session_id: str) -> None:
        reply = await self._stub.CloseSession(
            pb.CloseSessionRequest(execution_id=execution_id, session_id=session_id)
        )
        if not reply.ok:
            raise ProtocolError(f"close session failed: {reply.detail or 'no detail'}")

    async def prefill(self, message: ShardMessage) -> ShardMessage:
        reply = await self._stub.Prefill(mapper.message_to_forward_request(message))
        return mapper.forward_reply_to_message(reply)

    async def decode(self, message: ShardMessage) -> ShardMessage:
        reply = await self._stub.Decode(mapper.message_to_forward_request(message))
        return mapper.forward_reply_to_message(reply)

    async def close(self) -> None:
        await self._channel.close()

    async def __aenter__(self) -> ShardRuntimeClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()
