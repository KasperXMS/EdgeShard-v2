"""Serialized stage-boundary implementation for the local pipeline (0E).

Every hop goes through the full wire path — domain message, protobuf DTO,
bytes, back to domain — and inbound forward messages are validated against
spec 16.2 sequencing rules. This is the ``StageTransport`` implementation
the inference port is composed with (spec 7 keeps the dependency pointing
protocol -> inference, never the reverse).
"""

from __future__ import annotations

import uuid

from edgeshard.inference.state import (
    ExecutionContext,
    InferencePhase,
    LogitsOutput,
    ShardState,
)
from edgeshard.protocol import protobuf_mapper as mapper
from edgeshard.protocol.domain import (
    MASTER_STAGE,
    PROTOCOL_VERSION,
    HiddenStatePayload,
    LogitsPayload,
    MessageExpectation,
    ProtocolError,
    ShardMessage,
    ShardMessageHeader,
    TokenPayload,
    validate_forward_message,
)
from edgeshard.protocol.pb import shard_runtime_pb2 as pb


class SerializedStageBoundary:
    """Carries pipeline hops through canonical wire serialization."""

    def __init__(self, *, execution_id: str) -> None:
        self._execution_id = execution_id

    def carry_state(
        self,
        *,
        session_id: str,
        phase: InferencePhase,
        step: int,
        source_stage: int,
        target_stage: int,
        state: ShardState,
    ) -> ShardState:
        message = self._message(
            session_id=session_id,
            phase=phase,
            step=step,
            source_stage=source_stage,
            target_stage=target_stage,
            context=state.context,
            payload=HiddenStatePayload(hidden_states=state.hidden_states),
        )
        restored = self._roundtrip(message, target_stage=target_stage, step=step)
        payload = restored.payload
        if not isinstance(payload, HiddenStatePayload):
            raise ProtocolError("hidden-state hop restored a non-hidden-state payload")
        return ShardState(hidden_states=payload.hidden_states, context=restored.context)

    def carry_token(
        self,
        *,
        session_id: str,
        step: int,
        token_id: int,
        context: ExecutionContext,
    ) -> int:
        message = self._message(
            session_id=session_id,
            phase=InferencePhase.DECODE,
            step=step,
            source_stage=MASTER_STAGE,
            target_stage=0,
            context=context,
            payload=TokenPayload(token_id=token_id),
        )
        restored = self._roundtrip(message, target_stage=0, step=step)
        payload = restored.payload
        if not isinstance(payload, TokenPayload):
            raise ProtocolError("token hop restored a non-token payload")
        return payload.token_id

    def carry_reply(
        self,
        *,
        session_id: str,
        phase: InferencePhase,
        step: int,
        source_stage: int,
        output: LogitsOutput,
    ) -> LogitsOutput:
        message = self._message(
            session_id=session_id,
            phase=phase,
            step=step,
            source_stage=source_stage,
            target_stage=MASTER_STAGE,
            context=output.context,
            payload=LogitsPayload(logits=output.logits),
        )
        wire = mapper.message_to_forward_reply(message)
        restored = mapper.forward_reply_to_message(
            pb.ForwardReply.FromString(wire.SerializeToString())
        )
        payload = restored.payload
        if not isinstance(payload, LogitsPayload):
            raise ProtocolError("reply hop restored a non-logits payload")
        return LogitsOutput(logits=payload.logits, context=restored.context)

    def _message(
        self,
        *,
        session_id: str,
        phase: InferencePhase,
        step: int,
        source_stage: int,
        target_stage: int,
        context: ExecutionContext,
        payload: HiddenStatePayload | LogitsPayload | TokenPayload,
    ) -> ShardMessage:
        header = ShardMessageHeader(
            protocol_version=PROTOCOL_VERSION,
            execution_id=self._execution_id,
            session_id=session_id,
            request_id=uuid.uuid4().hex,
            phase=phase,
            step=step,
            source_stage=source_stage,
            target_stage=target_stage,
        )
        return ShardMessage(header=header, context=context, payload=payload)

    def _roundtrip(
        self, message: ShardMessage, *, target_stage: int, step: int
    ) -> ShardMessage:
        wire = mapper.message_to_forward_request(message)
        restored = mapper.forward_request_to_message(
            pb.ForwardRequest.FromString(wire.SerializeToString())
        )
        validate_forward_message(
            restored,
            MessageExpectation(
                execution_id=self._execution_id,
                stage_index=target_stage,
                next_step=step,
                session_open=True,
            ),
        )
        return restored
