"""Explicit mapping between protocol domain objects and wire DTOs (spec 17.2).

Protobuf generated objects are wire DTOs only; this module is the single
place that translates both directions. Tensor fields travel through the
safetensors tensor bundle (spec 17.3): the ``positions`` tensor under
``TENSOR_KEY_POSITIONS`` (when present) and the payload tensor under its
payload-declared key.
"""

from __future__ import annotations

import torch

from edgeshard.inference.state import ExecutionContext, InferencePhase
from edgeshard.protocol import domain
from edgeshard.protocol.domain import (
    HiddenStatePayload,
    LogitsPayload,
    ProtocolError,
    ShardMessage,
    ShardMessageHeader,
    ShardPayload,
    TokenPayload,
)
from edgeshard.protocol.pb import shard_runtime_pb2 as pb
from edgeshard.protocol.tensor_codec import decode_tensors, encode_tensors

_PHASE_TO_WIRE: dict[InferencePhase, pb.Phase] = {
    InferencePhase.PREFILL: pb.PHASE_PREFILL,
    InferencePhase.DECODE: pb.PHASE_DECODE,
}
_WIRE_TO_PHASE: dict[int, InferencePhase] = {int(v): k for k, v in _PHASE_TO_WIRE.items()}


def message_to_forward_request(message: ShardMessage) -> pb.ForwardRequest:
    """Map a domain message to the Prefill/Decode request wire DTO."""
    payload, tensors = _payload_to_wire(message.payload)
    if message.context.positions is not None:
        tensors[domain.TENSOR_KEY_POSITIONS] = message.context.positions
    return pb.ForwardRequest(
        header=_header_to_wire(message.header),
        context=_context_to_wire(message.context),
        payload=payload,
        tensor_bundle=encode_tensors(tensors),
    )


def forward_request_to_message(wire: pb.ForwardRequest) -> ShardMessage:
    """Map a Prefill/Decode request wire DTO back to the domain message."""
    return _message_from_parts(wire.header, wire.context, wire.payload, wire.tensor_bundle)


def message_to_forward_reply(message: ShardMessage) -> pb.ForwardReply:
    """Map a domain message to the pipeline-output reply wire DTO."""
    payload, tensors = _payload_to_wire(message.payload)
    if message.context.positions is not None:
        tensors[domain.TENSOR_KEY_POSITIONS] = message.context.positions
    return pb.ForwardReply(
        header=_header_to_wire(message.header),
        context=_context_to_wire(message.context),
        payload=payload,
        tensor_bundle=encode_tensors(tensors),
    )


def forward_reply_to_message(wire: pb.ForwardReply) -> ShardMessage:
    """Map a pipeline-output reply wire DTO back to the domain message."""
    return _message_from_parts(wire.header, wire.context, wire.payload, wire.tensor_bundle)


def _message_from_parts(
    header_wire: pb.MessageHeader,
    context_wire: pb.ExecutionContext,
    payload_wire: pb.ShardPayload,
    tensor_bundle: bytes,
) -> ShardMessage:
    header = _header_from_wire(header_wire)
    if header.protocol_version != domain.PROTOCOL_VERSION:
        raise ProtocolError(
            f"unsupported protocol version {header.protocol_version} "
            f"(expected {domain.PROTOCOL_VERSION})"
        )
    tensors = decode_tensors(tensor_bundle)
    positions = tensors.pop(domain.TENSOR_KEY_POSITIONS, None)
    context = _context_from_wire(context_wire, positions)
    payload = _payload_from_wire(payload_wire, tensors)
    return ShardMessage(header=header, context=context, payload=payload)


def _header_to_wire(header: ShardMessageHeader) -> pb.MessageHeader:
    return pb.MessageHeader(
        protocol_version=header.protocol_version,
        execution_id=header.execution_id,
        session_id=header.session_id,
        request_id=header.request_id,
        phase=_PHASE_TO_WIRE[header.phase],
        step=header.step,
        source_stage=header.source_stage,
        target_stage=header.target_stage,
    )


def _header_from_wire(wire: pb.MessageHeader) -> ShardMessageHeader:
    phase = _WIRE_TO_PHASE.get(int(wire.phase))
    if phase is None:
        raise ProtocolError(f"unknown wire phase: {int(wire.phase)}")
    return ShardMessageHeader(
        protocol_version=wire.protocol_version,
        execution_id=wire.execution_id,
        session_id=wire.session_id,
        request_id=wire.request_id,
        phase=phase,
        step=wire.step,
        source_stage=wire.source_stage,
        target_stage=wire.target_stage,
    )


def _context_to_wire(context: ExecutionContext) -> pb.ExecutionContext:
    return pb.ExecutionContext(
        phase=_PHASE_TO_WIRE[context.phase],
        step=context.step,
        batch_size=context.batch_size,
        sequence_lengths=list(context.sequence_lengths),
        past_length=context.past_length,
    )


def _context_from_wire(
    wire: pb.ExecutionContext, positions: torch.Tensor | None
) -> ExecutionContext:
    phase = _WIRE_TO_PHASE.get(int(wire.phase))
    if phase is None:
        raise ProtocolError(f"unknown wire phase: {int(wire.phase)}")
    return ExecutionContext(
        phase=phase,
        step=wire.step,
        batch_size=wire.batch_size,
        sequence_lengths=tuple(wire.sequence_lengths),
        past_length=wire.past_length,
        positions=positions,
    )


def _payload_to_wire(payload: ShardPayload) -> tuple[pb.ShardPayload, dict[str, torch.Tensor]]:
    if isinstance(payload, TokenPayload):
        return pb.ShardPayload(token=pb.TokenPayload(token_ids=list(payload.token_ids))), {}
    if isinstance(payload, HiddenStatePayload):
        return (
            pb.ShardPayload(
                hidden_states=pb.HiddenStatePayload(tensor_key=domain.TENSOR_KEY_HIDDEN_STATES)
            ),
            {domain.TENSOR_KEY_HIDDEN_STATES: payload.hidden_states},
        )
    if isinstance(payload, LogitsPayload):
        return (
            pb.ShardPayload(logits=pb.LogitsPayload(tensor_key=domain.TENSOR_KEY_LOGITS)),
            {domain.TENSOR_KEY_LOGITS: payload.logits},
        )
    raise ProtocolError(f"unknown payload type: {type(payload)!r}")


def _payload_from_wire(
    wire: pb.ShardPayload, tensors: dict[str, torch.Tensor]
) -> ShardPayload:
    kind = wire.WhichOneof("value")
    if kind == "token":
        token_ids = tuple(wire.token.token_ids)
        if not token_ids:
            raise ProtocolError("token payload carries no tokens")
        return TokenPayload(token_ids=token_ids)
    if kind in ("hidden_states", "logits"):
        declared = wire.hidden_states if kind == "hidden_states" else wire.logits
        key = declared.tensor_key
        if not key:
            raise ProtocolError(f"{kind} payload declares no tensor key")
        tensor = tensors.get(key)
        if tensor is None:
            raise ProtocolError(f"tensor bundle lacks payload tensor {key!r}")
        if kind == "hidden_states":
            return HiddenStatePayload(hidden_states=tensor)
        return LogitsPayload(logits=tensor)
    raise ProtocolError("forward message carries no payload")
