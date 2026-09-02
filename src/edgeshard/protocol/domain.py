"""Canonical shard protocol domain model (spec 16).

These dataclasses are the inference domain model; protobuf objects are wire
DTOs only (spec 17.2) and are mapped explicitly in ``protobuf_mapper``.

Identity scopes (spec 16.1): ``execution_id`` names one concrete pipeline
deployment, ``session_id`` names one generation session, ``request_id`` names
one API-level action. They are not interchangeable.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from edgeshard.inference.state import ExecutionContext, InferencePhase, LogitsMode
from edgeshard.model.errors import EdgeShardError

PROTOCOL_VERSION: int = 1
"""Version of the canonical shard protocol carried in every message header."""

MASTER_STAGE: int = -1
"""Stage index of the pipeline driver / Mock Master, which is not a shard.

The first shard stage receives messages with ``source_stage == MASTER_STAGE``;
every other stage ``k`` receives messages from stage ``k - 1``.
"""

TENSOR_KEY_POSITIONS = "positions"
TENSOR_KEY_HIDDEN_STATES = "hidden_states"
TENSOR_KEY_LOGITS = "logits"


class ProtocolError(EdgeShardError):
    """Canonical protocol violation: identity, routing, version, or payload."""


class SequencingError(ProtocolError):
    """Out-of-order, duplicate, or phase-mismatched step (spec 16.2)."""


@dataclass(frozen=True)
class TokenPayload:
    """Tokens consumed by the input stage.

    A prefill step carries the prompt sequence; a decode step carries
    exactly one sampled token. The header phase disambiguates.
    """

    token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.token_ids:
            raise ValueError("token payload must carry at least one token")


@dataclass(frozen=True)
class HiddenStatePayload:
    """Hidden states crossing a shard boundary."""

    hidden_states: torch.Tensor


@dataclass(frozen=True)
class LogitsPayload:
    """Final-shard logits over the vocabulary."""

    logits: torch.Tensor


ShardPayload = TokenPayload | HiddenStatePayload | LogitsPayload


@dataclass(frozen=True)
class ShardMessageHeader:
    """Minimum header fields required by spec 16."""

    protocol_version: int

    execution_id: str
    session_id: str
    request_id: str

    phase: InferencePhase
    step: int
    """Wire step index: prefill is step 0, the Nth decode is step N (spec 16.2)."""

    source_stage: int
    target_stage: int

    logits_mode: LogitsMode = LogitsMode.FULL
    """Request-scoped directive for the final shard's prefill logits.

    Forwarded unchanged stage to stage so the final shard knows what to
    compute. Defaults to full logits, the original Phase 0 semantics and
    the wire default (``LOGITS_FULL = 0``).
    """


@dataclass(frozen=True)
class ShardMessage:
    """One canonical message exchanged between pipeline stages."""

    header: ShardMessageHeader
    context: ExecutionContext
    payload: ShardPayload


@dataclass(frozen=True)
class MessageExpectation:
    """What a shard runtime expects from the next inbound forward message."""

    execution_id: str
    stage_index: int
    next_step: int
    """Step index the session will execute next (0 for prefill)."""

    session_open: bool


def validate_forward_message(message: ShardMessage, expectation: MessageExpectation) -> None:
    """Reject invalid sequencing and routing explicitly (spec 16.2).

    Raises :class:`ProtocolError` (or its :class:`SequencingError` subclass)
    on any violation; silent recovery is forbidden.
    """
    header = message.header
    if header.protocol_version != PROTOCOL_VERSION:
        raise ProtocolError(
            f"unsupported protocol version {header.protocol_version} "
            f"(expected {PROTOCOL_VERSION})"
        )
    if not header.execution_id:
        raise ProtocolError("empty execution_id")
    if not header.session_id:
        raise ProtocolError("empty session_id")
    if not header.request_id:
        raise ProtocolError("empty request_id")
    if header.execution_id != expectation.execution_id:
        raise ProtocolError(
            f"wrong execution ID: got {header.execution_id!r}, "
            f"expected {expectation.execution_id!r}"
        )
    if not expectation.session_open:
        raise ProtocolError(f"session {header.session_id!r} is closed")
    if header.target_stage != expectation.stage_index:
        raise ProtocolError(
            f"wrong target stage: got {header.target_stage}, "
            f"expected {expectation.stage_index}"
        )
    expected_source = expectation.stage_index - 1
    if header.source_stage != expected_source:
        raise ProtocolError(
            f"wrong source stage: got {header.source_stage}, expected {expected_source}"
        )
    if header.step < 0:
        raise SequencingError(f"negative step {header.step}")
    expected_phase = InferencePhase.PREFILL if header.step == 0 else InferencePhase.DECODE
    if header.phase is not expected_phase:
        raise SequencingError(
            f"phase {header.phase.value!r} does not match step {header.step} "
            f"(expected {expected_phase.value!r})"
        )
    if header.step != expectation.next_step:
        raise SequencingError(
            f"out-of-order step: got {header.step}, expected {expectation.next_step}"
        )
    if message.context.phase is not header.phase:
        raise ProtocolError(
            f"header/context phase mismatch: {header.phase.value!r} vs "
            f"{message.context.phase.value!r}"
        )
