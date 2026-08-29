"""Protocol domain model: identity scopes, payloads, and message shape (spec 16)."""

from __future__ import annotations

import torch

from edgeshard.inference.state import ExecutionContext, InferencePhase
from edgeshard.protocol.domain import (
    MASTER_STAGE,
    PROTOCOL_VERSION,
    HiddenStatePayload,
    LogitsPayload,
    ShardMessage,
    ShardMessageHeader,
    TokenPayload,
)


def make_header(**overrides: object) -> ShardMessageHeader:
    defaults: dict[str, object] = {
        "protocol_version": PROTOCOL_VERSION,
        "execution_id": "exec-1",
        "session_id": "session-1",
        "request_id": "request-1",
        "phase": InferencePhase.PREFILL,
        "step": 0,
        "source_stage": MASTER_STAGE,
        "target_stage": 0,
    }
    defaults.update(overrides)
    return ShardMessageHeader(**defaults)  # type: ignore[arg-type]


def make_context(phase: InferencePhase = InferencePhase.PREFILL) -> ExecutionContext:
    return ExecutionContext(
        phase=phase,
        step=0,
        batch_size=1,
        sequence_lengths=(3,),
        past_length=0,
        positions=None,
    )


def test_header_carries_all_identity_and_routing_fields() -> None:
    header = make_header()
    assert header.protocol_version == PROTOCOL_VERSION
    assert header.execution_id == "exec-1"
    assert header.session_id == "session-1"
    assert header.request_id == "request-1"
    assert header.phase is InferencePhase.PREFILL
    assert header.step == 0
    assert header.source_stage == MASTER_STAGE
    assert header.target_stage == 0


def test_master_stage_precedes_the_first_shard_stage() -> None:
    # Stage 0's predecessor is the driver/master, not a shard.
    assert MASTER_STAGE == -1
    assert make_header().source_stage == MASTER_STAGE


def test_payload_categories_are_distinct() -> None:
    token = TokenPayload(token_id=7)
    hidden = HiddenStatePayload(hidden_states=torch.zeros(1, 2, 4))
    logits = LogitsPayload(logits=torch.zeros(1, 2, 8))

    assert token.token_id == 7
    assert hidden.hidden_states.shape == (1, 2, 4)
    assert logits.logits.shape == (1, 2, 8)
    assert not isinstance(token, HiddenStatePayload)
    assert not isinstance(hidden, LogitsPayload)


def test_message_bundles_header_context_and_payload() -> None:
    message = ShardMessage(
        header=make_header(),
        context=make_context(),
        payload=TokenPayload(token_id=3),
    )
    assert message.header.execution_id == "exec-1"
    assert message.context.sequence_lengths == (3,)
    assert isinstance(message.payload, TokenPayload)
