"""Sequencing validation: all spec 16.2 rejection cases fail explicitly."""

from __future__ import annotations

import pytest
import torch

from edgeshard.inference.state import ExecutionContext, InferencePhase
from edgeshard.protocol.domain import (
    MASTER_STAGE,
    PROTOCOL_VERSION,
    MessageExpectation,
    ProtocolError,
    SequencingError,
    ShardMessage,
    ShardMessageHeader,
    TokenPayload,
    validate_forward_message,
)


def make_message(
    *,
    phase: InferencePhase = InferencePhase.PREFILL,
    step: int = 0,
    source_stage: int = MASTER_STAGE,
    target_stage: int = 0,
    protocol_version: int = PROTOCOL_VERSION,
    execution_id: str = "exec-1",
    session_id: str = "session-1",
    request_id: str = "request-1",
    context_phase: InferencePhase | None = None,
) -> ShardMessage:
    header = ShardMessageHeader(
        protocol_version=protocol_version,
        execution_id=execution_id,
        session_id=session_id,
        request_id=request_id,
        phase=phase,
        step=step,
        source_stage=source_stage,
        target_stage=target_stage,
    )
    context = ExecutionContext(
        phase=context_phase if context_phase is not None else phase,
        step=step,
        batch_size=1,
        sequence_lengths=(1,),
        past_length=step,
        positions=torch.zeros(1, 1, dtype=torch.long) if step else None,
    )
    return ShardMessage(header=header, context=context, payload=TokenPayload(token_ids=(1,)))


def make_expectation(
    *,
    execution_id: str = "exec-1",
    stage_index: int = 0,
    next_step: int = 0,
    session_open: bool = True,
) -> MessageExpectation:
    return MessageExpectation(
        execution_id=execution_id,
        stage_index=stage_index,
        next_step=next_step,
        session_open=session_open,
    )


def test_valid_prefill_and_decode_pass() -> None:
    validate_forward_message(make_message(step=0), make_expectation(next_step=0))
    validate_forward_message(
        make_message(phase=InferencePhase.DECODE, step=3, source_stage=MASTER_STAGE),
        make_expectation(next_step=3),
    )
    # Intermediate stage: messages arrive from the previous shard stage.
    validate_forward_message(
        make_message(phase=InferencePhase.DECODE, step=1, source_stage=0, target_stage=1),
        make_expectation(stage_index=1, next_step=1),
    )


def test_wrong_protocol_version_is_rejected() -> None:
    with pytest.raises(ProtocolError, match="protocol version"):
        validate_forward_message(
            make_message(protocol_version=PROTOCOL_VERSION + 1), make_expectation()
        )


@pytest.mark.parametrize("field", ["execution_id", "session_id", "request_id"])
def test_empty_identity_fields_are_rejected(field: str) -> None:
    with pytest.raises(ProtocolError, match=f"empty {field}"):
        validate_forward_message(make_message(**{field: ""}), make_expectation())


def test_wrong_execution_id_is_rejected() -> None:
    with pytest.raises(ProtocolError, match="wrong execution ID"):
        validate_forward_message(make_message(execution_id="other"), make_expectation())


def test_closed_session_is_rejected() -> None:
    with pytest.raises(ProtocolError, match="closed"):
        validate_forward_message(make_message(), make_expectation(session_open=False))


def test_wrong_target_stage_is_rejected() -> None:
    with pytest.raises(ProtocolError, match="wrong target stage"):
        validate_forward_message(make_message(target_stage=2), make_expectation(stage_index=1))


def test_wrong_source_stage_is_rejected() -> None:
    with pytest.raises(ProtocolError, match="wrong source stage"):
        validate_forward_message(make_message(source_stage=5), make_expectation())
    # Stage 0 only accepts the master as source, never another shard.
    with pytest.raises(ProtocolError, match="wrong source stage"):
        validate_forward_message(make_message(source_stage=0), make_expectation())


def test_negative_step_is_rejected() -> None:
    with pytest.raises(SequencingError, match="negative step"):
        validate_forward_message(make_message(step=-1), make_expectation())


def test_phase_must_match_step() -> None:
    # Step 0 is prefill only.
    with pytest.raises(SequencingError, match="does not match step"):
        validate_forward_message(
            make_message(phase=InferencePhase.DECODE, step=0), make_expectation()
        )
    # Later steps are decode only.
    with pytest.raises(SequencingError, match="does not match step"):
        validate_forward_message(
            make_message(phase=InferencePhase.PREFILL, step=1),
            make_expectation(next_step=1),
        )


def test_out_of_order_step_is_rejected() -> None:
    with pytest.raises(SequencingError, match="out-of-order"):
        validate_forward_message(make_message(step=0), make_expectation(next_step=2))


def test_duplicate_step_is_rejected() -> None:
    # Replaying the previous step (already executed) must fail.
    with pytest.raises(SequencingError, match="out-of-order"):
        validate_forward_message(
            make_message(phase=InferencePhase.DECODE, step=1),
            make_expectation(next_step=2),
        )


def test_header_context_phase_mismatch_is_rejected() -> None:
    with pytest.raises(ProtocolError, match="phase mismatch"):
        validate_forward_message(
            make_message(context_phase=InferencePhase.DECODE), make_expectation()
        )


def test_sequencing_error_is_a_protocol_error() -> None:
    with pytest.raises(ProtocolError):
        validate_forward_message(make_message(step=9), make_expectation(next_step=0))
