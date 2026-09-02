"""Protobuf mapper: lossless domain<->wire roundtrips (spec 17.2, 17.3).

The gate for milestone 0D: serialization preserves inference semantics —
real shard outputs survive a full wire roundtrip (bytes included) with every
header/context field and tensor value intact.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from transformers import LlamaForCausalLM

from edgeshard.inference.shard import ShardModule
from edgeshard.inference.state import (
    ExecutionContext,
    InferencePhase,
    LogitsMode,
    LogitsOutput,
    ShardState,
)
from edgeshard.model.source import ModelSource
from edgeshard.model.spec import BlockRange, ShardSpec
from edgeshard.protocol import protobuf_mapper as mapper
from edgeshard.protocol.domain import (
    MASTER_STAGE,
    PROTOCOL_VERSION,
    HiddenStatePayload,
    LogitsPayload,
    ProtocolError,
    ShardMessage,
    ShardMessageHeader,
    TokenPayload,
)
from edgeshard.protocol.pb import shard_runtime_pb2 as pb


def make_header(**overrides: object) -> ShardMessageHeader:
    defaults: dict[str, object] = {
        "protocol_version": PROTOCOL_VERSION,
        "execution_id": "exec-9",
        "session_id": "sess-3",
        "request_id": "req-42",
        "phase": InferencePhase.PREFILL,
        "step": 0,
        "source_stage": MASTER_STAGE,
        "target_stage": 0,
    }
    defaults.update(overrides)
    return ShardMessageHeader(**defaults)  # type: ignore[arg-type]


def make_context(
    phase: InferencePhase = InferencePhase.PREFILL, positions: torch.Tensor | None = None
) -> ExecutionContext:
    return ExecutionContext(
        phase=phase,
        step=1,
        batch_size=1,
        sequence_lengths=(7,),
        past_length=0,
        positions=positions,
    )


def assert_headers_equal(actual: ShardMessageHeader, expected: ShardMessageHeader) -> None:
    assert actual == expected


def assert_contexts_equal(actual: ExecutionContext, expected: ExecutionContext) -> None:
    assert actual.phase is expected.phase
    assert actual.step == expected.step
    assert actual.batch_size == expected.batch_size
    assert actual.sequence_lengths == expected.sequence_lengths
    assert actual.past_length == expected.past_length
    if expected.positions is None:
        assert actual.positions is None
    else:
        assert actual.positions is not None
        assert torch.equal(actual.positions, expected.positions)


def roundtrip_request(message: ShardMessage) -> ShardMessage:
    """Domain -> wire -> bytes -> wire -> domain (the actual transmission path)."""
    wire = mapper.message_to_forward_request(message)
    data = wire.SerializeToString()
    return mapper.forward_request_to_message(pb.ForwardRequest.FromString(data))


def test_token_payload_roundtrip() -> None:
    message = ShardMessage(
        header=make_header(phase=InferencePhase.DECODE, step=3),
        context=make_context(InferencePhase.DECODE, positions=torch.tensor([[9]])),
        payload=TokenPayload(token_ids=(127,)),
    )
    restored = roundtrip_request(message)

    assert_headers_equal(restored.header, message.header)
    assert_contexts_equal(restored.context, message.context)
    assert isinstance(restored.payload, TokenPayload)
    assert restored.payload.token_ids == (127,)


def test_hidden_state_payload_roundtrip_preserves_tensor() -> None:
    hidden = torch.randn(1, 7, 64)
    message = ShardMessage(
        header=make_header(source_stage=0, target_stage=1),
        context=make_context(positions=torch.arange(7).unsqueeze(0)),
        payload=HiddenStatePayload(hidden_states=hidden),
    )
    restored = roundtrip_request(message)

    assert_headers_equal(restored.header, message.header)
    assert_contexts_equal(restored.context, message.context)
    assert isinstance(restored.payload, HiddenStatePayload)
    assert torch.equal(restored.payload.hidden_states, hidden)


def test_logits_payload_roundtrip_via_reply() -> None:
    logits = torch.randn(1, 7, 128)
    message = ShardMessage(
        header=make_header(source_stage=2, target_stage=MASTER_STAGE),
        context=make_context(),
        payload=LogitsPayload(logits=logits),
    )
    wire = mapper.message_to_forward_reply(message)
    restored = mapper.forward_reply_to_message(
        pb.ForwardReply.FromString(wire.SerializeToString())
    )

    assert_headers_equal(restored.header, message.header)
    assert_contexts_equal(restored.context, message.context)
    assert isinstance(restored.payload, LogitsPayload)
    assert torch.equal(restored.payload.logits, logits)


def test_bfloat16_tensor_survives_the_wire() -> None:
    hidden = torch.randn(1, 3, 16, dtype=torch.bfloat16)
    message = ShardMessage(
        header=make_header(),
        context=make_context(),
        payload=HiddenStatePayload(hidden_states=hidden),
    )
    restored = roundtrip_request(message)

    assert isinstance(restored.payload, HiddenStatePayload)
    assert restored.payload.hidden_states.dtype is torch.bfloat16
    assert torch.equal(restored.payload.hidden_states, hidden)


def test_wrong_protocol_version_on_read_is_rejected() -> None:
    message = ShardMessage(
        header=make_header(), context=make_context(), payload=TokenPayload(token_ids=(1,))
    )
    wire = mapper.message_to_forward_request(message)
    wire.header.protocol_version = PROTOCOL_VERSION + 1
    with pytest.raises(ProtocolError, match="protocol version"):
        mapper.forward_request_to_message(wire)


def test_unknown_wire_phase_is_rejected() -> None:
    message = ShardMessage(
        header=make_header(), context=make_context(), payload=TokenPayload(token_ids=(1,))
    )
    wire = mapper.message_to_forward_request(message)
    wire.header.phase = 99
    with pytest.raises(ProtocolError, match="unknown wire phase"):
        mapper.forward_request_to_message(wire)


def test_logits_mode_roundtrips_through_the_header() -> None:
    message = ShardMessage(
        header=make_header(logits_mode=LogitsMode.LAST_TOKEN),
        context=make_context(),
        payload=TokenPayload(token_ids=(1,)),
    )
    restored = roundtrip_request(message)
    assert restored.header.logits_mode is LogitsMode.LAST_TOKEN


def test_unset_wire_logits_mode_is_full() -> None:
    # LOGITS_FULL is wire value 0: messages without the field keep the
    # original full-logits semantics.
    message = ShardMessage(
        header=make_header(), context=make_context(), payload=TokenPayload(token_ids=(1,))
    )
    wire = mapper.message_to_forward_request(message)
    assert wire.header.logits_mode == pb.LOGITS_FULL
    restored = mapper.forward_request_to_message(
        pb.ForwardRequest.FromString(wire.SerializeToString())
    )
    assert restored.header.logits_mode is LogitsMode.FULL


def test_unknown_wire_logits_mode_is_rejected() -> None:
    message = ShardMessage(
        header=make_header(), context=make_context(), payload=TokenPayload(token_ids=(1,))
    )
    wire = mapper.message_to_forward_request(message)
    wire.header.logits_mode = 99
    with pytest.raises(ProtocolError, match="unknown wire logits mode"):
        mapper.forward_request_to_message(wire)


def test_missing_payload_is_rejected() -> None:
    header = make_header()
    wire = pb.ForwardRequest(
        header=pb.MessageHeader(
            protocol_version=header.protocol_version,
            execution_id=header.execution_id,
            session_id=header.session_id,
            request_id=header.request_id,
            phase=pb.PHASE_PREFILL,
        ),
        context=pb.ExecutionContext(phase=pb.PHASE_PREFILL, batch_size=1),
    )
    with pytest.raises(ProtocolError, match="no payload"):
        mapper.forward_request_to_message(wire)


def test_payload_tensor_missing_from_bundle_is_rejected() -> None:
    message = ShardMessage(
        header=make_header(),
        context=make_context(),
        payload=HiddenStatePayload(hidden_states=torch.zeros(1, 2, 4)),
    )
    wire = mapper.message_to_forward_request(message)
    wire.tensor_bundle = b""  # simulate a corrupted/dropped bundle
    with pytest.raises(ProtocolError, match="lacks payload tensor"):
        mapper.forward_request_to_message(wire)


def test_real_shard_output_semantics_survive_roundtrip(
    tiny_llama_source: ModelSource, tiny_llama_dir: Path
) -> None:
    """0D gate: canonical state is sufficient for complete inference.

    Real prefill logits and middle-shard hidden states from an actual
    ShardModule must come back bit-identical after serialization.
    """
    full = ShardModule.build(
        source=tiny_llama_source,
        shard=ShardSpec(
            model_id="tiny/llama",
            blocks=BlockRange(0, 4),
            include_input_stage=True,
            include_output_stage=True,
        ),
    )
    full.create_session("s")
    prompt = torch.tensor([[1, 9, 17, 25, 33]])
    output = full.prefill("s", input_ids=prompt)
    assert isinstance(output, LogitsOutput)

    logits_message = ShardMessage(
        header=make_header(source_stage=0, target_stage=MASTER_STAGE),
        context=output.context,
        payload=LogitsPayload(logits=output.logits),
    )
    restored = roundtrip_request(logits_message)
    assert isinstance(restored.payload, LogitsPayload)
    assert torch.equal(restored.payload.logits, output.logits)
    assert_contexts_equal(restored.context, output.context)

    reference = LlamaForCausalLM.from_pretrained(tiny_llama_dir).eval()
    hidden = reference(prompt, output_hidden_states=True).hidden_states[1]
    state = ShardState(
        hidden_states=hidden,
        context=ExecutionContext(
            phase=InferencePhase.PREFILL,
            step=1,
            batch_size=1,
            sequence_lengths=(5,),
            past_length=0,
            positions=torch.arange(5).unsqueeze(0),
        ),
    )
    state_message = ShardMessage(
        header=make_header(source_stage=0, target_stage=1),
        context=state.context,
        payload=HiddenStatePayload(hidden_states=state.hidden_states),
    )
    restored_state = roundtrip_request(state_message)
    assert isinstance(restored_state.payload, HiddenStatePayload)
    assert torch.equal(restored_state.payload.hidden_states, state.hidden_states)
