"""Slice equivalence: first/middle/last shards match native hidden-state slices.

For a shard covering blocks ``[start, end)``, its output must equal the full
model's hidden state after block ``end - 1`` — i.e. entry ``end`` of the
native ``output_hidden_states`` tuple (entry ``k`` is the input to block
``k``). Verified for prefill and multi-step decode, which is the partition
invariance foundation of the later pipeline milestones (spec 4.9).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from transformers import LlamaForCausalLM

from edgeshard.inference.session import ShardSession
from edgeshard.inference.shard import ShardModule
from edgeshard.inference.state import LogitsOutput, ShardState
from edgeshard.model.source import ModelSource
from edgeshard.model.spec import BlockRange, ShardSpec

TOLERANCE = {"atol": 1e-6, "rtol": 1e-5}


@pytest.fixture
def llama_reference(tiny_llama_dir: Path) -> LlamaForCausalLM:
    return LlamaForCausalLM.from_pretrained(tiny_llama_dir).eval()


@pytest.fixture
def prompt_ids() -> torch.Tensor:
    generator = torch.Generator().manual_seed(7)
    return torch.randint(0, 128, (1, 6), generator=generator)


def build_slice(source: ModelSource, start: int, end: int, **stages: bool) -> ShardModule:
    shard = ShardSpec(
        model_id="tiny/llama",
        blocks=BlockRange(start, end),
        include_input_stage=stages.get("input", False),
        include_output_stage=stages.get("output", False),
    )
    return ShardModule.build(source=source, shard=shard)


def test_first_shard_hidden_states_match_reference(
    tiny_llama_source: ModelSource,
    llama_reference: LlamaForCausalLM,
    prompt_ids: torch.Tensor,
) -> None:
    shard = build_slice(tiny_llama_source, 0, 2, input=True)
    shard.create_session("s")

    output = shard.prefill("s", input_ids=prompt_ids)
    reference = llama_reference(prompt_ids, output_hidden_states=True)

    assert isinstance(output, ShardState)
    # Entry 2 of the hidden-state tuple is the input to block 2, i.e. the
    # output of blocks [0, 2).
    assert torch.allclose(output.hidden_states, reference.hidden_states[2], **TOLERANCE)
    assert output.context.sequence_lengths == (6,)


def test_middle_shard_prefill_matches_reference(
    tiny_llama_source: ModelSource,
    llama_reference: LlamaForCausalLM,
    prompt_ids: torch.Tensor,
) -> None:
    shard = build_slice(tiny_llama_source, 1, 3)
    shard.create_session("s")

    reference = llama_reference(prompt_ids, output_hidden_states=True)
    output = shard.prefill("s", hidden_states=reference.hidden_states[1])

    assert isinstance(output, ShardState)
    assert torch.allclose(output.hidden_states, reference.hidden_states[3], **TOLERANCE)
    assert output.context.past_length == 0
    assert output.context.sequence_lengths == (6,)


def test_middle_shard_multistep_decode_matches_reference(
    tiny_llama_source: ModelSource,
    llama_reference: LlamaForCausalLM,
    prompt_ids: torch.Tensor,
) -> None:
    shard = build_slice(tiny_llama_source, 1, 3)
    shard.create_session("s")

    ref_output = llama_reference(prompt_ids, output_hidden_states=True, use_cache=True)
    ref_cache = ref_output.past_key_values
    shard.prefill("s", hidden_states=ref_output.hidden_states[1])
    next_token = int(ref_output.logits[0, -1].argmax())

    for step in range(3):
        ref_output = llama_reference(
            torch.tensor([[next_token]]),
            past_key_values=ref_cache,
            use_cache=True,
            output_hidden_states=True,
        )
        ref_cache = ref_output.past_key_values

        output = shard.decode("s", hidden_states=ref_output.hidden_states[1])
        assert isinstance(output, ShardState)
        assert torch.allclose(output.hidden_states, ref_output.hidden_states[3], **TOLERANCE)
        assert output.context.past_length == 6 + step
        assert output.context.sequence_lengths == (7 + step,)

        next_token = int(ref_output.logits[0, -1].argmax())


def test_last_shard_logits_match_reference(
    tiny_llama_source: ModelSource,
    llama_reference: LlamaForCausalLM,
    prompt_ids: torch.Tensor,
) -> None:
    shard = build_slice(tiny_llama_source, 2, 4, output=True)
    shard.create_session("s")

    reference = llama_reference(prompt_ids, output_hidden_states=True)
    output = shard.prefill("s", hidden_states=reference.hidden_states[2])

    assert isinstance(output, LogitsOutput)
    assert torch.allclose(output.logits, reference.logits, **TOLERANCE)


def test_middle_shard_kv_cache_is_local_and_correctly_sized(
    tiny_llama_source: ModelSource,
    llama_reference: LlamaForCausalLM,
    prompt_ids: torch.Tensor,
) -> None:
    shard = build_slice(tiny_llama_source, 1, 3)
    shard.create_session("s")

    reference = llama_reference(prompt_ids, output_hidden_states=True)
    shard.prefill("s", hidden_states=reference.hidden_states[1])
    shard.decode("s", hidden_states=reference.hidden_states[1][:, -1:])

    session: ShardSession = shard.session("s")
    cache = session.kv_cache
    assert len(cache.layers) == 2  # only the retained blocks hold cache entries
    assert cache.get_seq_length() == session.sequence_length == 7
