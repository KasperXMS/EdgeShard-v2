"""L2 equivalence: whole-shard execution matches the HF reference model (spec 25).

The full-model shard (all blocks + input + output stages) must reproduce the
reference model's logits exactly within CPU FP32 tolerance, for both prefill
and multi-step greedy decode.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from transformers import LlamaForCausalLM, Qwen2ForCausalLM

from edgeshard.inference.shard import ShardModule
from edgeshard.inference.state import InferencePhase, LogitsOutput
from edgeshard.model.source import ModelSource
from edgeshard.model.spec import BlockRange, ShardSpec

#: CPU FP32 equivalence tolerance (spec 25.5); validated diffs are 0.0.
TOLERANCE = {"atol": 1e-6, "rtol": 1e-5}


def full_shard_spec(model_id: str, num_blocks: int) -> ShardSpec:
    return ShardSpec(
        model_id=model_id,
        blocks=BlockRange(0, num_blocks),
        include_input_stage=True,
        include_output_stage=True,
    )


@pytest.fixture
def prompt_ids() -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260829)
    return torch.randint(0, 128, (1, 7), generator=generator)


@pytest.fixture
def llama_reference(tiny_llama_dir: Path) -> LlamaForCausalLM:
    return LlamaForCausalLM.from_pretrained(tiny_llama_dir).eval()


@pytest.fixture
def full_llama_shard(tiny_llama_source: ModelSource) -> ShardModule:
    return ShardModule.build(
        source=tiny_llama_source,
        shard=full_shard_spec("tiny/llama", 4),
    )


def test_build_resolves_adapter_and_layout(full_llama_shard: ShardModule) -> None:
    assert full_llama_shard.layout.num_blocks == 4
    assert full_llama_shard.layout.model_type == "llama"
    assert full_llama_shard.device == torch.device("cpu")
    assert full_llama_shard.dtype is torch.float32


def test_prefill_matches_reference(
    full_llama_shard: ShardModule, llama_reference: LlamaForCausalLM, prompt_ids: torch.Tensor
) -> None:
    full_llama_shard.create_session("s")
    output = full_llama_shard.prefill("s", input_ids=prompt_ids)
    reference = llama_reference(prompt_ids)

    assert isinstance(output, LogitsOutput)
    assert torch.allclose(output.logits, reference.logits, **TOLERANCE)

    context = output.context
    assert context.phase is InferencePhase.PREFILL
    assert context.step == 1
    assert context.batch_size == 1
    assert context.sequence_lengths == (7,)
    assert context.past_length == 0
    assert context.positions is not None
    assert torch.equal(context.positions, torch.arange(7).unsqueeze(0))


def test_greedy_multistep_decode_matches_reference(
    full_llama_shard: ShardModule, llama_reference: LlamaForCausalLM, prompt_ids: torch.Tensor
) -> None:
    full_llama_shard.create_session("s")
    output = full_llama_shard.prefill("s", input_ids=prompt_ids)
    ref_output = llama_reference(prompt_ids, use_cache=True)
    ref_cache = ref_output.past_key_values

    assert isinstance(output, LogitsOutput)
    assert torch.allclose(output.logits, ref_output.logits, **TOLERANCE)
    token = int(output.logits[0, -1].argmax())
    assert token == int(ref_output.logits[0, -1].argmax())

    for step in range(4):
        output = full_llama_shard.decode("s", token_id=token)
        ref_output = llama_reference(
            torch.tensor([[token]]), past_key_values=ref_cache, use_cache=True
        )
        ref_cache = ref_output.past_key_values

        assert isinstance(output, LogitsOutput)
        assert torch.allclose(output.logits, ref_output.logits, **TOLERANCE)
        token = int(output.logits[0, -1].argmax())
        assert token == int(ref_output.logits[0, -1].argmax())

        context = output.context
        assert context.phase is InferencePhase.DECODE
        assert context.step == step + 2
        assert context.sequence_lengths == (8 + step,)
        assert context.past_length == 7 + step
        assert context.positions is not None
        assert torch.equal(context.positions, torch.tensor([[7 + step]]))


def test_qwen2_prefill_matches_reference(
    tiny_qwen2_source: ModelSource, tiny_qwen2_dir: Path, prompt_ids: torch.Tensor
) -> None:
    shard = ShardModule.build(
        source=tiny_qwen2_source,
        shard=full_shard_spec("tiny/qwen2", 4),
    )
    reference = Qwen2ForCausalLM.from_pretrained(tiny_qwen2_dir).eval()

    shard.create_session("s")
    output = shard.prefill("s", input_ids=prompt_ids)
    assert isinstance(output, LogitsOutput)
    assert torch.allclose(output.logits, reference(prompt_ids).logits, **TOLERANCE)
