"""BF16 numerical alignment: EdgeShard shards match the HF reference bitwise.

The adapter delegates block execution to the HF backbone forward and keeps
the rotary inverse-frequency table in float32 (as ``from_pretrained`` does),
so BF16 shard logits reproduce ``LlamaForCausalLM.from_pretrained(...,
torch_dtype=bfloat16)`` exactly — for a single full shard, a two-shard
split, the local pipeline, and a remote gRPC runtime alike (spec 25.5).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from transformers import LlamaForCausalLM

from edgeshard.control.mock.client import RemotePipeline
from edgeshard.inference.pipeline import LocalPipeline
from edgeshard.inference.shard import ShardModule
from edgeshard.inference.state import LogitsMode
from edgeshard.model.source import ModelSource
from edgeshard.model.spec import BlockRange, ShardSpec
from edgeshard.protocol.boundary import SerializedStageBoundary
from edgeshard.runtime.config import ShardRuntimeConfig
from edgeshard.runtime.shard_server import ShardRuntimeServer

EXECUTION_ID = "exec-bf16"


def bf16_reference(tiny_llama_dir: Path) -> LlamaForCausalLM:
    return LlamaForCausalLM.from_pretrained(
        tiny_llama_dir, torch_dtype=torch.bfloat16
    ).eval()


def build_shard(source: ModelSource, blocks: BlockRange, *, output: bool) -> ShardModule:
    return ShardModule.build(
        source=source,
        shard=ShardSpec(
            model_id="tiny/llama",
            blocks=blocks,
            include_input_stage=(blocks.start == 0),
            include_output_stage=output,
        ),
        dtype=torch.bfloat16,
    )


def build_local_pipeline(source: ModelSource, cuts: list[tuple[int, int]]) -> LocalPipeline:
    stages = [
        build_shard(
            source,
            BlockRange(start, end),
            output=(index == len(cuts) - 1),
        )
        for index, (start, end) in enumerate(cuts)
    ]
    return LocalPipeline(
        execution_id=EXECUTION_ID,
        stages=stages,
        transport=SerializedStageBoundary(execution_id=EXECUTION_ID),
    )


@pytest.fixture
def prompt_ids() -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260902)
    return torch.randint(0, 128, (1, 7), generator=generator)


def test_bf16_full_shard_matches_hf_reference_bitwise(
    tiny_llama_source: ModelSource, tiny_llama_dir: Path, prompt_ids: torch.Tensor
) -> None:
    """A BF16 full shard reproduces the BF16 HF reference exactly."""
    reference = bf16_reference(tiny_llama_dir)
    shard = build_shard(
        tiny_llama_source, BlockRange(0, 4), output=True
    )

    shard.create_session("s")
    output = shard.prefill("s", input_ids=prompt_ids)
    assert output.logits.dtype is torch.bfloat16
    assert torch.equal(output.logits, reference(prompt_ids).logits)

    # Greedy decode stays bitwise on the reference trajectory.
    ref_output = reference(prompt_ids, use_cache=True)
    ref_cache = ref_output.past_key_values
    token = int(output.logits[0, -1].argmax())
    for _ in range(3):
        output = shard.decode("s", token_id=token)
        ref_output = reference(
            torch.tensor([[token]]), past_key_values=ref_cache, use_cache=True
        )
        ref_cache = ref_output.past_key_values
        assert torch.equal(output.logits, ref_output.logits)
        token = int(output.logits[0, -1].argmax())


def test_bf16_single_shard_equals_two_shard_split(
    tiny_llama_source: ModelSource, prompt_ids: torch.Tensor
) -> None:
    """BF16 single-shard logits equal an uneven two-shard split bitwise."""
    single = build_local_pipeline(tiny_llama_source, [(0, 4)])
    split = build_local_pipeline(tiny_llama_source, [(0, 1), (1, 4)])

    single.create_session("s")
    split.create_session("s")
    single_logits = single.prefill("s", prompt_ids).logits
    split_logits = split.prefill("s", prompt_ids).logits
    assert single_logits.dtype is torch.bfloat16
    assert torch.equal(single_logits, split_logits)


def test_bf16_last_token_prefill_matches_full_projection(
    tiny_llama_source: ModelSource, prompt_ids: torch.Tensor
) -> None:
    """BF16 LAST_TOKEN prefill equals the final slice of the FULL prefill."""
    pipeline = build_local_pipeline(tiny_llama_source, [(0, 2), (2, 4)])
    pipeline.create_session("full")
    pipeline.create_session("last")
    full = pipeline.prefill("full", prompt_ids)
    last = pipeline.prefill("last", prompt_ids, logits_mode=LogitsMode.LAST_TOKEN)
    assert last.logits.shape == (1, 1, 128)
    assert torch.equal(last.logits, full.logits[:, -1:, :])


def bf16_runtime_config(tiny_llama_dir: Path) -> ShardRuntimeConfig:
    return ShardRuntimeConfig.model_validate(
        {
            "runtime": {
                "backend": "edgeshard_shard",
                "runtime_id": "stage-0",
                "execution_id": EXECUTION_ID,
            },
            "model": {"id": "tiny/llama", "path": str(tiny_llama_dir)},
            "shard": {
                "start_block": 0,
                "end_block": 4,
                "include_input_stage": True,
                "include_output_stage": True,
            },
            "pipeline": {"stage_index": 0, "stage_count": 1},
            "inference": {"dtype": "bf16"},
            "server": {"listen_host": "127.0.0.1", "listen_port": 0},
        }
    )


async def test_bf16_remote_runtime_matches_hf_reference_bitwise(
    tiny_llama_dir: Path, prompt_ids: torch.Tensor
) -> None:
    """A BF16 remote runtime answers bitwise-identical logits over gRPC."""
    reference = bf16_reference(tiny_llama_dir)
    server = await ShardRuntimeServer.create(bf16_runtime_config(tiny_llama_dir))
    try:
        async with RemotePipeline(
            endpoint=f"127.0.0.1:{server.port}", execution_id=EXECUTION_ID
        ) as pipeline:
            await pipeline.create_session("s")
            output = await pipeline.prefill("s", prompt_ids)
            assert output.logits.dtype is torch.bfloat16
            assert torch.equal(output.logits, reference(prompt_ids).logits)

            ref_output = reference(prompt_ids, use_cache=True)
            ref_cache = ref_output.past_key_values
            token = int(output.logits[0, -1].argmax())
            for _ in range(3):
                output = await pipeline.decode("s", token_id=token)
                ref_output = reference(
                    torch.tensor([[token]]), past_key_values=ref_cache, use_cache=True
                )
                ref_cache = ref_output.past_key_values
                assert torch.equal(output.logits, ref_output.logits)
                token = int(output.logits[0, -1].argmax())
            await pipeline.close_session("s")
    finally:
        await server.stop()
