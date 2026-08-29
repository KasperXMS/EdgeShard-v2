"""Session lifecycle and KV isolation (spec 12.3, 13).

Sessions are owned by the shard module; each carries its own native KV
cache. Lifecycle violations fail explicitly, and concurrent sessions must
not influence each other's results.
"""

from __future__ import annotations

import pytest
import torch

from edgeshard.inference.session import SessionError
from edgeshard.inference.shard import ShardModule
from edgeshard.inference.state import LogitsOutput
from edgeshard.model.source import ModelSource
from edgeshard.model.spec import BlockRange, ShardSpec


@pytest.fixture
def full_llama_shard(tiny_llama_source: ModelSource) -> ShardModule:
    return ShardModule.build(
        source=tiny_llama_source,
        shard=ShardSpec(
            model_id="tiny/llama",
            blocks=BlockRange(0, 4),
            include_input_stage=True,
            include_output_stage=True,
        ),
    )


@pytest.fixture
def middle_llama_shard(tiny_llama_source: ModelSource) -> ShardModule:
    return ShardModule.build(
        source=tiny_llama_source,
        shard=ShardSpec(model_id="tiny/llama", blocks=BlockRange(1, 3)),
    )


def test_duplicate_session_is_rejected(full_llama_shard: ShardModule) -> None:
    full_llama_shard.create_session("s")
    with pytest.raises(SessionError, match="already exists"):
        full_llama_shard.create_session("s")


def test_unknown_and_closed_sessions_fail_explicitly(full_llama_shard: ShardModule) -> None:
    with pytest.raises(SessionError, match="no such session"):
        full_llama_shard.session("missing")
    with pytest.raises(SessionError, match="no such session"):
        full_llama_shard.close_session("missing")
    with pytest.raises(SessionError, match="no such session"):
        full_llama_shard.prefill("missing", input_ids=torch.zeros((1, 1), dtype=torch.long))

    full_llama_shard.create_session("s")
    full_llama_shard.close_session("s")
    with pytest.raises(SessionError, match="no such session"):
        full_llama_shard.session("s")
    # The id is released by close and can be reused.
    full_llama_shard.create_session("s")


def test_decode_requires_prior_prefill(full_llama_shard: ShardModule) -> None:
    full_llama_shard.create_session("s")
    with pytest.raises(SessionError, match="decode before prefill"):
        full_llama_shard.decode("s", token_id=0)


def test_prefill_is_only_the_first_step(full_llama_shard: ShardModule) -> None:
    ids = torch.tensor([[1, 2, 3]])
    full_llama_shard.create_session("s")
    full_llama_shard.prefill("s", input_ids=ids)
    with pytest.raises(SessionError, match="after step"):
        full_llama_shard.prefill("s", input_ids=ids)


def test_input_shard_rejects_hidden_states(full_llama_shard: ShardModule) -> None:
    full_llama_shard.create_session("s")
    hidden = torch.zeros((1, 2, 64))
    with pytest.raises(ValueError, match="input shard prefill"):
        full_llama_shard.prefill("s", hidden_states=hidden)
    with pytest.raises(ValueError, match="input shard prefill"):
        full_llama_shard.prefill("s", input_ids=torch.tensor([[1]]), hidden_states=hidden)
    with pytest.raises(ValueError, match="input shard prefill"):
        full_llama_shard.prefill("s")
    full_llama_shard.prefill("s", input_ids=torch.tensor([[1]]))
    with pytest.raises(ValueError, match="input shard decode"):
        full_llama_shard.decode("s", hidden_states=hidden[:, :1])


def test_middle_shard_rejects_token_inputs(middle_llama_shard: ShardModule) -> None:
    middle_llama_shard.create_session("s")
    hidden = torch.zeros((1, 2, 64))
    with pytest.raises(ValueError, match="hidden_states"):
        middle_llama_shard.prefill("s", input_ids=torch.tensor([[1]]))
    middle_llama_shard.prefill("s", hidden_states=hidden)
    with pytest.raises(ValueError, match="hidden_states"):
        middle_llama_shard.decode("s", token_id=5)


def test_concurrent_sessions_produce_identical_results(
    tiny_llama_source: ModelSource, full_llama_shard: ShardModule
) -> None:
    """Interleaved sessions must not perturb each other's KV state."""
    prompt_a = torch.tensor([[1, 5, 9, 14, 23]])
    prompt_b = torch.tensor([[42, 17, 3]])

    solo = ShardModule.build(
        source=tiny_llama_source,
        shard=full_llama_shard.shard_spec,
    )
    solo.create_session("a")
    solo_outputs: list[torch.Tensor] = []
    out = solo.prefill("a", input_ids=prompt_a)
    assert isinstance(out, LogitsOutput)
    solo_outputs.append(out.logits)
    token_a = int(out.logits[0, -1].argmax())
    for _ in range(3):
        out = solo.decode("a", token_id=token_a)
        assert isinstance(out, LogitsOutput)
        solo_outputs.append(out.logits)
        token_a = int(out.logits[0, -1].argmax())

    shared = full_llama_shard
    shared.create_session("a")
    shared.create_session("b")
    out_a = shared.prefill("a", input_ids=prompt_a)
    out_b = shared.prefill("b", input_ids=prompt_b)
    assert isinstance(out_a, LogitsOutput)
    assert isinstance(out_b, LogitsOutput)
    shared_outputs = [out_a.logits]
    token_a = int(out_a.logits[0, -1].argmax())
    token_b = int(out_b.logits[0, -1].argmax())
    for _ in range(3):
        out_b = shared.decode("b", token_id=token_b)
        out_a = shared.decode("a", token_id=token_a)
        assert isinstance(out_a, LogitsOutput)
        assert isinstance(out_b, LogitsOutput)
        shared_outputs.append(out_a.logits)
        token_a = int(out_a.logits[0, -1].argmax())
        token_b = int(out_b.logits[0, -1].argmax())

    # Session isolation must be exact on deterministic CPU FP32 execution.
    assert len(solo_outputs) == len(shared_outputs)
    for solo_logits, shared_logits in zip(solo_outputs, shared_outputs, strict=True):
        assert torch.equal(solo_logits, shared_logits)

    # Each session owns a distinct cache object.
    assert shared.session("a").kv_cache is not shared.session("b").kv_cache
