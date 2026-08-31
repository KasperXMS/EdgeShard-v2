"""0G gate (L5-lite): host/container inference equivalence on the CPU image.

Builds the CPU correctness image (spec 21.2) from this repository, launches
one full-model shard runtime container through ``EdgeShardShardRuntimeDriver``,
and checks prefill + greedy decode against the untouched host HF reference.
Multi-container pipeline E2E is the 0I gate; container stop/cleanup covers
the spec 25.4 lifecycle item here.

These tests need a reachable Docker daemon; without one they are skipped
explicitly (this development host has none — Tier 2/3 machines run them).
"""

from __future__ import annotations

import uuid
from pathlib import Path

import docker as docker_sdk
import pytest
import torch
import yaml
from transformers import LlamaForCausalLM

from edgeshard.inference.state import ExecutionContext, InferencePhase
from edgeshard.protocol.domain import (
    MASTER_STAGE,
    PROTOCOL_VERSION,
    LogitsPayload,
    ShardMessage,
    ShardMessageHeader,
    TokenPayload,
)
from edgeshard.protocol.grpc_client import ShardRuntimeClient
from edgeshard.runtime.drivers.edgeshard_shard import (
    EdgeShardShardRuntimeDriver,
    EdgeShardShardRuntimeSpec,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE_TAG = "edgeshard/hf-shard:cpu-phase0-test"
EXECUTION_ID = "exec-container"
CONTAINER_PORT = 9100
TOLERANCE = {"atol": 1e-6, "rtol": 1e-5}
PROMPT = [3, 7, 11, 19]


def _docker_available() -> bool:
    try:
        docker_sdk.from_env().ping()
    except Exception:  # any docker problem means "unavailable"
        return False
    return True


pytestmark = [
    pytest.mark.container,
    pytest.mark.skipif(not _docker_available(), reason="no reachable Docker daemon"),
]


@pytest.fixture(scope="module")
def cpu_image() -> str:
    client = docker_sdk.from_env()
    client.images.build(
        path=str(REPO_ROOT),
        dockerfile="containers/hf/Dockerfile.cpu",
        tag=IMAGE_TAG,
        rm=True,
    )
    return IMAGE_TAG


@pytest.fixture(scope="module")
def llama_reference(tiny_llama_dir: Path) -> LlamaForCausalLM:
    return LlamaForCausalLM.from_pretrained(tiny_llama_dir).eval()


def write_container_config(config_dir: Path) -> Path:
    payload = {
        "runtime": {
            "backend": "edgeshard_shard",
            "runtime_id": "stage-0",
            "execution_id": EXECUTION_ID,
        },
        "model": {"id": "tiny/llama", "path": "/models"},
        "shard": {
            "start_block": 0,
            "end_block": 4,
            "include_input_stage": True,
            "include_output_stage": True,
        },
        "pipeline": {"stage_index": 0, "stage_count": 1},
        "server": {"listen_host": "0.0.0.0", "listen_port": CONTAINER_PORT},
    }
    path = config_dir / "runtime.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def container_message(
    session_id: str,
    token_ids: list[int],
    *,
    phase: InferencePhase,
    step: int,
    past_length: int,
) -> ShardMessage:
    return ShardMessage(
        header=ShardMessageHeader(
            protocol_version=PROTOCOL_VERSION,
            execution_id=EXECUTION_ID,
            session_id=session_id,
            request_id=uuid.uuid4().hex,
            phase=phase,
            step=step,
            source_stage=MASTER_STAGE,
            target_stage=0,
        ),
        context=ExecutionContext(
            phase=phase,
            step=step,
            batch_size=1,
            sequence_lengths=(past_length + len(token_ids),),
            past_length=past_length,
            positions=None,
        ),
        payload=TokenPayload(token_ids=tuple(token_ids)),
    )


async def test_container_runtime_matches_host_reference(
    cpu_image: str,
    tiny_llama_dir: Path,
    llama_reference: LlamaForCausalLM,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    config_dir = tmp_path_factory.mktemp("container-config")
    config_path = write_container_config(config_dir)

    driver = EdgeShardShardRuntimeDriver(
        docker_client=docker_sdk.from_env(),
        model_cache_dir=tiny_llama_dir,
    )
    spec = EdgeShardShardRuntimeSpec(
        backend="edgeshard_shard",
        runtime_id="stage-0",
        execution_id=EXECUTION_ID,
        image=cpu_image,
        config_path=config_path,
        host_port=0,
    )
    handle = await driver.start(spec)
    try:
        await driver.wait_ready(handle)
        info = await driver.info(handle)
        assert info.runtime_id == "stage-0"
        assert info.blocks.start == 0 and info.blocks.end == 4

        async with ShardRuntimeClient(handle.endpoint) as client:
            await client.create_session(EXECUTION_ID, "s")

            reply = await client.prefill(
                container_message(
                    "s", PROMPT, phase=InferencePhase.PREFILL, step=0, past_length=0
                )
            )
            assert isinstance(reply.payload, LogitsPayload)
            ref_output = llama_reference(torch.tensor([PROMPT]), use_cache=True)
            assert torch.allclose(reply.payload.logits, ref_output.logits, **TOLERANCE)
            token = int(reply.payload.logits[0, -1].argmax())
            assert token == int(ref_output.logits[0, -1].argmax())

            ref_cache = ref_output.past_key_values
            past = len(PROMPT)
            for wire_step in range(1, 3):
                reply = await client.decode(
                    container_message(
                        "s",
                        [token],
                        phase=InferencePhase.DECODE,
                        step=wire_step,
                        past_length=past,
                    )
                )
                assert isinstance(reply.payload, LogitsPayload)
                ref_output = llama_reference(
                    torch.tensor([[token]]), past_key_values=ref_cache, use_cache=True
                )
                ref_cache = ref_output.past_key_values
                assert torch.allclose(
                    reply.payload.logits, ref_output.logits, **TOLERANCE
                )
                token = int(reply.payload.logits[0, -1].argmax())
                assert token == int(ref_output.logits[0, -1].argmax())
                past += 1

            await client.close_session(EXECUTION_ID, "s")
    finally:
        await driver.stop(handle)

    # Cleanup is real: the container is gone after stop.
    with pytest.raises(docker_sdk.errors.NotFound):
        docker_sdk.from_env().containers.get(handle.container_id)
