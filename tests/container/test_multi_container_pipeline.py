"""0I gate (primary Phase 0 gate): multi-container pipeline via Mock Master.

Complete E2E through the Mock Master: three-stage and two-stage container
pipelines over the same tiny model. Both match the untouched host HF
reference, and changing the valid partition preserves the greedy output
(invariant 9). Containers, networks, and generated configs are cleaned up
after each deployment.

These tests need a reachable Docker daemon; without one they are skipped
explicitly (this development host has none — Tier 2/3 machines run them).
"""

from __future__ import annotations

from pathlib import Path

import docker as docker_sdk
import pytest
import torch
from transformers import LlamaForCausalLM

from edgeshard.control.mock.client import RemoteGenerationDriver, RemotePipeline
from edgeshard.control.mock.manifest import DeploymentManifest
from edgeshard.control.mock.master import MockMaster
from edgeshard.runtime.model_store import ModelStore

REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE_TAG = "edgeshard/hf-shard:cpu-phase0-test"
TOLERANCE = {"atol": 1e-6, "rtol": 1e-5}
PROMPT = [3, 7, 11, 19]
MAX_NEW_TOKENS = 5

#: execution id -> shard block layout (start, end, input stage, output stage)
PARTITIONS: dict[str, list[tuple[int, int, bool, bool]]] = {
    "exec-0i-three": [
        (0, 1, True, False),
        (1, 3, False, False),
        (3, 4, False, True),
    ],
    "exec-0i-two": [
        (0, 2, True, False),
        (2, 4, False, True),
    ],
}


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


def reference_greedy(reference: LlamaForCausalLM, prompt: list[int], count: int) -> list[int]:
    output = reference(torch.tensor([prompt]), use_cache=True)
    cache = output.past_key_values
    tokens: list[int] = []
    token = int(output.logits[0, -1].argmax())
    tokens.append(token)
    for _ in range(count - 1):
        output = reference(torch.tensor([[token]]), past_key_values=cache, use_cache=True)
        cache = output.past_key_values
        token = int(output.logits[0, -1].argmax())
        tokens.append(token)
    return tokens


def make_manifest(
    execution_id: str, tiny_llama_dir: Path, cpu_image: str
) -> DeploymentManifest:
    blocks = PARTITIONS[execution_id]
    runtimes = [
        {
            "id": f"shard-{index}",
            "backend": "edgeshard_shard",
            "image": cpu_image,
            "shard": {
                "start": start,
                "end": end,
                "include_input_stage": include_input,
                "include_output_stage": include_output,
            },
        }
        for index, (start, end, include_input, include_output) in enumerate(blocks)
    ]
    return DeploymentManifest.model_validate(
        {
            "execution_id": execution_id,
            "model": {
                "id": "tiny/llama",
                "local_name": tiny_llama_dir.name,
            },
            "runtimes": runtimes,
            "pipeline": [f"shard-{index}" for index in range(len(blocks))],
        }
    )


async def test_multi_container_pipeline_reference_and_partition_invariance(
    cpu_image: str,
    tiny_llama_dir: Path,
    llama_reference: LlamaForCausalLM,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    work_dir = tmp_path_factory.mktemp("multi-container")
    master = MockMaster(
        docker_client=docker_sdk.from_env(),
        model_store=ModelStore(model_root=tiny_llama_dir.parent),
        work_dir=work_dir,
        default_image=cpu_image,
    )
    reference_tokens = reference_greedy(llama_reference, PROMPT, MAX_NEW_TOKENS)
    tokens_by_partition: dict[str, list[int]] = {}

    for execution_id in PARTITIONS:
        deployment = await master.deploy(make_manifest(execution_id, tiny_llama_dir, cpu_image))
        container_ids = [handle.container_id for handle in deployment.handles]
        try:
            assert len(deployment.handles) == len(PARTITIONS[execution_id])

            async with RemotePipeline(
                endpoint=deployment.entry_endpoint, execution_id=execution_id
            ) as pipeline:
                # Prefill + decode match the untouched host reference.
                await pipeline.create_session("s")
                output = await pipeline.prefill("s", torch.tensor([PROMPT]))
                ref_output = llama_reference(torch.tensor([PROMPT]), use_cache=True)
                ref_cache = ref_output.past_key_values
                assert torch.allclose(output.logits, ref_output.logits, **TOLERANCE)
                token = int(output.logits[0, -1].argmax())
                assert token == int(ref_output.logits[0, -1].argmax())
                for _ in range(3):
                    output = await pipeline.decode("s", token_id=token)
                    ref_output = llama_reference(
                        torch.tensor([[token]]),
                        past_key_values=ref_cache,
                        use_cache=True,
                    )
                    ref_cache = ref_output.past_key_values
                    assert torch.allclose(output.logits, ref_output.logits, **TOLERANCE)
                    token = int(output.logits[0, -1].argmax())
                    assert token == int(ref_output.logits[0, -1].argmax())
                await pipeline.close_session("s")

                # Deterministic generation matches the reference greedy loop.
                await pipeline.create_session("g")
                driver = RemoteGenerationDriver(pipeline)
                tokens_by_partition[execution_id] = await driver.generate(
                    "g", torch.tensor([PROMPT]), max_new_tokens=MAX_NEW_TOKENS
                )
                await pipeline.close_session("g")
        finally:
            await master.shutdown(deployment)

        # Cleanup is real for every deployment.
        docker_client = docker_sdk.from_env()
        for container_id in container_ids:
            with pytest.raises(docker_sdk.errors.NotFound):
                docker_client.containers.get(container_id)
        with pytest.raises(docker_sdk.errors.NotFound):
            docker_client.networks.get(deployment.network)
        assert not (work_dir / execution_id).exists()

    # Container pipeline matches the HF reference...
    for tokens in tokens_by_partition.values():
        assert tokens == reference_tokens
    # ...and changing the valid partition preserves semantics (invariant 9).
    assert tokens_by_partition["exec-0i-three"] == tokens_by_partition["exec-0i-two"]
