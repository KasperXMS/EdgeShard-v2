"""0H gate: manifest-driven deployment works without manual Docker commands.

The Mock Master deploys a two-runtime shard pipeline from a
DeploymentManifest — it creates the execution network, generates the
runtime configs, launches the containers, and waits for readiness.
Generation over the entry endpoint matches the untouched host HF
reference; shutdown removes every container, the network, and the
generated configs. The partition-invariance E2E across partition
choices is the 0I gate.

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
from edgeshard.control.mock.deployment import network_name
from edgeshard.control.mock.manifest import DeploymentManifest
from edgeshard.control.mock.master import MockMaster
from edgeshard.runtime.model_store import ModelStore

REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE_TAG = "edgeshard/hf-shard:cpu-phase0-test"
EXECUTION_ID = "exec-mock-master"
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


def make_manifest(tiny_llama_dir: Path, cpu_image: str) -> DeploymentManifest:
    return DeploymentManifest.model_validate(
        {
            "execution_id": EXECUTION_ID,
            "model": {
                "id": "tiny/llama",
                "local_name": tiny_llama_dir.name,
            },
            "runtimes": [
                {
                    "id": "shard-0",
                    "backend": "edgeshard_shard",
                    "image": cpu_image,
                    "shard": {"start": 0, "end": 2, "include_input_stage": True},
                },
                {
                    "id": "shard-1",
                    "backend": "edgeshard_shard",
                    "image": cpu_image,
                    "shard": {"start": 2, "end": 4, "include_output_stage": True},
                },
            ],
            "pipeline": ["shard-0", "shard-1"],
        }
    )


async def test_manifest_driven_deployment_and_cleanup(
    cpu_image: str,
    tiny_llama_dir: Path,
    llama_reference: LlamaForCausalLM,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    work_dir = tmp_path_factory.mktemp("mock-master")
    master = MockMaster(
        docker_client=docker_sdk.from_env(),
        model_store=ModelStore(model_root=tiny_llama_dir.parent),
        work_dir=work_dir,
        default_image=cpu_image,
    )
    manifest = make_manifest(tiny_llama_dir, cpu_image)

    deployment = await master.deploy(manifest)
    container_ids = [handle.container_id for handle in deployment.handles]
    try:
        assert deployment.network == network_name(EXECUTION_ID)
        assert [handle.runtime_id for handle in deployment.handles] == [
            "shard-0",
            "shard-1",
        ]

        async with RemotePipeline(
            endpoint=deployment.entry_endpoint, execution_id=EXECUTION_ID
        ) as pipeline:
            await pipeline.create_session("s")

            # Prefill + decode hops match the untouched reference.
            output = await pipeline.prefill("s", torch.tensor([PROMPT]))
            ref_output = llama_reference(torch.tensor([PROMPT]), use_cache=True)
            ref_cache = ref_output.past_key_values
            assert torch.allclose(output.logits, ref_output.logits, **TOLERANCE)
            token = int(output.logits[0, -1].argmax())
            assert token == int(ref_output.logits[0, -1].argmax())
            for _ in range(2):
                output = await pipeline.decode("s", token_id=token)
                ref_output = llama_reference(
                    torch.tensor([[token]]), past_key_values=ref_cache, use_cache=True
                )
                ref_cache = ref_output.past_key_values
                assert torch.allclose(output.logits, ref_output.logits, **TOLERANCE)
                token = int(output.logits[0, -1].argmax())
                assert token == int(ref_output.logits[0, -1].argmax())
            await pipeline.close_session("s")

            # The generation driver produces reference-identical tokens.
            await pipeline.create_session("g")
            driver = RemoteGenerationDriver(pipeline)
            tokens = await driver.generate("g", torch.tensor([PROMPT]), max_new_tokens=3)
            assert tokens == reference_greedy(llama_reference, PROMPT, 3)
            await pipeline.close_session("g")
    finally:
        await master.shutdown(deployment)

    # Cleanup is real: containers, network, and generated configs are gone.
    docker_client = docker_sdk.from_env()
    for container_id in container_ids:
        with pytest.raises(docker_sdk.errors.NotFound):
            docker_client.containers.get(container_id)
    with pytest.raises(docker_sdk.errors.NotFound):
        docker_client.networks.get(deployment.network)
    assert not (work_dir / EXECUTION_ID).exists()
