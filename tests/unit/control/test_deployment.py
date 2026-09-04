"""Runtime config generation for Mock Master deployments (specs 19/23).

Generated configs must use Docker-network aliases for downstream
endpoints, bind beyond loopback inside the container, and always
round-trip through ``ShardRuntimeConfig``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from edgeshard.control.mock.deployment import (
    SHARD_LISTEN_PORT,
    build_runtime_config_payload,
    host_model_path,
    network_name,
    write_runtime_configs,
)
from edgeshard.control.mock.manifest import DeploymentManifest
from edgeshard.runtime.config import ShardRuntimeConfig
from edgeshard.runtime.model_store import ModelStore


def make_manifest(
    *, local_name: str = "tiny-llama", runtimes: list[dict[str, Any]] | None = None
) -> DeploymentManifest:
    payload: dict[str, Any] = {
        "execution_id": "exec-gen",
        "model": {"id": "tiny/llama", "local_name": local_name},
        "runtimes": runtimes
        or [
            {
                "id": "shard-0",
                "backend": "edgeshard_shard",
                "shard": {"start": 0, "end": 2, "include_input_stage": True},
            },
            {
                "id": "shard-1",
                "backend": "edgeshard_shard",
                "shard": {"start": 2, "end": 4, "include_output_stage": True},
            },
        ],
        "pipeline": [runtime["id"] for runtime in (runtimes or [])] or ["shard-0", "shard-1"],
    }
    return DeploymentManifest.model_validate(payload)


def test_network_name() -> None:
    assert network_name("exec-001") == "edgeshard-exec-exec-001"


def test_entry_runtime_payload() -> None:
    manifest = make_manifest()
    payload = build_runtime_config_payload(manifest, 0)
    assert payload["runtime"] == {
        "backend": "edgeshard_shard",
        "runtime_id": "shard-0",
        "execution_id": "exec-gen",
    }
    assert payload["model"] == {"id": "tiny/llama", "path": "/models/tiny-llama"}
    assert payload["shard"] == {
        "start_block": 0,
        "end_block": 2,
        "include_input_stage": True,
        "include_output_stage": False,
    }
    # Downstream endpoint is a Docker-network alias; no host IP (spec 23).
    assert payload["pipeline"] == {
        "stage_index": 0,
        "stage_count": 2,
        "next_endpoint": f"shard-1:{SHARD_LISTEN_PORT}",
    }
    assert payload["device"] == {"type": "cpu"}
    assert payload["inference"] == {"dtype": "fp32"}
    assert payload["server"] == {"listen_host": "0.0.0.0", "listen_port": SHARD_LISTEN_PORT}


def test_final_runtime_payload_has_no_next_endpoint() -> None:
    manifest = make_manifest()
    payload = build_runtime_config_payload(manifest, 1)
    assert payload["pipeline"] == {"stage_index": 1, "stage_count": 2}
    assert payload["shard"]["include_output_stage"] is True


def test_device_and_inference_flow_into_payload() -> None:
    manifest = make_manifest(
        runtimes=[
            {
                "id": "shard-0",
                "backend": "edgeshard_shard",
                "shard": {
                    "start": 0,
                    "end": 4,
                    "include_input_stage": True,
                    "include_output_stage": True,
                },
                "device": {"type": "cuda", "index": 1},
                "inference": {"dtype": "bf16"},
            }
        ]
    )
    payload = build_runtime_config_payload(manifest, 0)
    assert payload["device"] == {"type": "cuda", "index": 1}
    assert payload["inference"] == {"dtype": "bf16"}


def test_generated_payloads_round_trip(tmp_path: Path) -> None:
    manifest = make_manifest()
    paths = write_runtime_configs(manifest, tmp_path)
    assert set(paths) == {"shard-0", "shard-1"}
    entry = ShardRuntimeConfig.from_yaml(paths["shard-0"])
    assert entry.pipeline.next_endpoint == f"shard-1:{SHARD_LISTEN_PORT}"
    final = ShardRuntimeConfig.from_yaml(paths["shard-1"])
    assert final.is_final_stage and final.pipeline.next_endpoint is None
    assert (tmp_path / "shard-0.yaml").exists()
    assert (tmp_path / "shard-1.yaml").exists()


def test_host_model_path_maps_local_name_into_the_store(tmp_path: Path) -> None:
    manifest = make_manifest(local_name="tiny-llama")
    store = ModelStore(model_root=tmp_path)
    assert host_model_path(manifest, store) == tmp_path / "tiny-llama"


def test_host_model_path_follows_each_store_root(tmp_path: Path) -> None:
    manifest = make_manifest(local_name="tiny-llama")
    store_a = ModelStore(model_root=tmp_path / "worker-a")
    store_b = ModelStore(model_root=tmp_path / "worker-b")
    assert host_model_path(manifest, store_a) == tmp_path / "worker-a" / "tiny-llama"
    assert host_model_path(manifest, store_b) == tmp_path / "worker-b" / "tiny-llama"
