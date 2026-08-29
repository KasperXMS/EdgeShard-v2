"""Runtime config parsing/validation (spec 19.1)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import torch
from pydantic import ValidationError

from edgeshard.model.spec import BlockRange
from edgeshard.runtime.config import DeviceSection, InferenceSection, ShardRuntimeConfig

BASE_CONFIG: dict[str, Any] = {
    "runtime": {
        "backend": "edgeshard_shard",
        "runtime_id": "stage-1",
        "execution_id": "exec-1",
    },
    "model": {"id": "tiny/llama", "path": "/tmp/tiny-llama"},
    "shard": {
        "start_block": 1,
        "end_block": 3,
        "include_input_stage": False,
        "include_output_stage": False,
    },
    "pipeline": {
        "stage_index": 1,
        "stage_count": 3,
        "next_endpoint": "127.0.0.1:9002",
    },
    "server": {"listen_port": 9001},
}


def write_config(tmp_path: Path, payload: dict[str, Any]) -> Path:
    import yaml

    path = tmp_path / "runtime.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def mutate(payload: dict[str, Any], section: str, **fields: Any) -> dict[str, Any]:
    data = copy.deepcopy(payload)
    data[section].update(fields)
    return data


def test_full_config_parses(tmp_path: Path) -> None:
    config = ShardRuntimeConfig.from_yaml(write_config(tmp_path, BASE_CONFIG))
    assert config.runtime.backend == "edgeshard_shard"
    assert config.runtime.runtime_id == "stage-1"
    assert config.runtime.execution_id == "exec-1"
    assert config.model.id == "tiny/llama"
    assert config.shard.blocks() == BlockRange(1, 3)
    assert config.pipeline.stage_index == 1
    assert config.pipeline.next_endpoint == "127.0.0.1:9002"
    assert config.server.listen_host == "127.0.0.1"  # default
    assert config.server.listen_port == 9001
    assert not config.is_final_stage


def test_defaults_can_be_omitted(tmp_path: Path) -> None:
    config = ShardRuntimeConfig.from_yaml(write_config(tmp_path, BASE_CONFIG))
    assert config.device.type == "cpu"
    assert config.device.index is None
    assert config.inference.dtype == "fp32"


def test_half_open_block_range(tmp_path: Path) -> None:
    config = ShardRuntimeConfig.from_yaml(write_config(tmp_path, BASE_CONFIG))
    blocks = config.shard.blocks()
    assert blocks.start == 1 and blocks.end == 3
    assert len(blocks) == 2
    assert 1 in blocks and 2 in blocks and 3 not in blocks


def test_final_stage_property(tmp_path: Path) -> None:
    payload = mutate(BASE_CONFIG, "pipeline", stage_index=2, next_endpoint=None)
    payload["shard"] = {
        "start_block": 3,
        "end_block": 4,
        "include_input_stage": False,
        "include_output_stage": True,
    }
    config = ShardRuntimeConfig.from_yaml(write_config(tmp_path, payload))
    assert config.is_final_stage


def test_torch_dtype_mapping() -> None:
    assert InferenceSection(dtype="fp32").torch_dtype() is torch.float32
    assert InferenceSection(dtype="fp16").torch_dtype() is torch.float16
    assert InferenceSection(dtype="bf16").torch_dtype() is torch.bfloat16


def test_torch_device_mapping() -> None:
    assert DeviceSection().torch_device() == torch.device("cpu")
    assert DeviceSection(type="cuda").torch_device() == torch.device("cuda:0")
    assert DeviceSection(type="cuda", index=2).torch_device() == torch.device("cuda:2")


def test_non_final_stage_requires_next_endpoint(tmp_path: Path) -> None:
    payload = mutate(BASE_CONFIG, "pipeline", next_endpoint=None)
    with pytest.raises(ValidationError, match="next_endpoint"):
        ShardRuntimeConfig.from_yaml(write_config(tmp_path, payload))


def test_final_stage_forbids_next_endpoint(tmp_path: Path) -> None:
    payload = mutate(
        BASE_CONFIG, "pipeline", stage_index=2, next_endpoint="127.0.0.1:9999"
    )
    with pytest.raises(ValidationError, match="final stage"):
        ShardRuntimeConfig.from_yaml(write_config(tmp_path, payload))


def test_stage_index_must_fit_stage_count(tmp_path: Path) -> None:
    payload = mutate(BASE_CONFIG, "pipeline", stage_index=3)
    with pytest.raises(ValidationError, match="stage_index"):
        ShardRuntimeConfig.from_yaml(write_config(tmp_path, payload))


def test_stage_count_must_be_positive(tmp_path: Path) -> None:
    payload = mutate(BASE_CONFIG, "pipeline", stage_count=0, stage_index=0)
    with pytest.raises(ValidationError, match="stage_count"):
        ShardRuntimeConfig.from_yaml(write_config(tmp_path, payload))


def test_backend_is_fixed(tmp_path: Path) -> None:
    payload = mutate(BASE_CONFIG, "runtime", backend="vllm")
    with pytest.raises(ValidationError):
        ShardRuntimeConfig.from_yaml(write_config(tmp_path, payload))


def test_ids_must_be_non_empty(tmp_path: Path) -> None:
    payload = mutate(BASE_CONFIG, "runtime", runtime_id="")
    with pytest.raises(ValidationError, match="non-empty"):
        ShardRuntimeConfig.from_yaml(write_config(tmp_path, payload))


def test_listen_port_range(tmp_path: Path) -> None:
    payload = mutate(BASE_CONFIG, "server", listen_port=70000)
    with pytest.raises(ValidationError, match="listen_port"):
        ShardRuntimeConfig.from_yaml(write_config(tmp_path, payload))


def test_unknown_keys_are_rejected(tmp_path: Path) -> None:
    payload = copy.deepcopy(BASE_CONFIG)
    payload["unexpected"] = 1
    with pytest.raises(ValidationError):
        ShardRuntimeConfig.from_yaml(write_config(tmp_path, payload))
    payload = mutate(BASE_CONFIG, "shard", blocks="0-4")
    with pytest.raises(ValidationError):
        ShardRuntimeConfig.from_yaml(write_config(tmp_path, payload))


def test_invalid_block_bounds_raise_on_blocks(tmp_path: Path) -> None:
    payload = mutate(BASE_CONFIG, "shard", start_block=3, end_block=1)
    config = ShardRuntimeConfig.from_yaml(write_config(tmp_path, payload))
    with pytest.raises(ValueError, match="invalid block range"):
        config.shard.blocks()


def test_malformed_yaml_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "runtime.yaml"
    path.write_text("- not\n- a mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        ShardRuntimeConfig.from_yaml(path)
