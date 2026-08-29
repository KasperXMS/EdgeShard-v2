"""RuntimeInfo: config -> domain -> wire roundtrip (spec 17.1)."""

from __future__ import annotations

from pathlib import Path

import yaml

from edgeshard.protocol.domain import PROTOCOL_VERSION
from edgeshard.runtime.config import ShardRuntimeConfig
from edgeshard.runtime.info import (
    runtime_info_from_config,
    runtime_info_from_wire,
    runtime_info_to_wire,
)

CONFIG_PAYLOAD = {
    "runtime": {
        "backend": "edgeshard_shard",
        "runtime_id": "stage-2",
        "execution_id": "exec-1",
    },
    "model": {"id": "tiny/llama", "path": "/tmp/tiny-llama"},
    "shard": {
        "start_block": 3,
        "end_block": 4,
        "include_input_stage": False,
        "include_output_stage": True,
    },
    "pipeline": {"stage_index": 2, "stage_count": 3},
    "server": {"listen_port": 9002},
}


def test_info_matches_config(tmp_path: Path) -> None:
    path = tmp_path / "runtime.yaml"
    path.write_text(yaml.safe_dump(CONFIG_PAYLOAD), encoding="utf-8")
    config = ShardRuntimeConfig.from_yaml(path)

    info = runtime_info_from_config(config)
    assert info.runtime_id == "stage-2"
    assert info.model_id == "tiny/llama"
    assert info.stage_index == 2
    assert info.stage_count == 3
    assert info.blocks.start == 3 and info.blocks.end == 4
    assert not info.include_input_stage
    assert info.include_output_stage
    assert info.protocol_version == PROTOCOL_VERSION


def test_wire_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "runtime.yaml"
    path.write_text(yaml.safe_dump(CONFIG_PAYLOAD), encoding="utf-8")
    config = ShardRuntimeConfig.from_yaml(path)

    info = runtime_info_from_config(config)
    wire = runtime_info_to_wire(info)
    assert wire.block_start == 3 and wire.block_end == 4  # half-open on the wire
    assert runtime_info_from_wire(wire) == info
