"""ShardRuntimeServer composition: cascaded downstream readiness (spec 23).

Only the entry runtime is reachable from the host in a container
deployment, so a stage must not serve until its downstream chain answers
GetRuntimeInfo. The whole pipeline is therefore ready exactly when the
entry is.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from edgeshard.runtime.config import ShardRuntimeConfig
from edgeshard.runtime.shard_server import RuntimeStartupError, ShardRuntimeServer

EXECUTION_ID = "exec-cascade"


def stage_payload(
    *,
    runtime_id: str,
    stage_index: int,
    start_block: int,
    end_block: int,
    tiny_llama_dir: Path,
    include_input_stage: bool = False,
    include_output_stage: bool = False,
    next_endpoint: str | None = None,
) -> ShardRuntimeConfig:
    pipeline: dict[str, Any] = {"stage_index": stage_index, "stage_count": 2}
    if next_endpoint is not None:
        pipeline["next_endpoint"] = next_endpoint
    return ShardRuntimeConfig.model_validate(
        {
            "runtime": {
                "backend": "edgeshard_shard",
                "runtime_id": runtime_id,
                "execution_id": EXECUTION_ID,
            },
            "model": {"id": "tiny/llama", "path": str(tiny_llama_dir)},
            "shard": {
                "start_block": start_block,
                "end_block": end_block,
                "include_input_stage": include_input_stage,
                "include_output_stage": include_output_stage,
            },
            "pipeline": pipeline,
            "server": {"listen_host": "127.0.0.1", "listen_port": 0},
        }
    )


async def test_create_succeeds_once_downstream_is_ready(tiny_llama_dir: Path) -> None:
    final = await ShardRuntimeServer.create(
        stage_payload(
            runtime_id="stage-1",
            stage_index=1,
            start_block=2,
            end_block=4,
            include_output_stage=True,
            tiny_llama_dir=tiny_llama_dir,
        )
    )
    try:
        entry = await ShardRuntimeServer.create(
            stage_payload(
                runtime_id="stage-0",
                stage_index=0,
                start_block=0,
                end_block=2,
                include_input_stage=True,
                next_endpoint=f"127.0.0.1:{final.port}",
                tiny_llama_dir=tiny_llama_dir,
            )
        )
        assert entry.port > 0
        await entry.stop()
    finally:
        await final.stop()


async def test_create_fails_when_downstream_never_becomes_ready(
    tiny_llama_dir: Path,
) -> None:
    config = stage_payload(
        runtime_id="stage-0",
        stage_index=0,
        start_block=0,
        end_block=2,
        include_input_stage=True,
        next_endpoint="127.0.0.1:1",  # nothing listens there
        tiny_llama_dir=tiny_llama_dir,
    )
    with pytest.raises(RuntimeStartupError, match="not ready"):
        await ShardRuntimeServer.create(config, downstream_ready_timeout_s=0.5)
