"""Runtime identity/info (spec 17.1 GetRuntimeInfo).

``RuntimeInfo`` is the runtime layer's domain view; the wire DTO is the
protobuf ``RuntimeInfoReply``. Mapping between the two is explicit and
lives here.
"""

from __future__ import annotations

from dataclasses import dataclass

from edgeshard.model.spec import BlockRange
from edgeshard.protocol.domain import PROTOCOL_VERSION
from edgeshard.protocol.pb import shard_runtime_pb2 as pb
from edgeshard.runtime.config import ShardRuntimeConfig


@dataclass(frozen=True)
class RuntimeInfo:
    """Static description of one shard runtime."""

    runtime_id: str
    model_id: str

    stage_index: int
    stage_count: int

    blocks: BlockRange
    include_input_stage: bool
    include_output_stage: bool

    protocol_version: int = PROTOCOL_VERSION


def runtime_info_from_config(config: ShardRuntimeConfig) -> RuntimeInfo:
    return RuntimeInfo(
        runtime_id=config.runtime.runtime_id,
        model_id=config.model.id,
        stage_index=config.pipeline.stage_index,
        stage_count=config.pipeline.stage_count,
        blocks=config.shard.blocks(),
        include_input_stage=config.shard.include_input_stage,
        include_output_stage=config.shard.include_output_stage,
    )


def runtime_info_to_wire(info: RuntimeInfo) -> pb.RuntimeInfoReply:
    return pb.RuntimeInfoReply(
        runtime_id=info.runtime_id,
        model_id=info.model_id,
        stage_index=info.stage_index,
        stage_count=info.stage_count,
        block_start=info.blocks.start,
        block_end=info.blocks.end,
        include_input_stage=info.include_input_stage,
        include_output_stage=info.include_output_stage,
        protocol_version=info.protocol_version,
    )


def runtime_info_from_wire(wire: pb.RuntimeInfoReply) -> RuntimeInfo:
    return RuntimeInfo(
        runtime_id=wire.runtime_id,
        model_id=wire.model_id,
        stage_index=wire.stage_index,
        stage_count=wire.stage_count,
        blocks=BlockRange(wire.block_start, wire.block_end),
        include_input_stage=wire.include_input_stage,
        include_output_stage=wire.include_output_stage,
        protocol_version=wire.protocol_version,
    )
