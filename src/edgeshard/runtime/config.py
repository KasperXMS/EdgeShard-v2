"""Shard runtime configuration (spec 19.1).

The YAML shape mirrors the spec example. Block bounds use the same
half-open semantics as ``BlockRange`` (spec 8.3): ``[start_block,
end_block)``. This config describes the EdgeShard shard backend only; the
vLLM runtime gets its own spec when its driver lands.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from edgeshard.model.spec import BlockRange

_SUPPORTED_DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


class RuntimeSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: Literal["edgeshard_shard"]
    runtime_id: str
    execution_id: str

    @model_validator(mode="after")
    def _check_ids(self) -> RuntimeSection:
        if not self.runtime_id or not self.execution_id:
            raise ValueError("runtime_id and execution_id must be non-empty")
        return self


class ModelSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    path: Path


class ShardSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_block: int
    end_block: int
    include_input_stage: bool
    include_output_stage: bool

    def blocks(self) -> BlockRange:
        """Half-open block range (spec 8.3); raises on invalid bounds."""
        return BlockRange(self.start_block, self.end_block)


class PipelineSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage_index: int
    stage_count: int
    next_endpoint: str | None = None

    @model_validator(mode="after")
    def _check_stages(self) -> PipelineSection:
        if self.stage_count < 1:
            raise ValueError("stage_count must be >= 1")
        if not 0 <= self.stage_index < self.stage_count:
            raise ValueError("stage_index must be within [0, stage_count)")
        if self.stage_index < self.stage_count - 1 and not self.next_endpoint:
            raise ValueError("non-final stages require next_endpoint")
        if self.stage_index == self.stage_count - 1 and self.next_endpoint:
            raise ValueError("the final stage must not declare next_endpoint")
        return self


class DeviceSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["cpu", "cuda"] = "cpu"
    index: int | None = None

    def torch_device(self) -> torch.device:
        if self.type == "cpu":
            return torch.device("cpu")
        return torch.device(f"cuda:{self.index or 0}")


class InferenceSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dtype: Literal["fp32", "fp16", "bf16"] = "fp32"

    def torch_dtype(self) -> torch.dtype:
        return _SUPPORTED_DTYPES[self.dtype]


class ServerSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    listen_host: str = "127.0.0.1"
    listen_port: int

    @model_validator(mode="after")
    def _check_port(self) -> ServerSection:
        if not 0 <= self.listen_port <= 65535:
            raise ValueError("listen_port out of range")
        return self


class ShardRuntimeConfig(BaseModel):
    """Complete configuration of one EdgeShard shard runtime process."""

    model_config = ConfigDict(extra="forbid")

    runtime: RuntimeSection
    model: ModelSection
    shard: ShardSection
    pipeline: PipelineSection
    device: DeviceSection = DeviceSection()
    inference: InferenceSection = InferenceSection()
    server: ServerSection

    @classmethod
    def from_yaml(cls, path: Path) -> ShardRuntimeConfig:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"malformed runtime config in {path}")
        return cls.model_validate(payload)

    @property
    def is_final_stage(self) -> bool:
        return self.pipeline.stage_index == self.pipeline.stage_count - 1
