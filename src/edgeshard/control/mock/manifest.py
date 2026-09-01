"""Deployment manifest for the Mock Master (spec 22.2).

The manifest *provides* the partition; the Mock Master only validates it
(spec 22.1: no partition calculation, no scheduling, no profiling). Block
bounds use the project-wide half-open ``[start, end)`` convention (spec
8.3). Structural problems surface as ``pydantic.ValidationError`` from
``model_validate``; deployment-semantic problems raise
:class:`ManifestError`.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from edgeshard.model.errors import EdgeShardError
from edgeshard.runtime.config import DeviceSection, InferenceSection


class ManifestError(EdgeShardError):
    """A deployment manifest cannot be deployed as written."""


class ManifestShard(BaseModel):
    """Block range of one runtime; half-open ``[start, end)`` (spec 8.3)."""

    model_config = ConfigDict(extra="forbid")

    start: int
    end: int
    include_input_stage: bool = False
    include_output_stage: bool = False

    @model_validator(mode="after")
    def _check_bounds(self) -> Self:
        if self.start < 0:
            raise ValueError(f"shard start must be >= 0, got {self.start}")
        if self.end <= self.start:
            raise ValueError(
                f"shard range [{self.start}, {self.end}) must be non-empty"
            )
        return self


class ManifestModel(BaseModel):
    """Model identity; ``path`` is container-side (under ``/models``)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    path: Path

    @model_validator(mode="after")
    def _check_id(self) -> Self:
        if not self.id:
            raise ValueError("model id must be non-empty")
        return self


class ManifestRuntime(BaseModel):
    """One runtime of the deployment and the shard it executes."""

    model_config = ConfigDict(extra="forbid")

    id: str
    backend: Literal["edgeshard_shard"]
    image: str | None = None
    shard: ManifestShard
    device: DeviceSection = Field(default_factory=DeviceSection)
    inference: InferenceSection = Field(default_factory=InferenceSection)

    @model_validator(mode="after")
    def _check_id(self) -> Self:
        if not self.id:
            raise ValueError("runtime id must be non-empty")
        return self


class DeploymentManifest(BaseModel):
    """A complete deployment: model, runtimes, and pipeline order.

    Static topology validation (spec 22.1) happens at parse time: the
    pipeline must use every runtime exactly once, the partition must be
    contiguous starting at block 0, and the input/output stage flags must
    sit on the first/last pipeline runtime. The partition's upper bound
    against the real model is checked at deploy time, when the model
    layout is known.
    """

    model_config = ConfigDict(extra="forbid")

    execution_id: str
    model: ManifestModel
    runtimes: list[ManifestRuntime]
    pipeline: list[str]

    @model_validator(mode="after")
    def _check_topology(self) -> Self:
        if not self.execution_id:
            raise ValueError("execution_id must be non-empty")
        if not self.runtimes:
            raise ValueError("manifest needs at least one runtime")
        ids = [runtime.id for runtime in self.runtimes]
        duplicates = sorted({rid for rid in ids if ids.count(rid) > 1})
        if duplicates:
            raise ValueError(f"duplicate runtime ids: {duplicates}")
        if not self.pipeline:
            raise ValueError("pipeline must order all runtimes for execution")
        pipeline_duplicates = sorted(
            {rid for rid in self.pipeline if self.pipeline.count(rid) > 1}
        )
        if pipeline_duplicates:
            raise ValueError(f"pipeline lists runtimes more than once: {pipeline_duplicates}")
        by_id = {runtime.id: runtime for runtime in self.runtimes}
        unknown = sorted(set(self.pipeline) - set(ids))
        if unknown:
            raise ValueError(f"pipeline references unknown runtimes: {unknown}")
        unused = sorted(set(ids) - set(self.pipeline))
        if unused:
            raise ValueError(f"runtimes not used by the pipeline: {unused}")

        ordered = [by_id[rid] for rid in self.pipeline]
        if ordered[0].shard.start != 0:
            raise ValueError(
                f"the partition must start at block 0, got {ordered[0].shard.start}"
            )
        if not ordered[0].shard.include_input_stage:
            raise ValueError(
                f"first runtime {ordered[0].id!r} must include the input stage"
            )
        if not ordered[-1].shard.include_output_stage:
            raise ValueError(
                f"last runtime {ordered[-1].id!r} must include the output stage"
            )
        for runtime in ordered[1:]:
            if runtime.shard.include_input_stage:
                raise ValueError(
                    f"only the first runtime may include the input stage: {runtime.id!r}"
                )
        for runtime in ordered[:-1]:
            if runtime.shard.include_output_stage:
                raise ValueError(
                    f"only the last runtime may include the output stage: {runtime.id!r}"
                )
        for previous, current in pairwise(ordered):
            if current.shard.start != previous.shard.end:
                raise ValueError(
                    f"partition gap between {previous.id!r} (end {previous.shard.end}) "
                    f"and {current.id!r} (start {current.shard.start})"
                )
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> DeploymentManifest:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"malformed deployment manifest in {path}")
        return cls.model_validate(payload)

    def runtime(self, runtime_id: str) -> ManifestRuntime:
        for runtime in self.runtimes:
            if runtime.id == runtime_id:
                return runtime
        raise KeyError(f"unknown runtime id: {runtime_id!r}")

    def pipeline_runtimes(self) -> list[ManifestRuntime]:
        """Runtimes in pipeline (execution) order."""
        return [self.runtime(rid) for rid in self.pipeline]
