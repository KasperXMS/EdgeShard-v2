"""Deployment manifest for the Mock Master (spec 22.2).

The manifest *provides* the partition; the Mock Master only validates it
(spec 22.1: no partition calculation, no scheduling, no profiling). Block
bounds use the project-wide half-open ``[start, end)`` convention (spec
8.3). Structural problems surface as ``pydantic.ValidationError`` from
``model_validate``; deployment-semantic problems raise
:class:`ManifestError`.

Two runtime kinds coexist (spec 4): ``edgeshard_shard`` runtimes carry a
shard and line up in the pipeline; ``vllm`` runtimes are independent
full-model servers (spec 4.7) that never join the shard pipeline. The
pipeline therefore orders exactly the shard runtimes, and a manifest may
deploy standalone runtimes only (e.g. vLLM alone).
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


class ManifestVLLM(BaseModel):
    """Optional vLLM engine arguments (spec 19.2).

    ``VLLMRuntimeDriver`` translates every set field into official vLLM
    CLI arguments; unset fields keep vLLM's own defaults.
    """

    model_config = ConfigDict(extra="forbid")

    max_model_len: int | None = None
    tensor_parallel_size: int | None = None

    @model_validator(mode="after")
    def _check_values(self) -> Self:
        if self.max_model_len is not None and self.max_model_len < 1:
            raise ValueError(f"max_model_len must be >= 1, got {self.max_model_len}")
        if self.tensor_parallel_size is not None and self.tensor_parallel_size < 1:
            raise ValueError(
                f"tensor_parallel_size must be >= 1, got {self.tensor_parallel_size}"
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
    """One runtime of the deployment.

    ``edgeshard_shard`` runtimes require the shard they execute; ``vllm``
    runtimes serve the whole model and reject a shard section.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    backend: Literal["edgeshard_shard", "vllm"]
    image: str | None = None
    shard: ManifestShard | None = None
    vllm: ManifestVLLM | None = None
    device: DeviceSection = Field(default_factory=DeviceSection)
    inference: InferenceSection = Field(default_factory=InferenceSection)

    @model_validator(mode="after")
    def _check_backend_sections(self) -> Self:
        if not self.id:
            raise ValueError("runtime id must be non-empty")
        if self.backend == "edgeshard_shard":
            if self.shard is None:
                raise ValueError(
                    f"runtime {self.id!r}: backend edgeshard_shard requires "
                    f"a shard section"
                )
            if self.vllm is not None:
                raise ValueError(
                    f"runtime {self.id!r}: the vllm section is only valid "
                    f"on vllm runtimes"
                )
        elif self.shard is not None:
            raise ValueError(
                f"runtime {self.id!r}: backend vllm serves the full model; "
                f"a shard section is not allowed"
            )
        return self


class DeploymentManifest(BaseModel):
    """A complete deployment: model, runtimes, and pipeline order.

    Static topology validation (spec 22.1) happens at parse time: the
    pipeline orders every shard runtime exactly once, standalone runtimes
    stay out of it, the partition must be contiguous starting at block 0,
    and the input/output stage flags must sit on the first/last pipeline
    runtime. The partition's upper bound against the real model is checked
    at deploy time, when the model layout is known.
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
        by_id = {runtime.id: runtime for runtime in self.runtimes}
        shard_ids = sorted(
            runtime.id
            for runtime in self.runtimes
            if runtime.backend == "edgeshard_shard"
        )
        if not self.pipeline:
            if shard_ids:
                raise ValueError(
                    f"pipeline must order every edgeshard_shard runtime; "
                    f"missing: {shard_ids}"
                )
            return self  # standalone-only deployment (e.g. vLLM alone)
        pipeline_duplicates = sorted(
            {rid for rid in self.pipeline if self.pipeline.count(rid) > 1}
        )
        if pipeline_duplicates:
            raise ValueError(f"pipeline lists runtimes more than once: {pipeline_duplicates}")
        unknown = sorted(set(self.pipeline) - set(ids))
        if unknown:
            raise ValueError(f"pipeline references unknown runtimes: {unknown}")
        non_shard = sorted(
            {rid for rid in self.pipeline if by_id[rid].backend != "edgeshard_shard"}
        )
        if non_shard:
            raise ValueError(
                f"pipeline orders shard runtimes only; non-shard runtimes: {non_shard}"
            )
        missing = sorted(set(shard_ids) - set(self.pipeline))
        if missing:
            raise ValueError(f"edgeshard_shard runtimes not in the pipeline: {missing}")

        ordered: list[tuple[str, ManifestShard]] = []
        for rid in self.pipeline:
            shard = by_id[rid].shard
            if shard is None:  # unreachable: non-shard runtimes rejected above
                raise ValueError(f"runtime {rid!r} has no shard section")
            ordered.append((rid, shard))
        first_id, first_shard = ordered[0]
        last_id, last_shard = ordered[-1]
        if first_shard.start != 0:
            raise ValueError(
                f"the partition must start at block 0, got {first_shard.start}"
            )
        if not first_shard.include_input_stage:
            raise ValueError(
                f"first runtime {first_id!r} must include the input stage"
            )
        if not last_shard.include_output_stage:
            raise ValueError(
                f"last runtime {last_id!r} must include the output stage"
            )
        for rid, shard in ordered[1:]:
            if shard.include_input_stage:
                raise ValueError(
                    f"only the first runtime may include the input stage: {rid!r}"
                )
        for rid, shard in ordered[:-1]:
            if shard.include_output_stage:
                raise ValueError(
                    f"only the last runtime may include the output stage: {rid!r}"
                )
        for (previous_id, previous_shard), (current_id, current_shard) in pairwise(
            ordered
        ):
            if current_shard.start != previous_shard.end:
                raise ValueError(
                    f"partition gap between {previous_id!r} "
                    f"(end {previous_shard.end}) and {current_id!r} "
                    f"(start {current_shard.start})"
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
        """Shard runtimes in pipeline (execution) order."""
        return [self.runtime(rid) for rid in self.pipeline]

    def standalone_runtimes(self) -> list[ManifestRuntime]:
        """Runtimes outside the shard pipeline (e.g. vLLM), manifest order."""
        return [
            runtime
            for runtime in self.runtimes
            if runtime.backend != "edgeshard_shard"
        ]
