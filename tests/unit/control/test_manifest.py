"""DeploymentManifest parsing and static topology validation (spec 22.2).

The manifest provides the partition; validation only checks it. All
structural rules fail at parse time with explicit messages — the Mock
Master never deploys a topology it could not validate (spec 22.1).
"""

from __future__ import annotations

from typing import Any

import pytest
import yaml

from edgeshard.control.mock.manifest import DeploymentManifest

SPEC_EXAMPLE = """
execution_id: exec-001

model:
  id: tiny-qwen
  path: /models/tiny-qwen

runtimes:
  - id: shard-0
    backend: edgeshard_shard
    shard:
      start: 0
      end: 2
      include_input_stage: true

  - id: shard-1
    backend: edgeshard_shard
    shard:
      start: 2
      end: 4
      include_output_stage: true

pipeline:
  - shard-0
  - shard-1
"""


def make_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "execution_id": "exec-1",
        "model": {"id": "tiny/llama", "path": "/models/tiny-llama"},
        "runtimes": [
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
        "pipeline": ["shard-0", "shard-1"],
    }
    payload.update(overrides)
    return payload


def test_spec_example_parses() -> None:
    manifest = DeploymentManifest.model_validate(yaml.safe_load(SPEC_EXAMPLE))
    assert manifest.execution_id == "exec-001"
    assert manifest.model.id == "tiny-qwen"
    assert manifest.model.path.as_posix() == "/models/tiny-qwen"
    assert [runtime.id for runtime in manifest.pipeline_runtimes()] == [
        "shard-0",
        "shard-1",
    ]
    first, last = manifest.pipeline_runtimes()
    assert (first.shard.start, first.shard.end) == (0, 2)
    assert first.shard.include_input_stage and not first.shard.include_output_stage
    assert (last.shard.start, last.shard.end) == (2, 4)
    assert last.shard.include_output_stage and not last.shard.include_input_stage
    assert manifest.runtime("shard-1").id == "shard-1"


def test_single_runtime_manifest() -> None:
    payload = make_payload(
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
            }
        ],
        pipeline=["shard-0"],
    )
    manifest = DeploymentManifest.model_validate(payload)
    assert manifest.pipeline_runtimes()[0].shard.include_input_stage


def test_from_yaml(tmp_path: Any) -> None:
    path = tmp_path / "manifest.yaml"
    path.write_text(SPEC_EXAMPLE, encoding="utf-8")
    manifest = DeploymentManifest.from_yaml(path)
    assert manifest.execution_id == "exec-001"


def test_from_yaml_rejects_non_mapping(tmp_path: Any) -> None:
    path = tmp_path / "manifest.yaml"
    path.write_text("- not\n- a mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed deployment manifest"):
        DeploymentManifest.from_yaml(path)


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"execution_id": ""}, "execution_id"),
        ({"runtimes": []}, "at least one runtime"),
        ({"pipeline": []}, "pipeline"),
        ({"pipeline": ["shard-0"]}, "not in the pipeline"),
        ({"pipeline": ["shard-0", "shard-1", "shard-2"]}, "unknown runtimes"),
        ({"pipeline": ["shard-0", "shard-0"]}, "more than once"),
    ],
)
def test_rejects_bad_structure(override: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        DeploymentManifest.model_validate(make_payload(**override))


def test_rejects_duplicate_runtime_ids() -> None:
    runtimes = [
        {
            "id": "shard-0",
            "backend": "edgeshard_shard",
            "shard": {"start": 0, "end": 2, "include_input_stage": True},
        },
        {
            "id": "shard-0",
            "backend": "edgeshard_shard",
            "shard": {"start": 2, "end": 4, "include_output_stage": True},
        },
    ]
    with pytest.raises(ValueError, match="duplicate runtime ids"):
        DeploymentManifest.model_validate(make_payload(runtimes=runtimes))


def test_rejects_unknown_backend() -> None:
    runtimes = [
        {
            "id": "shard-0",
            "backend": "sglang",
            "shard": {
                "start": 0,
                "end": 4,
                "include_input_stage": True,
                "include_output_stage": True,
            },
        }
    ]
    with pytest.raises(ValueError, match="backend"):
        DeploymentManifest.model_validate(
            make_payload(runtimes=runtimes, pipeline=["shard-0"])
        )


def test_mixed_manifest_orders_only_shard_runtimes() -> None:
    """The pipeline orders shard runtimes; vLLM runtimes stay standalone."""
    payload = make_payload(
        runtimes=[
            {
                "id": "shard-0",
                "backend": "edgeshard_shard",
                "shard": {"start": 0, "end": 2, "include_input_stage": True},
            },
            {"id": "vllm-0", "backend": "vllm"},
            {
                "id": "shard-1",
                "backend": "edgeshard_shard",
                "shard": {"start": 2, "end": 4, "include_output_stage": True},
            },
        ],
        pipeline=["shard-0", "shard-1"],
    )
    manifest = DeploymentManifest.model_validate(payload)
    assert [runtime.id for runtime in manifest.pipeline_runtimes()] == [
        "shard-0",
        "shard-1",
    ]
    assert [runtime.id for runtime in manifest.standalone_runtimes()] == ["vllm-0"]
    assert manifest.runtime("vllm-0").shard is None


def test_vllm_only_manifest_with_empty_pipeline() -> None:
    payload = make_payload(
        runtimes=[
            {
                "id": "vllm-0",
                "backend": "vllm",
                "device": {"type": "cuda", "index": 0},
                "vllm": {"max_model_len": 2048},
            }
        ],
        pipeline=[],
    )
    manifest = DeploymentManifest.model_validate(payload)
    assert manifest.pipeline_runtimes() == []
    (runtime,) = manifest.standalone_runtimes()
    assert runtime.backend == "vllm"
    assert runtime.vllm is not None
    assert runtime.vllm.max_model_len == 2048
    assert runtime.device.type == "cuda"


@pytest.mark.parametrize(
    ("override", "match"),
    [
        # vLLM serves the full model: a shard section is not allowed.
        (
            {
                "runtimes": [{"id": "vllm-0", "backend": "vllm",
                              "shard": {"start": 0, "end": 4}}],
                "pipeline": [],
            },
            "shard section is not allowed",
        ),
        # Shard runtimes require their shard section.
        (
            {
                "runtimes": [{"id": "shard-0", "backend": "edgeshard_shard"}],
                "pipeline": ["shard-0"],
            },
            "requires a shard section",
        ),
        # The vllm section belongs to vllm runtimes only.
        (
            {
                "runtimes": [
                    {
                        "id": "shard-0",
                        "backend": "edgeshard_shard",
                        "shard": {
                            "start": 0,
                            "end": 4,
                            "include_input_stage": True,
                            "include_output_stage": True,
                        },
                        "vllm": {"max_model_len": 512},
                    }
                ],
                "pipeline": ["shard-0"],
            },
            "only valid on vllm runtimes",
        ),
        # Standalone runtimes never join the shard pipeline.
        (
            {
                "runtimes": [
                    {
                        "id": "shard-0",
                        "backend": "edgeshard_shard",
                        "shard": {
                            "start": 0,
                            "end": 4,
                            "include_input_stage": True,
                            "include_output_stage": True,
                        },
                    },
                    {"id": "vllm-0", "backend": "vllm"},
                ],
                "pipeline": ["shard-0", "vllm-0"],
            },
            "non-shard runtimes",
        ),
    ],
)
def test_rejects_bad_backend_sections(override: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        DeploymentManifest.model_validate(make_payload(**override))


@pytest.mark.parametrize(
    "vllm_section",
    [{"max_model_len": 0}, {"tensor_parallel_size": 0}],
)
def test_rejects_bad_vllm_knobs(vllm_section: dict[str, Any]) -> None:
    payload = make_payload(
        runtimes=[{"id": "vllm-0", "backend": "vllm", "vllm": vllm_section}],
        pipeline=[],
    )
    with pytest.raises(ValueError, match="must be >= 1"):
        DeploymentManifest.model_validate(payload)


def _shard(
    start: int,
    end: int,
    *,
    input_stage: bool = False,
    output_stage: bool = False,
) -> dict[str, Any]:
    return {
        "start": start,
        "end": end,
        "include_input_stage": input_stage,
        "include_output_stage": output_stage,
    }


@pytest.mark.parametrize(
    ("runtimes", "match"),
    [
        # Partition must start at block 0.
        (
            [
                {"id": "a", "backend": "edgeshard_shard",
                 "shard": _shard(1, 4, input_stage=True, output_stage=True)},
            ],
            "start at block 0",
        ),
        # Gap between shards.
        (
            [
                {"id": "a", "backend": "edgeshard_shard",
                 "shard": _shard(0, 2, input_stage=True)},
                {"id": "b", "backend": "edgeshard_shard",
                 "shard": _shard(3, 4, output_stage=True)},
            ],
            "partition gap",
        ),
        # First runtime must include the input stage.
        (
            [
                {"id": "a", "backend": "edgeshard_shard", "shard": _shard(0, 4)},
            ],
            "input stage",
        ),
        # Last runtime must include the output stage.
        (
            [
                {"id": "a", "backend": "edgeshard_shard",
                 "shard": _shard(0, 4, input_stage=True)},
            ],
            "output stage",
        ),
        # Only the first runtime may include the input stage.
        (
            [
                {"id": "a", "backend": "edgeshard_shard",
                 "shard": _shard(0, 2, input_stage=True)},
                {"id": "b", "backend": "edgeshard_shard",
                 "shard": _shard(2, 4, input_stage=True, output_stage=True)},
            ],
            "input stage",
        ),
        # Only the last runtime may include the output stage.
        (
            [
                {"id": "a", "backend": "edgeshard_shard",
                 "shard": _shard(0, 2, input_stage=True, output_stage=True)},
                {"id": "b", "backend": "edgeshard_shard",
                 "shard": _shard(2, 4, output_stage=True)},
            ],
            "output stage",
        ),
    ],
)
def test_rejects_bad_topology(runtimes: list[dict[str, Any]], match: str) -> None:
    pipeline = [runtime["id"] for runtime in runtimes]
    with pytest.raises(ValueError, match=match):
        DeploymentManifest.model_validate(make_payload(runtimes=runtimes, pipeline=pipeline))


@pytest.mark.parametrize(
    ("shard", "match"),
    [
        ({"start": 2, "end": 2}, "non-empty"),
        ({"start": 3, "end": 2}, "non-empty"),
        ({"start": -1, "end": 2}, "start must be >= 0"),
    ],
)
def test_rejects_bad_shard_bounds(shard: dict[str, Any], match: str) -> None:
    runtimes = [
        {
            "id": "a",
            "backend": "edgeshard_shard",
            "shard": {**shard, "include_input_stage": True, "include_output_stage": True},
        }
    ]
    with pytest.raises(ValueError, match=match):
        DeploymentManifest.model_validate(make_payload(runtimes=runtimes, pipeline=["a"]))


def test_rejects_extra_fields() -> None:
    with pytest.raises(ValueError, match="scheduler"):
        DeploymentManifest.model_validate(make_payload(scheduler="round-robin"))
