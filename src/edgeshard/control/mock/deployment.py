"""Deployment material generation for the Mock Master (specs 22/23).

The master never writes host IP addresses into shard configs (spec 23):
downstream endpoints are Docker-network aliases
(``<runtime-id>:SHARD_LISTEN_PORT``), every generated config binds the
same in-network listen port, and only the entry runtime ever gets a
published host port.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from edgeshard.control.mock.manifest import DeploymentManifest, ManifestError
from edgeshard.runtime.config import ShardRuntimeConfig

SHARD_LISTEN_PORT = 50051
"""In-network listen port of every deployed shard runtime (spec 23 example)."""

MODEL_MOUNT = "/models"
"""Container-side model mount that manifest paths are written against (spec 21.1)."""


def network_name(execution_id: str) -> str:
    """Dedicated bridge network of one execution (spec 23)."""
    return f"edgeshard-exec-{execution_id}"


def host_model_path(manifest: DeploymentManifest, model_cache_dir: Path) -> Path:
    """Resolve the manifest's container model path on the host model cache.

    ``/models/<rel>`` inside the container corresponds to
    ``model_cache_dir/<rel>`` on the host (the driver mounts the cache
    directory at ``/models``).
    """
    container_path = manifest.model.path
    if not container_path.is_relative_to(MODEL_MOUNT):
        raise ManifestError(
            f"manifest model path {container_path.as_posix()!r} must live under "
            f"the container model mount {MODEL_MOUNT!r}"
        )
    return Path(model_cache_dir) / container_path.relative_to(MODEL_MOUNT)


def build_runtime_config_payload(
    manifest: DeploymentManifest, stage_index: int
) -> dict[str, Any]:
    """One stage's runtime config as a plain YAML-ready dict (spec 19.1 shape)."""
    runtimes = manifest.pipeline_runtimes()
    runtime = runtimes[stage_index]
    stage_count = len(runtimes)
    pipeline: dict[str, Any] = {"stage_index": stage_index, "stage_count": stage_count}
    if stage_index + 1 < stage_count:
        next_id = manifest.pipeline[stage_index + 1]
        pipeline["next_endpoint"] = f"{next_id}:{SHARD_LISTEN_PORT}"
    device: dict[str, Any] = {"type": runtime.device.type}
    if runtime.device.index is not None:
        device["index"] = runtime.device.index
    return {
        "runtime": {
            "backend": runtime.backend,
            "runtime_id": runtime.id,
            "execution_id": manifest.execution_id,
        },
        "model": {"id": manifest.model.id, "path": manifest.model.path.as_posix()},
        "shard": {
            "start_block": runtime.shard.start,
            "end_block": runtime.shard.end,
            "include_input_stage": runtime.shard.include_input_stage,
            "include_output_stage": runtime.shard.include_output_stage,
        },
        "pipeline": pipeline,
        "device": device,
        "inference": {"dtype": runtime.inference.dtype},
        "server": {"listen_host": "0.0.0.0", "listen_port": SHARD_LISTEN_PORT},
    }


def write_runtime_configs(
    manifest: DeploymentManifest, config_dir: Path
) -> dict[str, Path]:
    """Generate, validate, and write one runtime config per pipeline stage.

    Every generated payload must round-trip through
    :class:`ShardRuntimeConfig` before it is written: the master never
    emits a config the runtime would reject.
    """
    paths: dict[str, Path] = {}
    for stage_index, runtime_id in enumerate(manifest.pipeline):
        payload = build_runtime_config_payload(manifest, stage_index)
        ShardRuntimeConfig.model_validate(payload)
        path = Path(config_dir) / f"{runtime_id}.yaml"
        path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        paths[runtime_id] = path
    return paths
