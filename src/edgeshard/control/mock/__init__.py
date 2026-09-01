"""Mock Master: thin manifest-driven integration orchestrator (not production).

Spec 22: executes a provided deployment manifest — Docker network, runtime
configs, containers, readiness, generation, cleanup. No scheduling, no
profiling, no partition calculation.
"""

from edgeshard.control.mock.client import (
    RemoteGenerationDriver,
    RemotePipeline,
    VLLMClient,
    VLLMCompletion,
)
from edgeshard.control.mock.deployment import (
    SHARD_LISTEN_PORT,
    build_runtime_config_payload,
    host_model_path,
    network_name,
    write_runtime_configs,
)
from edgeshard.control.mock.manifest import (
    DeploymentManifest,
    ManifestError,
    ManifestModel,
    ManifestRuntime,
    ManifestShard,
    ManifestVLLM,
)
from edgeshard.control.mock.master import Deployment, MockMaster, MockMasterError

__all__ = [
    "SHARD_LISTEN_PORT",
    "Deployment",
    "DeploymentManifest",
    "ManifestError",
    "ManifestModel",
    "ManifestRuntime",
    "ManifestShard",
    "ManifestVLLM",
    "MockMaster",
    "MockMasterError",
    "RemoteGenerationDriver",
    "RemotePipeline",
    "VLLMClient",
    "VLLMCompletion",
    "build_runtime_config_payload",
    "host_model_path",
    "network_name",
    "write_runtime_configs",
]
