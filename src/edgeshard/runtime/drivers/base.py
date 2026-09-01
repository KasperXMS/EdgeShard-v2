"""RuntimeDriver abstraction (spec 20).

Lifecycle normalization belongs on the control side: drivers start, probe,
describe, and stop runtimes. The native inference protocol of each backend
stays untouched (spec 4.7) — EdgeShard shards speak the ShardRuntime gRPC
service; vLLM keeps its OpenAI-compatible HTTP API (0J).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from edgeshard.model.errors import EdgeShardError
from edgeshard.runtime.info import RuntimeInfo


class DriverError(EdgeShardError):
    """Runtime lifecycle failure reported by a driver."""


@dataclass(frozen=True)
class RuntimeSpec:
    """Driver-agnostic request to launch one runtime."""

    backend: str
    runtime_id: str
    execution_id: str
    image: str


@dataclass(frozen=True)
class RuntimeHandle:
    """A started runtime and its primary endpoint.

    The endpoint is host-addressable when the driver published a port; for
    network-only runtimes (spec 23) it is resolvable only inside the
    deployment's Docker network.
    """

    runtime_id: str
    container_id: str
    endpoint: str


class RuntimeDriver(Protocol):
    """Normalized lifecycle across runtime backends (spec 20)."""

    async def start(self, spec: RuntimeSpec) -> RuntimeHandle:
        ...

    async def wait_ready(self, handle: RuntimeHandle) -> None:
        ...

    async def info(self, handle: RuntimeHandle) -> RuntimeInfo:
        ...

    async def stop(self, handle: RuntimeHandle) -> None:
        ...
