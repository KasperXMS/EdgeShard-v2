"""Worker Agent configuration (Phase 1 spec §26).

Pydantic parses and validates the YAML config only; parsed values flow
into cluster domain objects elsewhere - Pydantic classes never act as
cluster domain types (spec §26, §5).

All sections are optional so minimal configs work: the spec's Jetson
example overrides only ``model_store.root``. ``worker inspect`` (spec §43)
deliberately works without a Master, so ``worker.master`` stays optional
here; ``worker serve`` (milestone P1G) enforces its presence at startup.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from edgeshard.runtime.model_store import DEFAULT_MODEL_ROOT

DEFAULT_IDENTITY_PATH = Path("/var/lib/edgeshard/worker-id")
"""Production location of the persistent worker_id file (spec §10.1)."""


class ReconnectConfig(BaseModel):
    """Registration backoff bounds after a lost Master connection."""

    model_config = ConfigDict(extra="forbid")

    initial_delay_s: float = 1.0
    max_delay_s: float = 30.0

    @model_validator(mode="after")
    def _check_delays(self) -> ReconnectConfig:
        if self.initial_delay_s <= 0 or self.max_delay_s <= 0:
            raise ValueError("reconnect delays must be positive")
        if self.max_delay_s < self.initial_delay_s:
            raise ValueError("max_delay_s must be >= initial_delay_s")
        return self


class MasterConfig(BaseModel):
    """Master endpoint the Agent registers with."""

    model_config = ConfigDict(extra="forbid")

    endpoint: str

    @model_validator(mode="after")
    def _check_endpoint(self) -> MasterConfig:
        if not self.endpoint:
            raise ValueError("master endpoint must be non-empty")
        return self


class WorkerSection(BaseModel):
    """Worker-local Agent settings."""

    model_config = ConfigDict(extra="forbid")

    identity_path: Path = DEFAULT_IDENTITY_PATH
    master: MasterConfig | None = None
    heartbeat_interval_s: float = 5.0
    reconnect: ReconnectConfig = ReconnectConfig()
    capability_refresh_interval_s: float = 300.0
    """Static capability is discovered once at startup and cached; it is only
    re-discovered when a registration is refused for a stale revision or this
    interval elapses — never on every heartbeat (§16: capabilities rarely
    change, and re-probing NVML/Jetson/docker per beat is pure overhead)."""

    @model_validator(mode="after")
    def _check_intervals(self) -> WorkerSection:
        if self.heartbeat_interval_s <= 0:
            raise ValueError("heartbeat_interval_s must be positive")
        if self.capability_refresh_interval_s <= 0:
            raise ValueError("capability_refresh_interval_s must be positive")
        return self


class ModelStoreSection(BaseModel):
    """Worker-local ModelStore location (spec §3.5)."""

    model_config = ConfigDict(extra="forbid")

    root: Path = DEFAULT_MODEL_ROOT
    inventory_refresh_interval_s: float = 300.0
    """The model tree is re-scanned at most this often; a full recursive walk
    every heartbeat interval would hammer the disk for no benefit (§22)."""

    @model_validator(mode="after")
    def _check_interval(self) -> ModelStoreSection:
        if self.inventory_refresh_interval_s <= 0:
            raise ValueError("inventory_refresh_interval_s must be positive")
        return self


class RuntimePlatformConfig(BaseModel):
    """One operator-declared runtime platform the Worker can host (§15).

    Declared, never auto-guessed: the Worker reports the backends/images an
    operator has actually provisioned; discovery cannot infer which engines
    are *usable* from what happens to be installed.
    """

    model_config = ConfigDict(extra="forbid")

    backend: str
    platform: str
    image: str | None = None

    @model_validator(mode="after")
    def _check_fields(self) -> RuntimePlatformConfig:
        if not self.backend.strip():
            raise ValueError("runtime platform backend must be non-empty")
        if not self.platform.strip():
            raise ValueError("runtime platform platform must be non-empty")
        if self.image is not None and not self.image.strip():
            raise ValueError("runtime platform image must be non-empty when set")
        return self


class RuntimeSection(BaseModel):
    """Runtime inventory behavior (spec §19)."""

    model_config = ConfigDict(extra="forbid")

    discover_managed_containers: bool = True
    platforms: list[RuntimePlatformConfig] = []

    @model_validator(mode="after")
    def _check_platforms(self) -> RuntimeSection:
        seen: set[tuple[str, str]] = set()
        for entry in self.platforms:
            key = (entry.backend, entry.platform)
            if key in seen:
                raise ValueError(
                    f"duplicate declared runtime platform backend={key[0]!r} "
                    f"platform={key[1]!r}"
                )
            seen.add(key)
        return self


class TlsSection(BaseModel):
    """Transport security placeholder (spec §48): TLS lands later.

    Requesting TLS while it is unimplemented is a hard configuration error —
    silently serving insecure gRPC under ``tls.enabled=true`` would be a
    security lie (§47: fail loudly, never degrade silently).
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False

    @model_validator(mode="after")
    def _check_implemented(self) -> TlsSection:
        if self.enabled:
            raise ValueError(
                "tls.enabled=true is not supported yet: Phase 1 transport is "
                "insecure gRPC only (spec §48); refusing to start rather "
                "than serve an insecure channel as if it were secure"
            )
        return self


class WorkerConfig(BaseModel):
    """Complete Worker Agent configuration (spec §26)."""

    model_config = ConfigDict(extra="forbid")

    worker: WorkerSection = WorkerSection()
    model_store: ModelStoreSection = ModelStoreSection()
    runtime: RuntimeSection = RuntimeSection()
    tls: TlsSection = TlsSection()

    @classmethod
    def from_yaml(cls, path: Path) -> WorkerConfig:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if payload is None:
            payload = {}
        if not isinstance(payload, dict):
            raise ValueError(f"malformed worker config in {path}")
        return cls.model_validate(payload)
