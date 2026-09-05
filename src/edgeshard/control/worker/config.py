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

    @model_validator(mode="after")
    def _check_heartbeat(self) -> WorkerSection:
        if self.heartbeat_interval_s <= 0:
            raise ValueError("heartbeat_interval_s must be positive")
        return self


class ModelStoreSection(BaseModel):
    """Worker-local ModelStore location (spec §3.5)."""

    model_config = ConfigDict(extra="forbid")

    root: Path = DEFAULT_MODEL_ROOT


class RuntimeSection(BaseModel):
    """Runtime inventory behavior (spec §19)."""

    model_config = ConfigDict(extra="forbid")

    discover_managed_containers: bool = True


class TlsSection(BaseModel):
    """Transport security placeholder (spec §48): TLS lands later."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False


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
