"""Master configuration (Phase 1 spec §32, §45).

Liveness thresholds default to the spec values — heartbeat every 5 s,
SUSPECT after 10 s, OFFLINE after 20 s — and are configurable per §32.
Integration tests use shortened thresholds (spec §52 Test E) instead of
real-time sleeping (spec §37).

All durations are milliseconds to match the wire protocol's
``heartbeat_interval_ms``. :class:`MasterServeConfig` is the Pydantic YAML
layer ``edgeshard master serve`` (milestone P1G) parses; it validates only
shapes and hands the semantic core to :class:`MasterConfig`, whose
threshold invariants remain the single place cross-field timing rules live.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, model_validator


@dataclass(frozen=True)
class MasterConfig:
    """Master-side timing configuration (spec §32)."""

    heartbeat_interval_ms: int = 5_000
    """Interval the Master asks Workers to heartbeat at (RegisterWorkerResponse)."""

    suspect_after_ms: int = 10_000
    """No valid heartbeat for longer than this → SUSPECT (still ONLINE within it)."""

    offline_after_ms: int = 20_000
    """No valid heartbeat for longer than this → OFFLINE."""

    liveness_tick_ms: int = 1_000
    """Period of the LivenessManager evaluation task (spec §37)."""

    def __post_init__(self) -> None:
        if self.heartbeat_interval_ms <= 0:
            raise ValueError("heartbeat_interval_ms must be positive")
        if self.suspect_after_ms <= 0:
            raise ValueError("suspect_after_ms must be positive")
        if self.offline_after_ms <= self.suspect_after_ms:
            raise ValueError(
                "offline_after_ms must be greater than suspect_after_ms "
                f"(got {self.offline_after_ms} <= {self.suspect_after_ms})"
            )
        if self.liveness_tick_ms <= 0:
            raise ValueError("liveness_tick_ms must be positive")
        if self.suspect_after_ms < self.heartbeat_interval_ms:
            raise ValueError(
                "suspect_after_ms must be at least one heartbeat interval "
                "(a single missed heartbeat must not immediately degrade a Worker, spec §32)"
            )


DEFAULT_MASTER_HOST = "0.0.0.0"
DEFAULT_MASTER_PORT = 51_000
"""Control-plane listen defaults; port 0 (OS-chosen) is allowed so tests and
``master serve`` on a busy host can report the bound port in their READY line."""

DEFAULT_PROFILING_ADMIN_PORT = 51_001
"""Profiling-admin listen default (Phase 2 spec §49); port 0 allowed as above."""

DEFAULT_PROFILE_STORE_PATH = "edgeshard-profiles.sqlite3"
"""Where ``master serve`` persists the profile store when profiling is on (§43)."""


class MasterServeSection(BaseModel):
    """Listen address and timing knobs of ``edgeshard master serve`` (§45)."""

    model_config = ConfigDict(extra="forbid")

    host: str = DEFAULT_MASTER_HOST
    port: int = DEFAULT_MASTER_PORT
    heartbeat_interval_ms: int = 5_000
    suspect_after_ms: int = 10_000
    offline_after_ms: int = 20_000
    liveness_tick_ms: int = 1_000

    @model_validator(mode="after")
    def _check_listen(self) -> MasterServeSection:
        if not self.host:
            raise ValueError("master host must be non-empty")
        if not 0 <= self.port <= 65_535:
            raise ValueError(f"master port out of range: {self.port}")
        return self


class MasterProfilingSection(BaseModel):
    """Master-side profiling-admin hosting intent (Phase 2 spec §49).

    Additive: ``enabled=false`` (the default) keeps exact Phase 1 behavior —
    no admin server, no profile store, no extra files on disk. When enabled,
    ``master serve`` opens the SQLite profile store (§43) and hosts
    ``ProfilingAdminService`` for the CLI's ``profile`` commands.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    admin_host: str = DEFAULT_MASTER_HOST
    admin_port: int = DEFAULT_PROFILING_ADMIN_PORT
    store_path: str = DEFAULT_PROFILE_STORE_PATH

    @model_validator(mode="after")
    def _check_listen(self) -> MasterProfilingSection:
        if not self.admin_host:
            raise ValueError("profiling admin host must be non-empty")
        if not 0 <= self.admin_port <= 65_535:
            raise ValueError(f"profiling admin port out of range: {self.admin_port}")
        if not self.store_path:
            raise ValueError("profile store path must be non-empty")
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


class MasterServeConfig(BaseModel):
    """Complete Master serve configuration (spec §45).

    Cross-field timing invariants (offline > suspect >= heartbeat interval)
    are enforced by :meth:`to_master_config`, never duplicated here.
    """

    model_config = ConfigDict(extra="forbid")

    master: MasterServeSection = MasterServeSection()
    profiling: MasterProfilingSection = MasterProfilingSection()
    tls: TlsSection = TlsSection()

    @classmethod
    def from_yaml(cls, path: Path) -> MasterServeConfig:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if payload is None:
            payload = {}
        if not isinstance(payload, dict):
            raise ValueError(f"malformed master config in {path}")
        return cls.model_validate(payload)

    def to_master_config(self) -> MasterConfig:
        """Project onto the semantic core; threshold rules validate there."""
        return MasterConfig(
            heartbeat_interval_ms=self.master.heartbeat_interval_ms,
            suspect_after_ms=self.master.suspect_after_ms,
            offline_after_ms=self.master.offline_after_ms,
            liveness_tick_ms=self.master.liveness_tick_ms,
        )
