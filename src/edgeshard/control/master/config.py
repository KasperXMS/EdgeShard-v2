"""Master configuration (Phase 1 spec §32).

Liveness thresholds default to the spec values — heartbeat every 5 s,
SUSPECT after 10 s, OFFLINE after 20 s — and are configurable per §32.
Integration tests use shortened thresholds (spec §52 Test E) instead of
real-time sleeping (spec §37).

All durations are milliseconds to match the wire protocol's
``heartbeat_interval_ms``. YAML-based Master configuration arrives with
``master serve`` in milestone P1G; this dataclass is the semantic core it
will parse into.
"""

from __future__ import annotations

from dataclasses import dataclass


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
