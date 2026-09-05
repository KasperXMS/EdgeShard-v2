"""MasterConfig validation tests (Phase 1 spec §32)."""

from __future__ import annotations

import pytest

from edgeshard.control.master.config import MasterConfig


def test_defaults_match_spec_32() -> None:
    config = MasterConfig()
    assert config.heartbeat_interval_ms == 5_000
    assert config.suspect_after_ms == 10_000
    assert config.offline_after_ms == 20_000


def test_thresholds_are_configurable() -> None:
    config = MasterConfig(
        heartbeat_interval_ms=100, suspect_after_ms=200, offline_after_ms=400
    )
    assert (config.heartbeat_interval_ms, config.suspect_after_ms) == (100, 200)
    assert config.offline_after_ms == 400


@pytest.mark.parametrize(
    "kwargs",
    [
        {"heartbeat_interval_ms": 0},
        {"suspect_after_ms": 0},
        {"liveness_tick_ms": 0},
        {"offline_after_ms": 10_000},  # equal to suspect_after: must exceed it
        {"suspect_after_ms": 25_000},  # beyond offline_after
        # A single missed heartbeat (2x interval) must not degrade a Worker:
        {"heartbeat_interval_ms": 6_000, "suspect_after_ms": 5_000},
    ],
)
def test_invalid_config_rejected(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        MasterConfig(**kwargs)
