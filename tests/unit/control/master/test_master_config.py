"""MasterConfig validation tests (Phase 1 spec §32, §45)."""

from __future__ import annotations

from pathlib import Path

import pytest

from edgeshard.control.master.config import (
    MasterConfig,
    MasterProfilingSection,
    MasterServeConfig,
)


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


# ---------------------------------------------------------------------------
# MasterServeConfig: the YAML layer `master serve` parses (spec §45, P1G)
# ---------------------------------------------------------------------------


def write_master_config(tmp_path: Path, extra: str = "") -> Path:
    path = tmp_path / "master.yaml"
    path.write_text(extra, encoding="utf-8")
    return path


def test_serve_config_defaults() -> None:
    config = MasterServeConfig()
    assert config.master.host == "0.0.0.0"
    assert config.master.port == 51_000
    assert config.tls.enabled is False
    assert config.to_master_config() == MasterConfig()


def test_serve_config_from_yaml(tmp_path: Path) -> None:
    path = write_master_config(
        tmp_path,
        """
master:
  host: 127.0.0.1
  port: 0
  heartbeat_interval_ms: 100
  suspect_after_ms: 200
  offline_after_ms: 400
  liveness_tick_ms: 5
tls:
  enabled: false
""",
    )
    config = MasterServeConfig.from_yaml(path)
    assert config.master.host == "127.0.0.1"
    assert config.master.port == 0  # OS-chosen; READY reports the bound port
    assert config.to_master_config() == MasterConfig(
        heartbeat_interval_ms=100,
        suspect_after_ms=200,
        offline_after_ms=400,
        liveness_tick_ms=5,
    )


def test_serve_config_empty_file_uses_defaults(tmp_path: Path) -> None:
    config = MasterServeConfig.from_yaml(write_master_config(tmp_path))
    assert config == MasterServeConfig()


def test_serve_config_tls_enabled_true_is_a_hard_error(tmp_path: Path) -> None:
    """§47/§48: TLS is unimplemented in Phase 1 — requesting it must refuse to
    start rather than silently serve insecure gRPC under a 'secure' flag."""
    path = write_master_config(tmp_path, "tls:\n  enabled: true\n")
    with pytest.raises(ValueError, match="insecure channel as if it were secure"):
        MasterServeConfig.from_yaml(path)


@pytest.mark.parametrize(
    "payload",
    [
        "master:\n  host: ''\n",  # empty host
        "master:\n  port: -1\n",  # negative port
        "master:\n  port: 65536\n",  # port out of range
        "master:\n  unknown_knob: 1\n",  # extra keys forbidden (§26 style)
        "unknown_section:\n  x: 1\n",
        # Timing invariants live in MasterConfig and must surface here too:
        "master:\n  offline_after_ms: 5000\n",  # below default suspect 10 s
        "not-a-mapping",
    ],
)
def test_serve_config_invalid_rejected(tmp_path: Path, payload: str) -> None:
    path = write_master_config(tmp_path, payload)
    with pytest.raises(ValueError):
        config = MasterServeConfig.from_yaml(path)
        config.to_master_config()


# ---------------------------------------------------------------------------
# MasterProfilingSection: the additive profiling-admin layer (Phase 2 §49)
# ---------------------------------------------------------------------------


def test_profiling_defaults_are_disabled_and_frozen_shape() -> None:
    config = MasterServeConfig()
    assert config.profiling.enabled is False
    assert config.profiling.admin_host == "0.0.0.0"
    assert config.profiling.admin_port == 51_001
    assert config.profiling.store_path == "edgeshard-profiles.sqlite3"


def test_profiling_section_from_yaml(tmp_path: Path) -> None:
    path = write_master_config(
        tmp_path,
        """
profiling:
  enabled: true
  admin_host: 127.0.0.1
  admin_port: 0
  store_path: /tmp/profiles.sqlite3
""",
    )
    config = MasterServeConfig.from_yaml(path)
    assert config.profiling.enabled is True
    assert config.profiling.admin_host == "127.0.0.1"
    assert config.profiling.admin_port == 0  # OS-chosen; READY reports it
    assert config.profiling.store_path == "/tmp/profiles.sqlite3"


@pytest.mark.parametrize(
    "section",
    [
        {"admin_host": ""},
        {"admin_port": -1},
        {"admin_port": 65_536},
        {"store_path": ""},
        {"bogus": 1},  # extra keys are forbidden (Phase 0 config discipline)
    ],
)
def test_profiling_section_rejects_invalid_values(
    section: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        MasterProfilingSection(**section)
