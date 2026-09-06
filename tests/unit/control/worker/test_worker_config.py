"""WorkerConfig tests (Phase 1 spec §26)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from edgeshard.control.worker.config import (
    DEFAULT_PROFILING_HOST,
    DEFAULT_PROFILING_PORT,
    WorkerConfig,
)
from edgeshard.runtime.model_store import DEFAULT_MODEL_ROOT

SPEC_EXAMPLE = """\
worker:
  identity_path: /var/lib/edgeshard/worker-id

  master:
    endpoint: 192.168.1.100:51000

  heartbeat_interval_s: 5

  reconnect:
    initial_delay_s: 1
    max_delay_s: 30

model_store:
  root: /data/edgeshard-models

runtime:
  discover_managed_containers: true

tls:
  enabled: false
"""


def test_spec_example_parses() -> None:
    config = WorkerConfig.model_validate(yaml.safe_load(SPEC_EXAMPLE))
    assert config.worker.identity_path == Path("/var/lib/edgeshard/worker-id")
    assert config.worker.master is not None
    assert config.worker.master.endpoint == "192.168.1.100:51000"
    assert config.worker.heartbeat_interval_s == 5.0
    assert config.worker.reconnect.initial_delay_s == 1.0
    assert config.worker.reconnect.max_delay_s == 30.0
    assert config.model_store.root == Path("/data/edgeshard-models")
    assert config.runtime.discover_managed_containers is True
    assert config.tls.enabled is False


def test_empty_config_gets_documented_defaults() -> None:
    config = WorkerConfig.model_validate({})
    assert config.worker.identity_path == Path("/var/lib/edgeshard/worker-id")
    assert config.worker.master is None
    assert config.worker.heartbeat_interval_s == 5.0
    assert config.worker.reconnect.initial_delay_s == 1.0
    assert config.worker.reconnect.max_delay_s == 30.0
    assert config.model_store.root == DEFAULT_MODEL_ROOT
    assert config.runtime.discover_managed_containers is True
    assert config.tls.enabled is False


def test_jetson_style_minimal_override() -> None:
    """Spec §26 Jetson example overrides only model_store.root."""
    config = WorkerConfig.model_validate(
        {"model_store": {"root": "/mnt/ssd/edgeshard-models"}}
    )
    assert config.model_store.root == Path("/mnt/ssd/edgeshard-models")
    assert config.worker == WorkerConfig().worker


def test_unknown_keys_rejected() -> None:
    with pytest.raises(ValidationError):
        WorkerConfig.model_validate({"worker": {"bogus": 1}})
    with pytest.raises(ValidationError):
        WorkerConfig.model_validate({"bogus_section": {}})


def test_non_positive_heartbeat_rejected() -> None:
    with pytest.raises(ValidationError):
        WorkerConfig.model_validate({"worker": {"heartbeat_interval_s": 0}})


def test_empty_master_endpoint_rejected() -> None:
    with pytest.raises(ValidationError):
        WorkerConfig.model_validate({"worker": {"master": {"endpoint": ""}}})


def test_reconnect_bounds_rejected() -> None:
    with pytest.raises(ValidationError):
        WorkerConfig.model_validate(
            {"worker": {"reconnect": {"initial_delay_s": 10, "max_delay_s": 1}}}
        )
    with pytest.raises(ValidationError):
        WorkerConfig.model_validate({"worker": {"reconnect": {"initial_delay_s": -1}}})


def test_tls_enabled_true_is_a_hard_error() -> None:
    """§47/§48: never silently serve an insecure channel as if it were secure."""
    with pytest.raises(ValidationError, match="not supported yet"):
        WorkerConfig.model_validate({"tls": {"enabled": True}})
    with pytest.raises(ValidationError, match="insecure channel as if it were secure"):
        WorkerConfig.model_validate({"tls": {"enabled": True}})


def test_refresh_interval_defaults() -> None:
    config = WorkerConfig.model_validate({})
    assert config.worker.capability_refresh_interval_s == 300.0
    assert config.model_store.inventory_refresh_interval_s == 300.0


@pytest.mark.parametrize(
    "payload",
    [
        {"worker": {"capability_refresh_interval_s": 0}},
        {"worker": {"capability_refresh_interval_s": -1.0}},
        {"model_store": {"inventory_refresh_interval_s": 0}},
    ],
)
def test_non_positive_refresh_intervals_rejected(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="must be positive"):
        WorkerConfig.model_validate(payload)


def test_declared_runtime_platforms_parse() -> None:
    """§15: platforms are operator-declared facts, not auto-guessed ones."""
    config = WorkerConfig.model_validate(
        {
            "runtime": {
                "platforms": [
                    {"backend": "vllm", "platform": "cuda", "image": "vllm:v0.6.3"},
                    {"backend": "edgeshard-shard", "platform": "cuda"},
                ]
            }
        }
    )
    assert [(p.backend, p.platform, p.image) for p in config.runtime.platforms] == [
        ("vllm", "cuda", "vllm:v0.6.3"),
        ("edgeshard-shard", "cuda", None),
    ]
    assert WorkerConfig.model_validate({}).runtime.platforms == []


@pytest.mark.parametrize(
    "platform",
    [
        {"backend": "", "platform": "cuda"},  # empty backend
        {"backend": "  ", "platform": "cuda"},  # whitespace-only backend
        {"backend": "vllm", "platform": ""},  # empty platform
        {"backend": "vllm", "platform": "cuda", "image": " "},  # blank image
        {"backend": "vllm"},  # missing platform
    ],
)
def test_invalid_runtime_platform_rejected(platform: dict[str, str]) -> None:
    with pytest.raises(ValidationError):
        WorkerConfig.model_validate({"runtime": {"platforms": [platform]}})


def test_duplicate_runtime_platform_rejected() -> None:
    entry = {"backend": "vllm", "platform": "cuda"}
    other_image = {"backend": "vllm", "platform": "cuda", "image": "vllm:v0.6.3"}
    with pytest.raises(ValidationError, match="duplicate declared runtime platform"):
        WorkerConfig.model_validate({"runtime": {"platforms": [entry, other_image]}})
    # A different (backend, platform) pair is not a duplicate.
    config = WorkerConfig.model_validate(
        {"runtime": {"platforms": [entry, {"backend": "vllm", "platform": "cpu"}]}}
    )
    assert len(config.runtime.platforms) == 2


def test_from_yaml_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "worker.yaml"
    path.write_text(SPEC_EXAMPLE, encoding="utf-8")
    config = WorkerConfig.from_yaml(path)
    assert config.worker.heartbeat_interval_s == 5.0


def test_from_yaml_empty_file_is_all_defaults(tmp_path: Path) -> None:
    path = tmp_path / "worker.yaml"
    path.write_text("", encoding="utf-8")
    assert WorkerConfig.from_yaml(path) == WorkerConfig()


def test_from_yaml_non_mapping_rejected(tmp_path: Path) -> None:
    path = tmp_path / "worker.yaml"
    path.write_text("- 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        WorkerConfig.from_yaml(path)


def test_from_yaml_missing_file_raises_oserror(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        WorkerConfig.from_yaml(tmp_path / "missing.yaml")


# -- profiling section (Phase 2 spec §41, additive) --------------------------


def test_profiling_defaults_disabled_with_listen_defaults() -> None:
    """Phase 1 configs keep working unchanged: profiling is opt-in (§41)."""
    config = WorkerConfig.model_validate({})
    assert config.profiling.enabled is False
    assert config.profiling.host == DEFAULT_PROFILING_HOST
    assert config.profiling.port == DEFAULT_PROFILING_PORT
    # The spec example carries no profiling section and still parses.
    assert WorkerConfig.model_validate(yaml.safe_load(SPEC_EXAMPLE)).profiling.enabled is False


def test_profiling_section_parses() -> None:
    config = WorkerConfig.model_validate(
        {"profiling": {"enabled": True, "host": "127.0.0.1", "port": 0}}
    )
    assert config.profiling.enabled is True
    assert config.profiling.host == "127.0.0.1"
    assert config.profiling.port == 0


def test_profiling_port_zero_allowed_for_os_chosen_bind() -> None:
    """``worker serve`` advertises the *bound* port, so 0 is a valid knob."""
    config = WorkerConfig.model_validate({"profiling": {"port": 0}})
    assert config.profiling.port == 0


@pytest.mark.parametrize(
    "payload",
    [
        {"profiling": {"host": ""}},
        {"profiling": {"port": -1}},
        {"profiling": {"port": 65_536}},
        {"profiling": {"bogus": 1}},
    ],
)
def test_invalid_profiling_section_rejected(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        WorkerConfig.model_validate(payload)
