"""WorkerConfig tests (Phase 1 spec §26)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from edgeshard.control.worker.config import WorkerConfig
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
