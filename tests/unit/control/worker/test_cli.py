"""CLI tests for the Phase 1 command groups (spec §42-43).

`worker inspect` runs for real against a tmp config: identity, host
discovery, and inventories all work without any Master or Docker daemon.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from edgeshard.cli import app

runner = CliRunner()


def write_worker_config(tmp_path: Path, extra: dict[str, object] | None = None) -> Path:
    payload: dict[str, object] = {
        "worker": {"identity_path": str(tmp_path / "worker-id")},
        "model_store": {"root": str(tmp_path / "models")},
        "runtime": {"discover_managed_containers": False},
    }
    if extra is not None:
        payload.update(extra)
    path = tmp_path / "worker.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_worker_inspect_json(tmp_path: Path) -> None:
    config = write_worker_config(tmp_path)
    result = runner.invoke(app, ["worker", "inspect", "--config", str(config)])
    assert result.exit_code == 0, result.output

    payload = json.loads(result.output)
    assert set(payload) == {"identity", "capability", "state"}
    assert payload["identity"]["worker_id"]
    assert payload["capability"]["architecture"]
    assert payload["capability"]["capability_revision"]
    assert payload["state"]["worker_id"] == payload["identity"]["worker_id"]


def test_worker_inspect_yaml(tmp_path: Path) -> None:
    config = write_worker_config(tmp_path)
    result = runner.invoke(
        app, ["worker", "inspect", "--config", str(config), "--format", "yaml"]
    )
    assert result.exit_code == 0, result.output
    payload = yaml.safe_load(result.output)
    assert set(payload) == {"identity", "capability", "state"}


def test_worker_inspect_persists_identity(tmp_path: Path) -> None:
    config = write_worker_config(tmp_path)
    first = runner.invoke(app, ["worker", "inspect", "--config", str(config)])
    second = runner.invoke(app, ["worker", "inspect", "--config", str(config)])
    assert first.exit_code == 0 and second.exit_code == 0
    worker_id_first = json.loads(first.output)["identity"]["worker_id"]
    worker_id_second = json.loads(second.output)["identity"]["worker_id"]
    assert worker_id_first == worker_id_second


def test_worker_inspect_identity_failure_exits_nonzero(tmp_path: Path) -> None:
    # Identity path points at a directory: cannot load or create (spec §47).
    config = write_worker_config(
        tmp_path, extra={"worker": {"identity_path": str(tmp_path)}}
    )
    result = runner.invoke(app, ["worker", "inspect", "--config", str(config)])
    assert result.exit_code == 1


def test_worker_inspect_invalid_config_exits_nonzero(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"worker": {"bogus": 1}}), encoding="utf-8")
    result = runner.invoke(app, ["worker", "inspect", "--config", str(path)])
    assert result.exit_code == 1


def test_bare_invocation_shows_help() -> None:
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    assert "Usage" in result.output


def test_worker_group_lists_inspect() -> None:
    result = runner.invoke(app, ["worker", "--help"])
    assert result.exit_code == 0
    assert "inspect" in result.output


def test_worker_group_lists_serve() -> None:
    result = runner.invoke(app, ["worker", "--help"])
    assert result.exit_code == 0
    assert "serve" in result.output


def test_master_group_lists_serve() -> None:
    result = runner.invoke(app, ["master", "--help"])
    assert result.exit_code == 0
    assert "serve" in result.output


def test_runtime_group_lists_serve() -> None:
    result = runner.invoke(app, ["runtime", "--help"])
    assert result.exit_code == 0
    assert "serve" in result.output


def test_legacy_serve_alias_routes_to_runtime_serve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Path] = []

    def fake_runtime_serve(config: Path) -> None:
        calls.append(config)

    monkeypatch.setattr("edgeshard.cli.runtime_serve", fake_runtime_serve)
    config_file = tmp_path / "runtime.yaml"
    config_file.write_text("runtime: {}\n", encoding="utf-8")
    result = runner.invoke(app, ["serve", "--config", str(config_file)])
    assert result.exit_code == 0, result.output
    assert calls == [config_file]


def test_legacy_bare_config_routes_to_runtime_serve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Path] = []

    def fake_runtime_serve(config: Path) -> None:
        calls.append(config)

    monkeypatch.setattr("edgeshard.cli.runtime_serve", fake_runtime_serve)
    config_file = tmp_path / "runtime.yaml"
    config_file.write_text("runtime: {}\n", encoding="utf-8")
    result = runner.invoke(app, ["--config", str(config_file)])
    assert result.exit_code == 0, result.output
    assert calls == [config_file]


# ---------------------------------------------------------------------------
# Serve error paths (spec §44-45, P1G): fail loudly before any I/O.
# Happy paths run in tests/integration/control as real processes.
# ---------------------------------------------------------------------------


def test_worker_serve_without_master_endpoint_exits_nonzero(tmp_path: Path) -> None:
    # §26: worker.master is optional for inspect but required for serve.
    config = write_worker_config(tmp_path)
    result = runner.invoke(app, ["worker", "serve", "--config", str(config)])
    assert result.exit_code == 1


def test_worker_serve_invalid_config_exits_nonzero(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"worker": {"bogus": 1}}), encoding="utf-8")
    result = runner.invoke(app, ["worker", "serve", "--config", str(path)])
    assert result.exit_code == 1


def test_master_serve_invalid_config_exits_nonzero(tmp_path: Path) -> None:
    path = tmp_path / "bad-master.yaml"
    path.write_text(yaml.safe_dump({"master": {"port": -1}}), encoding="utf-8")
    result = runner.invoke(app, ["master", "serve", "--config", str(path)])
    assert result.exit_code == 1


def test_master_serve_unsafe_thresholds_exit_nonzero(tmp_path: Path) -> None:
    # offline_after_ms below the default suspect threshold violates §32;
    # the semantic core rejects it before the server ever binds.
    path = tmp_path / "unsafe-master.yaml"
    path.write_text(
        yaml.safe_dump({"master": {"offline_after_ms": 5_000}}), encoding="utf-8"
    )
    result = runner.invoke(app, ["master", "serve", "--config", str(path)])
    assert result.exit_code == 1
