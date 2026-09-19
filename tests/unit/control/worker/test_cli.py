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
from edgeshard.control.worker.config import WorkerConfig

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


# ---------------------------------------------------------------------------
# Profiling-plane serve wiring (Phase 2 spec §41)
# ---------------------------------------------------------------------------


def test_profiling_advertise_host_uses_concrete_bind_host() -> None:
    from edgeshard.cli import _profiling_advertise_host

    assert _profiling_advertise_host("192.168.0.12") == "192.168.0.12"


def test_profiling_advertise_host_overrides_wildcard_bind() -> None:
    from edgeshard.cli import _profiling_advertise_host

    assert (
        _profiling_advertise_host("0.0.0.0", "192.168.0.12")
        == "192.168.0.12"
    )


@pytest.mark.parametrize("wildcard", ["0.0.0.0", "::", ""])
def test_profiling_wildcard_without_advertise_host_fails(wildcard: str) -> None:
    from edgeshard.cli import _profiling_advertise_host

    with pytest.raises(ValueError, match="advertise_host is required"):
        _profiling_advertise_host(wildcard)


def test_profiling_endpoint_uses_actual_bound_port() -> None:
    from edgeshard.cli import _worker_profiling_endpoint

    assert (
        _worker_profiling_endpoint("0.0.0.0", "192.168.0.12", 49_321)
        == "192.168.0.12:49321"
    )


async def test_profiling_disabled_keeps_plain_phase1_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from edgeshard.cli import _worker_serve

    calls: list[WorkerConfig] = []

    class FakeWorkerAgent:
        def __init__(self, config: WorkerConfig) -> None:
            calls.append(config)

        async def run(self) -> None:
            pass

    monkeypatch.setattr("edgeshard.cli.WorkerAgent", FakeWorkerAgent)
    config = WorkerConfig()

    await _worker_serve(config)

    assert calls == [config]


async def test_worker_serve_binds_wildcard_and_advertises_dialable_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from edgeshard.cli import _worker_serve

    events: dict[str, object] = {}
    call_order: list[str] = []

    class FakeInspector:
        def __init__(self, config: WorkerConfig) -> None:
            events["inspector_config"] = config

        async def start(self) -> None:
            call_order.append("inspector.start")

        async def close(self) -> None:
            call_order.append("inspector.close")

        def require_docker_client(self) -> object:
            return object()

    class FakeExecutor:
        def __init__(self, **kwargs: object) -> None:
            call_order.append("executor.construct")
            events["executor_kwargs"] = kwargs

        def cleanup_stale_containers(self) -> None:
            call_order.append("cleanup_stale_containers")
            events["stale_cleanup"] = True

    class FakeRunner:
        def __init__(self, **kwargs: object) -> None:
            events["runner_kwargs"] = kwargs

        async def shutdown(self) -> None:
            events["runner_stopped"] = True

    class FakeServer:
        async def stop(self, grace: object) -> None:
            events["server_stopped"] = grace

    class FakeWorkerAgent:
        worker_id = "worker-12"
        instance_id = "instance-12"
        registration_session_id = "registration-12"

        def __init__(
            self,
            config: WorkerConfig,
            *,
            inspector: object,
            profiling_endpoint: str,
        ) -> None:
            events["agent_config"] = config
            events["agent_inspector"] = inspector
            events["profiling_endpoint"] = profiling_endpoint

        async def run(self) -> None:
            call_order.append("agent.run")
            events["agent_ran"] = True

    async def fake_start_profiling_server(
        runner: object, *, host: str, port: int
    ) -> tuple[FakeServer, int]:
        events["server_runner"] = runner
        events["bind"] = (host, port)
        return FakeServer(), 49_321

    monkeypatch.setattr("edgeshard.cli.LocalWorkerInspector", FakeInspector)
    monkeypatch.setattr("edgeshard.cli.WorkerAgent", FakeWorkerAgent)
    monkeypatch.setattr(
        "edgeshard.cli.start_profiling_server", fake_start_profiling_server
    )
    monkeypatch.setattr(
        "edgeshard.control.worker.compute_executor.ContainerComputeProfilingExecutor",
        FakeExecutor,
    )
    monkeypatch.setattr(
        "edgeshard.control.worker.profiling_runner.WorkerProfilingRunner",
        FakeRunner,
    )
    config = WorkerConfig.model_validate(
        {
            "profiling": {
                "enabled": True,
                "host": "0.0.0.0",
                "port": 0,
                "advertise_host": "192.168.0.12",
            }
        }
    )

    await _worker_serve(config)

    assert events["bind"] == ("0.0.0.0", 0)
    assert events["profiling_endpoint"] == "192.168.0.12:49321"
    assert events["agent_ran"] is True
    assert events["stale_cleanup"] is True
    assert events["runner_stopped"] is True
    assert "server_stopped" in events
    assert call_order.index("inspector.start") < call_order.index(
        "executor.construct"
    )
    assert call_order.index("executor.construct") < call_order.index(
        "cleanup_stale_containers"
    )
    assert call_order.index("cleanup_stale_containers") < call_order.index(
        "agent.run"
    )
    assert call_order[-1] == "inspector.close"


@pytest.mark.parametrize("failure_stage", ("server", "agent_construct", "agent_run"))
async def test_worker_serve_closes_inspector_on_startup_failure(
    failure_stage: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from edgeshard.cli import _worker_serve

    events: list[str] = []

    class FakeInspector:
        def __init__(self, config: WorkerConfig) -> None:
            pass

        async def start(self) -> None:
            events.append("inspector.start")

        async def close(self) -> None:
            events.append("inspector.close")

        def require_docker_client(self) -> object:
            return object()

    class FakeExecutor:
        def __init__(self, **kwargs: object) -> None:
            pass

        def cleanup_stale_containers(self) -> None:
            events.append("cleanup_stale_containers")

    class FakeRunner:
        def __init__(self, **kwargs: object) -> None:
            pass

        async def shutdown(self) -> None:
            events.append("runner.shutdown")

    class FakeServer:
        async def stop(self, grace: object) -> None:
            events.append("server.stop")

    class FakeWorkerAgent:
        worker_id = "worker-12"
        instance_id = "instance-12"
        registration_session_id = "registration-12"

        def __init__(
            self,
            config: WorkerConfig,
            *,
            inspector: object,
            profiling_endpoint: str,
        ) -> None:
            events.append("agent.construct")
            if failure_stage == "agent_construct":
                raise RuntimeError("agent construction failed")

        async def run(self) -> None:
            events.append("agent.run")
            if failure_stage == "agent_run":
                raise RuntimeError("agent startup failed")

    async def fake_start_profiling_server(
        runner: object, *, host: str, port: int
    ) -> tuple[FakeServer, int]:
        events.append("server.start")
        if failure_stage == "server":
            raise RuntimeError("server startup failed")
        return FakeServer(), 49_321

    monkeypatch.setattr("edgeshard.cli.LocalWorkerInspector", FakeInspector)
    monkeypatch.setattr("edgeshard.cli.WorkerAgent", FakeWorkerAgent)
    monkeypatch.setattr(
        "edgeshard.cli.start_profiling_server", fake_start_profiling_server
    )
    monkeypatch.setattr(
        "edgeshard.control.worker.compute_executor.ContainerComputeProfilingExecutor",
        FakeExecutor,
    )
    monkeypatch.setattr(
        "edgeshard.control.worker.profiling_runner.WorkerProfilingRunner",
        FakeRunner,
    )
    config = WorkerConfig.model_validate(
        {
            "profiling": {
                "enabled": True,
                "host": "127.0.0.1",
                "port": 0,
            }
        }
    )

    with pytest.raises(RuntimeError):
        await _worker_serve(config)

    assert events[0:2] == ["inspector.start", "cleanup_stale_containers"]
    assert events[-1] == "inspector.close"
    assert events.count("inspector.close") == 1
    assert "runner.shutdown" in events
    if failure_stage == "server":
        assert "server.stop" not in events
    else:
        assert "server.stop" in events


def test_worker_serve_profiling_enabled_without_master_exits_nonzero(
    tmp_path: Path,
) -> None:
    """The enabled path binds the profiling server *before* the Agent is
    constructed; the missing master endpoint still fails loudly (§26/§47),
    and the command exits instead of serving a half-wired Worker."""
    config = write_worker_config(
        tmp_path,
        extra={
            "profiling": {
                "enabled": True,
                "host": "127.0.0.1",
                "port": 0,
            }
        },
    )
    result = runner.invoke(app, ["worker", "serve", "--config", str(config)])
    assert result.exit_code == 1
