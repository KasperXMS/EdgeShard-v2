"""End-to-end profiling plane through the public CLI (Phase 2 §51 P2G DoD).

The literal acceptance shape of test_serve_processes.py, extended for the
profiling plane: real OS processes for ``master serve`` (with the admin
plane enabled) and ``worker serve`` (with the profiling server enabled),
then real ``edgeshard profile ...`` CLI subprocesses talking to the
advertised admin endpoint over gRPC. Pinned:

* the Master announces the OS-bound admin port in its READY line (§49);
* the Worker binds its profiling server *before* registering and the
  registration advertises the bound endpoint (§41);
* ``profile snapshot`` renders the (empty) store as yaml, exit 0;
* honest rejections surface as nonzero exits with the refusal detail
  (§52.2): single-worker RTT plans to zero cases, unknown experiments
  are not found.

Heavy execution chains run in-process (tests/integration/profiling) —
here the point is the process/wiring contract, not benchmarking.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import yaml
from test_serve_processes import (
    LOG_TIMEOUT_S,
    READY,
    REGISTERED_WORKER_SIDE,
    REGISTRATION,
    STARTUP_TIMEOUT_S,
    ProcessLines,
    launch,
    terminate,
)

PROFILING_ADMIN_READY = re.compile(r"READY profiling-admin endpoint=(\S+):(\d+)")
PROFILING_LISTENING = re.compile(r"profiling service listening, advertising (\S+):(\d+)")

CLI_TIMEOUT_S = 120.0


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    """One short-lived ``edgeshard`` CLI invocation, captured to completion."""
    return subprocess.run(
        [sys.executable, "-m", "edgeshard", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=CLI_TIMEOUT_S,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        check=False,
    )


def test_profiling_serve_and_cli_end_to_end(tmp_path: Path) -> None:
    master_config = tmp_path / "master.yaml"
    master_config.write_text(
        yaml.safe_dump(
            {
                "master": {
                    "host": "127.0.0.1",
                    "port": 0,
                    "heartbeat_interval_ms": 200,
                    "suspect_after_ms": 2_000,
                    "offline_after_ms": 4_000,
                    "liveness_tick_ms": 100,
                },
                "profiling": {
                    "enabled": True,
                    "admin_host": "127.0.0.1",
                    "admin_port": 0,  # OS-chosen; the READY line reports it
                    "store_path": str(tmp_path / "profile.sqlite"),
                },
                "tls": {"enabled": False},
            }
        ),
        encoding="utf-8",
    )

    master = launch("master", "serve", "--config", str(master_config))
    master_lines = ProcessLines(master)
    worker: subprocess.Popen[str] | None = None
    try:
        ready = master_lines.wait_for(READY, STARTUP_TIMEOUT_S)
        host, port = READY.search(ready).groups()  # type: ignore[union-attr]

        admin_ready = master_lines.wait_for(PROFILING_ADMIN_READY, LOG_TIMEOUT_S)
        admin_host, admin_port = PROFILING_ADMIN_READY.search(  # type: ignore[union-attr]
            admin_ready
        ).groups()
        admin_endpoint = f"{admin_host}:{admin_port}"
        assert admin_host == "127.0.0.1"
        assert int(admin_port) > 0  # port 0 intent resolved to a bound port
        assert (tmp_path / "profile.sqlite").exists()  # the store opened (§43)

        worker_config = tmp_path / "worker.yaml"
        worker_config.write_text(
            yaml.safe_dump(
                {
                    "worker": {
                        "identity_path": str(tmp_path / "worker-id"),
                        "master": {"endpoint": f"{host}:{port}"},
                        "heartbeat_interval_s": 30,
                        "reconnect": {"initial_delay_s": 0.5, "max_delay_s": 2},
                    },
                    "model_store": {"root": str(tmp_path / "models")},
                    "runtime": {"discover_managed_containers": False},
                    "profiling": {
                        "enabled": True,
                        "host": "127.0.0.1",
                        "port": 0,  # OS-chosen; the agent advertises the bound one
                    },
                    "tls": {"enabled": False},
                }
            ),
            encoding="utf-8",
        )

        worker = launch("worker", "serve", "--config", str(worker_config))
        worker_lines = ProcessLines(worker)

        # §41 ordering: the profiling server binds BEFORE registration, so
        # by the time the Master logs the registration the endpoint is live.
        listening = worker_lines.wait_for(PROFILING_LISTENING, STARTUP_TIMEOUT_S)
        bound_host, bound_port = PROFILING_LISTENING.search(  # type: ignore[union-attr]
            listening
        ).groups()
        assert bound_host == "127.0.0.1"
        assert int(bound_port) > 0
        worker_lines.wait_for(REGISTERED_WORKER_SIDE, STARTUP_TIMEOUT_S)
        master_lines.wait_for(REGISTRATION, LOG_TIMEOUT_S)

        # CLI over the real admin plane: an empty store still snapshots (§46).
        snapshot = run_cli("profile", "snapshot", "--master", admin_endpoint)
        assert snapshot.returncode == 0, snapshot.stdout + snapshot.stderr
        payload = yaml.safe_load(snapshot.stdout)
        assert payload["snapshot_id"]
        assert payload["measurements"] == []

        # One worker has no directed peer pair: honest zero-case refusal (§52.2).
        rtt = run_cli("profile", "network", "rtt", "--master", admin_endpoint)
        assert rtt.returncode == 1
        assert "zero cases" in rtt.stdout + rtt.stderr

        # Unknown experiments are not found, never fabricated (§43).
        status = run_cli(
            "profile", "status", "--master", admin_endpoint, "--experiment", "bogus"
        )
        assert status.returncode == 1
        cancel = run_cli(
            "profile", "cancel", "--master", admin_endpoint, "--experiment", "bogus"
        )
        assert cancel.returncode == 1
    finally:
        terminate(worker)
        terminate(master)
