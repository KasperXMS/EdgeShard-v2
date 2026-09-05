"""End-to-end `master serve` + `worker serve` (spec §52 Test A, §56 P1G).

The literal acceptance test: two real OS processes started through the
public CLI. The Master binds an OS-chosen port and announces it in its
READY line; the Worker registers, heartbeats at the Master-dictated
cadence, and the Master log shows exactly one worker reaching ONLINE.
Killing the Worker then degrades it through SUSPECT to OFFLINE on the
Master's real-time clock (§32) while the Master keeps serving — the
registry entry is never deleted.

Output is pumped by a background thread per process so every wait has a
hard deadline even when a child goes silent (no readline hang).
"""

from __future__ import annotations

import os
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml

STARTUP_TIMEOUT_S = 180.0  # edgeshard.cli transitively imports torch; cold hosts are slow
LOG_TIMEOUT_S = 60.0
OFFLINE_TIMEOUT_S = 30.0

READY = re.compile(r"READY master endpoint=(\S+):(\d+)")
REGISTERED_WORKER_SIDE = re.compile(r"registered worker_id=(\S+)")
REGISTRATION = re.compile(r"registration worker_id=(\S+)")
HEARTBEAT = re.compile(r"heartbeat worker_id=(\S+) session_id=(\S+) sequence=(\d+)")


def launch(*args: str) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-m", "edgeshard", *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )


class ProcessLines:
    """Non-blocking line access to a subprocess's merged output."""

    def __init__(self, process: subprocess.Popen[str]) -> None:
        self._process = process
        self._queue: queue.Queue[str | None] = queue.Queue()
        self.lines: list[str] = []
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        assert self._process.stdout is not None
        for line in self._process.stdout:
            self._queue.put(line)
        self._queue.put(None)  # EOF

    def wait_for(self, pattern: re.Pattern[str], timeout_s: float) -> str:
        # Lines already buffered by earlier waits count too: patterns may
        # have arrived before anyone was looking for them.
        for line in self.lines:
            if pattern.search(line):
                return line
        deadline = time.monotonic() + timeout_s
        while True:
            if self._process.poll() is not None and self._queue.empty():
                raise RuntimeError(
                    f"process exited early (code {self._process.returncode}):\n"
                    + "".join(self.lines[-40:])
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"timed out waiting for {pattern.pattern!r}; last lines:\n"
                    + "".join(self.lines[-40:])
                )
            try:
                item = self._queue.get(timeout=min(remaining, 0.2))
            except queue.Empty:
                continue
            if item is None:
                raise RuntimeError(
                    f"process output ended (code {self._process.poll()}):\n"
                    + "".join(self.lines[-40:])
                )
            self.lines.append(item)
            if pattern.search(item):
                return item


def terminate(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def test_master_serve_and_worker_serve_end_to_end(tmp_path: Path) -> None:
    master_config = tmp_path / "master.yaml"
    master_config.write_text(
        yaml.safe_dump(
            {
                "master": {
                    "host": "127.0.0.1",
                    "port": 0,  # OS-chosen; the READY line reports the bound port
                    "heartbeat_interval_ms": 200,
                    "suspect_after_ms": 2_000,
                    "offline_after_ms": 4_000,
                    "liveness_tick_ms": 100,
                },
                "tls": {"enabled": False},
            }
        ),
        encoding="utf-8",
    )

    master = launch("master", "serve", "--config", str(master_config))
    master_lines = ProcessLines(master)
    worker: subprocess.Popen[str] | None = None
    worker_lines: ProcessLines | None = None
    try:
        ready = master_lines.wait_for(READY, STARTUP_TIMEOUT_S)
        host, port = READY.search(ready).groups()  # type: ignore[union-attr]
        assert host == "127.0.0.1"

        worker_config = tmp_path / "worker.yaml"
        worker_config.write_text(
            yaml.safe_dump(
                {
                    "worker": {
                        "identity_path": str(tmp_path / "worker-id"),
                        "master": {"endpoint": f"{host}:{port}"},
                        # The Master's 200 ms cadence must override this (§29).
                        "heartbeat_interval_s": 30,
                        "reconnect": {"initial_delay_s": 0.5, "max_delay_s": 2},
                    },
                    "model_store": {"root": str(tmp_path / "models")},
                    "runtime": {"discover_managed_containers": False},
                    "tls": {"enabled": False},
                }
            ),
            encoding="utf-8",
        )

        worker = launch("worker", "serve", "--config", str(worker_config))
        worker_lines = ProcessLines(worker)

        # Worker side: the §27 lifecycle reached "registered".
        worker_lines.wait_for(REGISTERED_WORKER_SIDE, STARTUP_TIMEOUT_S)

        # Master side: exactly one registration, one worker.
        line = master_lines.wait_for(REGISTRATION, LOG_TIMEOUT_S)
        (worker_id,) = REGISTRATION.search(line).groups()  # type: ignore[union-attr]
        persisted = (tmp_path / "worker-id").read_text(encoding="utf-8").strip()
        assert worker_id == persisted

        # Heartbeats flow at the Master cadence with advancing sequences.
        master_lines.wait_for(
            re.compile(r"heartbeat worker_id=\S+ session_id=\S+ sequence=2\b"),
            LOG_TIMEOUT_S,
        )
        heartbeats = [m for m in (HEARTBEAT.search(x) for x in master_lines.lines) if m]
        assert all(m.group(1) == worker_id for m in heartbeats)
        sequences = [int(m.group(3)) for m in heartbeats]
        assert sequences[0] == 1
        assert sequences == sorted(sequences)
        assert len([x for x in master_lines.lines if REGISTRATION.search(x)]) == 1

        # Liveness: the worker reached ONLINE in the Master's own log (§46).
        master_lines.wait_for(
            re.compile(rf"worker_id={worker_id} state=online"), LOG_TIMEOUT_S
        )

        # Kill the worker: real-time degradation SUSPECT → OFFLINE (§32).
        terminate(worker)
        master_lines.wait_for(
            re.compile(rf"worker_id={worker_id} state=suspect"), OFFLINE_TIMEOUT_S
        )
        master_lines.wait_for(
            re.compile(rf"worker_id={worker_id} state=offline"), OFFLINE_TIMEOUT_S
        )
        assert master.poll() is None  # the Master keeps serving
    finally:
        terminate(worker)
        terminate(master)
