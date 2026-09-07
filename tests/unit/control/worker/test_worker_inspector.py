"""LocalWorkerInspector lifecycle tests (Phase 1 spec §27, §24, §16, §15).

The long-lived inspector must make ``worker serve`` heartbeats cheap:
static capability discovered once at startup and refreshed only on its
configured interval, telemetry sampled per inspect, ModelStore inventory
never fully re-walked every heartbeat, owned resources (Docker client,
default telemetry samplers such as the tegrastats reader) closed exactly
once, and injected resources left to their owner.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import edgeshard.control.worker.agent as agent_module
from edgeshard.cluster.capability import OSInfo
from edgeshard.cluster.state import MemoryPoolState, WorkerState
from edgeshard.control.worker.agent import LocalWorkerInspector
from edgeshard.control.worker.config import (
    ModelStoreSection,
    RuntimePlatformConfig,
    RuntimeSection,
    WorkerConfig,
    WorkerSection,
)
from edgeshard.control.worker.discovery.base import CapabilityFragment
from edgeshard.control.worker.telemetry.base import StateFragment


class FakeClock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def monotonic(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class CountingCapabilityProbe:
    """Static discovery stand-in that counts how often it is run."""

    def __init__(self) -> None:
        self.discover_calls = 0

    def discover(self) -> CapabilityFragment:
        self.discover_calls += 1
        return CapabilityFragment(
            architecture="x86_64", os=OSInfo(name="testos", version="1", kernel="1.0")
        )


class CountingTelemetryProbe:
    """Dynamic sampling stand-in; a well-behaved probe needs no close()."""

    def __init__(self) -> None:
        self.sample_calls = 0

    async def sample(self) -> StateFragment:
        self.sample_calls += 1
        return StateFragment()


class FreshCountingTelemetryProbe(CountingTelemetryProbe):
    def __init__(self) -> None:
        super().__init__()
        self.fresh_calls = 0

    async def sample(self) -> StateFragment:
        self.sample_calls += 1
        return StateFragment(
            memory_states=(MemoryPoolState("pool", 100),)
        )

    def sample_fresh(self) -> StateFragment:
        self.fresh_calls += 1
        return StateFragment(
            memory_states=(
                MemoryPoolState("pool", 100 - 10 * self.fresh_calls),
            )
        )


class ClosableTelemetryProbe(CountingTelemetryProbe):
    def __init__(self) -> None:
        super().__init__()
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1


class FakeDockerClient:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


def make_config(
    tmp_path: Path,
    *,
    capability_interval: float = 300.0,
    model_interval: float = 300.0,
    platforms: tuple[RuntimePlatformConfig, ...] = (),
) -> WorkerConfig:
    return WorkerConfig(
        worker=WorkerSection(
            identity_path=tmp_path / "worker-id",
            capability_refresh_interval_s=capability_interval,
        ),
        model_store=ModelStoreSection(
            root=tmp_path / "models",
            inventory_refresh_interval_s=model_interval,
        ),
        runtime=RuntimeSection(
            discover_managed_containers=False, platforms=list(platforms)
        ),
    )


def make_inspector(
    config: WorkerConfig,
    clock: FakeClock,
    capability_probe: CountingCapabilityProbe,
    telemetry_probe: CountingTelemetryProbe,
    docker_client: object = None,
) -> LocalWorkerInspector:
    return LocalWorkerInspector(
        config,
        capability_probes=(capability_probe,),
        telemetry_probes=(telemetry_probe,),
        docker_client=docker_client,
        monotonic=clock.monotonic,
    )


async def test_inspect_before_start_fails_loudly(tmp_path: Path) -> None:
    """§47: a programming error must surface, never yield half-built state."""
    inspector = make_inspector(
        make_config(tmp_path), FakeClock(), CountingCapabilityProbe(), CountingTelemetryProbe()
    )
    with pytest.raises(RuntimeError, match="before start"):
        await inspector.inspect()


async def test_start_is_idempotent(tmp_path: Path) -> None:
    clock = FakeClock()
    capability_probe = CountingCapabilityProbe()
    inspector = make_inspector(
        make_config(tmp_path), clock, capability_probe, CountingTelemetryProbe()
    )
    await inspector.start()
    await inspector.start()  # a duplicate start changes nothing
    assert capability_probe.discover_calls == 1
    await inspector.close()


async def test_static_capability_cached_while_telemetry_samples_every_beat(
    tmp_path: Path,
) -> None:
    """§27: discovery once at startup; every inspect pays only dynamic work."""
    clock = FakeClock()
    capability_probe = CountingCapabilityProbe()
    telemetry_probe = CountingTelemetryProbe()
    inspector = make_inspector(
        make_config(tmp_path), clock, capability_probe, telemetry_probe
    )
    await inspector.start()
    assert capability_probe.discover_calls == 1

    for beat in range(1, 6):
        clock.advance(5.0)  # heartbeat cadence, far below the 300 s interval
        inspection = await inspector.inspect()
        assert telemetry_probe.sample_calls == beat
        assert capability_probe.discover_calls == 1  # never re-discovered
        assert inspection.capability.capability_revision

    identity = inspection.identity
    state: WorkerState = inspection.state
    assert state.worker_id == identity.worker_id
    await inspector.close()


async def test_profiling_fresh_samples_reuse_live_inspector_backend(
    tmp_path: Path,
) -> None:
    probe = FreshCountingTelemetryProbe()
    inspector = make_inspector(
        make_config(tmp_path), FakeClock(), CountingCapabilityProbe(), probe
    )
    await inspector.start()
    initial = await inspector.inspect()

    first = inspector.sample_fresh_state()
    second = inspector.sample_fresh_state()

    assert initial.state.memory_states[0].available_bytes == 100
    assert first is not None and first.memory_states[0].available_bytes == 90
    assert second is not None and second.memory_states[0].available_bytes == 80
    assert probe.sample_calls == 1
    assert probe.fresh_calls == 2
    await inspector.close()


async def test_capability_refreshed_only_after_its_interval(tmp_path: Path) -> None:
    clock = FakeClock()
    capability_probe = CountingCapabilityProbe()
    inspector = make_inspector(
        make_config(tmp_path, capability_interval=60.0),
        clock,
        capability_probe,
        CountingTelemetryProbe(),
    )
    await inspector.start()
    assert capability_probe.discover_calls == 1

    clock.advance(59.9)
    await inspector.inspect()
    assert capability_probe.discover_calls == 1  # interval not yet elapsed

    clock.advance(0.2)  # now 60.1 s since discovery
    await inspector.inspect()
    assert capability_probe.discover_calls == 2

    clock.advance(1.0)
    await inspector.inspect()
    assert capability_probe.discover_calls == 2  # stamp reset by the refresh
    await inspector.close()


async def test_model_inventory_respects_its_refresh_interval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§27: the ModelStore is never fully re-walked on every heartbeat."""
    scan_calls = 0

    def counting_scan(store: object) -> tuple[()]:
        nonlocal scan_calls
        scan_calls += 1
        return ()

    monkeypatch.setattr(agent_module, "scan_model_inventory", counting_scan)

    clock = FakeClock()
    inspector = make_inspector(
        make_config(tmp_path, model_interval=120.0),
        clock,
        CountingCapabilityProbe(),
        CountingTelemetryProbe(),
    )
    await inspector.start()
    assert scan_calls == 1  # one full walk at startup

    for _ in range(10):
        clock.advance(5.0)  # 50 s of heartbeats inside the 120 s TTL
        await inspector.inspect()
    assert scan_calls == 1

    clock.advance(71.0)  # past the TTL
    await inspector.inspect()
    assert scan_calls == 2
    await inspector.close()


async def test_declared_runtime_platforms_are_merged(tmp_path: Path) -> None:
    """§15: operator-declared platforms reach the capability; never guessed."""
    platforms = (
        RuntimePlatformConfig(backend="vllm", platform="cuda", image="vllm:v0.6.3"),
        RuntimePlatformConfig(backend="edgeshard-shard", platform="cuda"),
    )
    clock = FakeClock()
    inspector = make_inspector(
        make_config(tmp_path, platforms=platforms),
        clock,
        CountingCapabilityProbe(),
        CountingTelemetryProbe(),
    )
    async with inspector:
        inspection = await inspector.inspect()
    declared = inspection.capability.runtime_platforms
    assert [(p.backend, p.platform, p.image) for p in declared] == [
        ("vllm", "cuda", "vllm:v0.6.3"),
        ("edgeshard-shard", "cuda", None),
    ]


async def test_owned_docker_client_closed_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§24: the client created at startup is closed at shutdown, once."""
    client = FakeDockerClient()
    monkeypatch.setattr(agent_module.docker, "from_env", lambda: client)

    inspector = LocalWorkerInspector(
        make_config(tmp_path),
        capability_probes=(CountingCapabilityProbe(),),
        telemetry_probes=(CountingTelemetryProbe(),),
    )
    await inspector.start()
    await inspector.inspect()
    assert client.close_calls == 0  # alive across heartbeats

    await inspector.close()
    await inspector.close()  # idempotent
    assert client.close_calls == 1


async def test_injected_docker_client_is_caller_owned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = FakeDockerClient()
    monkeypatch.setattr(
        agent_module.docker,
        "from_env",
        lambda: pytest.fail("injected client must replace from_env"),
    )
    inspector = LocalWorkerInspector(
        make_config(tmp_path),
        capability_probes=(CountingCapabilityProbe(),),
        telemetry_probes=(CountingTelemetryProbe(),),
        docker_client=client,
    )
    async with inspector:
        await inspector.inspect()
    assert client.close_calls == 0  # the injector closes it, not us


async def test_injected_telemetry_probes_are_caller_owned(tmp_path: Path) -> None:
    probe = ClosableTelemetryProbe()
    inspector = LocalWorkerInspector(
        make_config(tmp_path),
        capability_probes=(CountingCapabilityProbe(),),
        telemetry_probes=(probe,),
        docker_client=None,
    )
    async with inspector:
        await inspector.inspect()
    assert probe.sample_calls == 1
    assert probe.close_calls == 0  # injected samplers are not closed


async def test_jetson_default_backend_lives_for_the_whole_serve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§24: ONE tegrastats-backed backend for the entire ``worker serve``.

    The default Jetson telemetry backend is created at startup, sampled by
    every inspect (never re-spawned per heartbeat), and closed exactly once
    when the inspector shuts down.
    """
    class FakeJetsonBackend:
        def __init__(self, worker_id: str, **kwargs: object) -> None:
            self.worker_id = worker_id
            self.sample_calls = 0
            self.close_calls = 0
            created.append(self)

        async def sample(self) -> StateFragment:
            self.sample_calls += 1
            return StateFragment()

        async def close(self) -> None:
            self.close_calls += 1

    created: list[FakeJetsonBackend] = []

    monkeypatch.setattr(agent_module, "is_jetson_host", lambda: True)
    monkeypatch.setattr(agent_module, "JetsonTelemetryBackend", FakeJetsonBackend)

    inspector = LocalWorkerInspector(
        make_config(tmp_path),
        capability_probes=(CountingCapabilityProbe(),),
        docker_client=None,
    )
    await inspector.start()
    (backend,) = created  # exactly one backend for the whole lifecycle

    for beat in range(1, 4):
        await inspector.inspect()
        assert backend.sample_calls == beat

    await inspector.close()
    await inspector.close()  # idempotent
    assert backend.close_calls == 1
    assert len(created) == 1
