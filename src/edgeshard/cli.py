"""Command-line entry point.

Phase 1 command groups (spec §42)::

    edgeshard runtime serve --config runtime.yaml
    edgeshard worker inspect [--config worker.yaml] [--format json|yaml]
    edgeshard worker serve --config worker.yaml
    edgeshard master serve --config master.yaml

The Phase 0 forms stay as compatibility aliases during Phase 1::

    edgeshard serve --config runtime.yaml        # == runtime serve
    edgeshard --config runtime.yaml              # historical bare form

Phase 2 adds the profiling operator surface (spec §49)::

    edgeshard profile model inspect --model ID [--config worker.yaml]
    edgeshard profile model run --master EP --model ID --dtype D --device DEV
    edgeshard profile operator run --master EP --model ID --dtype D --device DEV
    edgeshard profile network rtt --master EP
    edgeshard profile network bandwidth --master EP [--path-class wired_lan]
    edgeshard profile status --master EP --experiment ID
    edgeshard profile cancel --master EP --experiment ID
    edgeshard profile snapshot --master EP [--format yaml]

``worker inspect`` works without any Master (spec §43); it is the primary
local validation tool for Phase 1 hardware discovery. ``worker serve``
(spec §44) registers with the Master and heartbeats until terminated, and —
when ``profiling.enabled`` — hosts the WorkerProfilingService (Phase 2 spec
§41). ``master serve`` (spec §45) runs the WorkerRegistryService, liveness
tracking and debug snapshots — no REST, dashboard or scheduler yet — and,
when ``profiling.enabled`` (Phase 2 spec §49), hosts the ProfilingAdminService
the ``profile`` commands talk to. The CLI never drives domain logic itself:
``profile`` submits intents and reads back status/snapshots (§49).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, NoReturn

import grpc
import typer
import yaml

if TYPE_CHECKING:
    # Runtime-serve only; imported lazily so the CLI stays torch-free (§49).
    from edgeshard.runtime.config import ShardRuntimeConfig

from edgeshard.control.master.config import MasterConfig, MasterServeConfig
from edgeshard.control.master.profiling_admin import MasterProfilingAdmin
from edgeshard.control.master.profiling_controller import ProfilingController
from edgeshard.control.master.service import MasterService
from edgeshard.control.master.snapshot import format_snapshot
from edgeshard.control.worker.agent import (
    LocalWorkerInspector,
    inspect_local_worker,
    to_plain_mapping,
)
from edgeshard.control.worker.config import WorkerConfig
from edgeshard.control.worker.master_client import WorkerAgent
from edgeshard.control.worker.model_inventory import scan_model_inventory
from edgeshard.control.worker.profiling_sessions import (
    ProfilingSessionManager,
    RegistrationTokens,
)
from edgeshard.model.errors import EdgeShardError
from edgeshard.profiling.codec import encode_payload
from edgeshard.profiling.domain.experiment import ProfilingRequest
from edgeshard.profiling.domain.model import ModelReference
from edgeshard.profiling.domain.network import NetworkPair, NetworkPathClass, ProbeKind
from edgeshard.profiling.domain.session import ProfilingSessionKind, ProfilingSessionRequest
from edgeshard.profiling.store.sqlite import SqliteProfileStore
from edgeshard.protocol.control.grpc_server import start_control_server
from edgeshard.protocol.profiling import mapper as profiling_mapper
from edgeshard.protocol.profiling.grpc_client import ProfilingAdminClient
from edgeshard.protocol.profiling.grpc_server import (
    start_admin_server,
    start_profiling_server,
)
from edgeshard.runtime.model_store import ModelStore

logger = logging.getLogger("edgeshard.cli")

app = typer.Typer(
    help="EdgeShard: shard runtime and cluster control-plane CLI.",
    add_completion=False,
    no_args_is_help=False,
)
runtime_app = typer.Typer(help="Shard runtime commands.")
worker_app = typer.Typer(help="Worker Agent commands.")
master_app = typer.Typer(help="Master commands.")
profile_app = typer.Typer(help="Profiling plane commands (Phase 2 spec §49).")
profile_model_app = typer.Typer(help="Model profiling commands.")
profile_operator_app = typer.Typer(help="Operator microprofiling commands.")
profile_network_app = typer.Typer(help="Network profiling commands.")
app.add_typer(runtime_app, name="runtime")
app.add_typer(worker_app, name="worker")
app.add_typer(master_app, name="master")
app.add_typer(profile_app, name="profile")
profile_app.add_typer(profile_model_app, name="model")
profile_app.add_typer(profile_operator_app, name="operator")
profile_app.add_typer(profile_network_app, name="network")

RuntimeConfigPath = Annotated[
    Path,
    typer.Option(exists=True, readable=True, help="Runtime YAML config (spec 19.1)."),
]
WorkerConfigPath = Annotated[
    Path,
    typer.Option(exists=True, readable=True, help="Worker YAML config (spec 26)."),
]
MasterConfigPath = Annotated[
    Path,
    typer.Option(exists=True, readable=True, help="Master YAML config (spec 45)."),
]


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    config: Annotated[
        Path | None,
        typer.Option(exists=True, readable=True, help="Runtime YAML config (legacy form)."),
    ] = None,
) -> None:
    """EdgeShard CLI; `edgeshard --config X.yaml` aliases `runtime serve`."""
    if ctx.invoked_subcommand is not None:
        return
    if config is None:
        typer.echo(ctx.get_help())
        raise typer.Exit
    runtime_serve(config)


@app.command("serve")
def serve(config: RuntimeConfigPath) -> None:
    """Compatibility alias for `edgeshard runtime serve` (spec 42)."""
    runtime_serve(config)


@runtime_app.command("serve")
def runtime_serve(config: RuntimeConfigPath) -> None:
    """Run one EdgeShard shard runtime stage until terminated."""
    # Lazy import: the runtime config/server pull in torch, which is the
    # optional inference extra — every other CLI command imports without it.
    try:
        from edgeshard.runtime.config import ShardRuntimeConfig
    except ImportError as exc:
        typer.secho(
            "runtime serve requires the inference extra "
            f"(pip install 'edgeshard[inference]'): {exc}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from exc

    try:
        runtime_config = ShardRuntimeConfig.from_yaml(config)
    except (OSError, ValueError) as exc:
        typer.secho(f"invalid runtime config: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    try:
        asyncio.run(_serve(runtime_config))
    except EdgeShardError as exc:
        typer.secho(f"runtime failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


@worker_app.command("inspect")
def worker_inspect(
    config: Annotated[
        Path | None,
        typer.Option(exists=True, readable=True, help="Worker YAML config (spec 26)."),
    ] = None,
    output_format: Annotated[
        Literal["json", "yaml"],
        typer.Option("--format", help="Output format (spec 43)."),
    ] = "json",
) -> None:
    """Print a machine-readable snapshot of local capability/state.

    Works without any Master: identity is loaded/created locally, capability
    and telemetry are probed on this host, and runtime/model inventories are
    scanned from Docker and the ModelStore.
    """
    try:
        worker_config = WorkerConfig.from_yaml(config) if config is not None else WorkerConfig()
    except (OSError, ValueError) as exc:
        typer.secho(f"invalid worker config: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    try:
        inspection = asyncio.run(inspect_local_worker(worker_config))
    except EdgeShardError as exc:
        typer.secho(f"worker inspect failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    payload = {
        "identity": to_plain_mapping(inspection.identity),
        "capability": to_plain_mapping(inspection.capability),
        "state": to_plain_mapping(inspection.state),
    }
    if output_format == "yaml":
        typer.echo(yaml.safe_dump(payload, sort_keys=False))
    else:
        typer.echo(json.dumps(payload, indent=2))


@worker_app.command("serve")
def worker_serve(config: WorkerConfigPath) -> None:
    """Run the Worker Agent until terminated (spec §44).

    Identity, local discovery, telemetry, inventories, registration,
    heartbeats and reconnect/backoff. With ``profiling.enabled`` the Worker
    also hosts WorkerProfilingService and advertises the bound endpoint at
    registration (Phase 2 spec §41); disabled keeps exact Phase 1 behavior.
    """
    try:
        worker_config = WorkerConfig.from_yaml(config)
    except (OSError, ValueError) as exc:
        typer.secho(f"invalid worker config: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    _configure_logging()
    try:
        # Construction itself fails loudly when worker.master is missing
        # (spec §26: serve requires an endpoint).
        asyncio.run(_worker_serve(worker_config))
    except EdgeShardError as exc:
        typer.secho(f"worker failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    except KeyboardInterrupt:
        typer.echo("worker stopped")


_WILDCARD_BIND_HOSTS = frozenset({"", "0.0.0.0", "::"})


def _profiling_advertise_host(bind_host: str) -> str:
    """A dialable host for the advertised endpoint (mirrors §19 inventory).

    A wildcard bind accepts on every interface but is not itself dialable;
    the Master must receive an address it can actually connect to, so
    wildcards advertise loopback — the single-host default topology.
    """
    return "127.0.0.1" if bind_host in _WILDCARD_BIND_HOSTS else bind_host


async def _worker_serve(config: WorkerConfig) -> None:
    """Serve loop; hosts the profiling plane alongside the Agent when enabled.

    Ordering is the §41 contract: the profiling server binds *before*
    registration so the Agent advertises the resolved ``host:bound_port``
    (port 0 → OS-chosen), and the shared inspector serves both loops (§37:
    one long-lived process, fresh state per RPC). Shutdown closes the Agent
    first — its ``run`` owns the inspector lifecycle — then releases every
    profiling session and lease (§39: no reservation outlives the runner).
    """
    if not config.profiling.enabled:
        plain_agent = WorkerAgent(config)  # exact Phase 1 behavior
        await plain_agent.run()
        return

    # Imported here, not at module scope: the runner needs torch (it
    # benchmarks), torch is the optional inference extra, and the rest of
    # the CLI — every Master-side and `profile` command — must import
    # without it.
    from edgeshard.control.worker.profiling_runner import WorkerProfilingRunner

    inspector = LocalWorkerInspector(config)
    agent: WorkerAgent | None = None

    def current_tokens() -> RegistrationTokens | None:
        """The Agent's live registration, or ``None`` between registrations.

        The §41 gate: while the Agent is unregistered (startup, reconnect
        backoff) every profiling RPC is refused STALE_SESSION rather than
        executing under a dead registration.
        """
        if agent is None:
            return None
        worker_id = agent.worker_id
        registration_session_id = agent.registration_session_id
        if worker_id is None or registration_session_id is None:
            return None
        return RegistrationTokens(
            worker_id=worker_id,
            instance_id=agent.instance_id,
            registration_session_id=registration_session_id,
        )

    runner = WorkerProfilingRunner(
        sessions=ProfilingSessionManager(token_source=current_tokens),
        inspector=inspector,
        model_store_root=config.model_store.root,
    )
    server, port = await start_profiling_server(
        runner, host=config.profiling.host, port=config.profiling.port
    )
    endpoint = f"{_profiling_advertise_host(config.profiling.host)}:{port}"
    logger.info("profiling service listening, advertising %s", endpoint)
    try:
        # Constructed inside the try: a config the Agent rejects (e.g. a
        # missing master endpoint) must still stop the bound server and
        # release the runner (§39), never leave them dangling on a closed
        # loop.
        agent = WorkerAgent(config, inspector=inspector, profiling_endpoint=endpoint)
        await agent.run()
    finally:
        await runner.shutdown()
        await server.stop(grace=None)


@master_app.command("serve")
def master_serve(
    config: MasterConfigPath,
    snapshot_interval_s: Annotated[
        float,
        typer.Option(
            min=0.0,
            help="Log an immutable ClusterSnapshot every N seconds "
            "(debug representation, spec §45); 0 disables it.",
        ),
    ] = 0.0,
) -> None:
    """Run the Master control plane until terminated (spec §45).

    WorkerRegistryService over gRPC, session/sequence handling, liveness
    tracking and immutable cluster snapshots — no REST, dashboard, database
    or scheduler yet (spec §45, §57). With ``--snapshot-interval-s`` set, a
    debug rendering of each snapshot is logged periodically.
    """
    try:
        serve_config = MasterServeConfig.from_yaml(config)
        # Timing invariants (§32) are enforced by the semantic core; failing
        # here keeps the error path identical to a malformed YAML shape.
        master_config = serve_config.to_master_config()
    except (OSError, ValueError) as exc:
        typer.secho(f"invalid master config: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    _configure_logging()
    try:
        asyncio.run(_master_serve(serve_config, master_config, snapshot_interval_s))
    except EdgeShardError as exc:
        typer.secho(f"master failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    except KeyboardInterrupt:
        typer.echo("master stopped")


def _configure_logging() -> None:
    """Root handler for the long-running serve commands (spec §46).

    ``basicConfig`` is a no-op when an embedding process (e.g. pytest)
    already configured logging, so component loggers keep flowing to
    whatever handler the host installed.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )


async def _master_serve(
    serve_config: MasterServeConfig,
    master_config: MasterConfig,
    snapshot_interval_s: float = 0.0,
) -> None:
    service = MasterService(master_config)
    server, port = await start_control_server(
        service, host=serve_config.master.host, port=serve_config.master.port
    )
    print(f"READY master endpoint={serve_config.master.host}:{port}", flush=True)
    await service.start()
    snapshot_task: asyncio.Task[None] | None = None
    if snapshot_interval_s > 0:
        snapshot_task = asyncio.create_task(
            _log_snapshots(service, snapshot_interval_s), name="master-snapshot-log"
        )
    # Phase 2 §49: the profiling admin plane is additive — disabled (the
    # default) keeps exact Phase 1 behavior, down to the files on disk.
    admin: MasterProfilingAdmin | None = None
    admin_server: grpc.aio.Server | None = None
    store: SqliteProfileStore | None = None
    if serve_config.profiling.enabled:
        store = SqliteProfileStore(serve_config.profiling.store_path)
        controller = ProfilingController(service=service, store=store)
        admin = MasterProfilingAdmin(controller=controller)
        admin_server, admin_port = await start_admin_server(
            admin,
            host=serve_config.profiling.admin_host,
            port=serve_config.profiling.admin_port,
        )
        advertise = _profiling_advertise_host(serve_config.profiling.admin_host)
        print(f"READY profiling-admin endpoint={advertise}:{admin_port}", flush=True)
    try:
        await server.wait_for_termination()
    finally:
        if snapshot_task is not None:
            snapshot_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await snapshot_task
        if admin is not None:
            # Stops background runs without cancelling the experiments: they
            # stay resumable in the store on the next `master serve` (§50).
            await admin.shutdown()
        if admin_server is not None:
            await admin_server.stop(grace=None)
        if store is not None:
            store.close()
        await service.stop()
        await server.stop(grace=None)


async def _log_snapshots(service: MasterService, interval_s: float) -> None:
    """Periodically log the debug rendering of a fresh snapshot (spec §45).

    ``build_snapshot`` is synchronous and cheap; the sleep is the only await,
    so this never contends with the registry/heartbeat handlers.
    """
    snapshot_logger = logging.getLogger("master.snapshot")
    while True:
        await asyncio.sleep(interval_s)
        snapshot_logger.info("cluster snapshot:\n%s", format_snapshot(service.build_snapshot()))


# ---------------------------------------------------------------------------
# profile commands (Phase 2 spec §49)
#
# The CLI never drives domain logic: every command either builds a
# ProfilingRequest *intent* the Master expands through its strategy, or reads
# back status/snapshots. Presentation (json/yaml) happens here and only here.
# ---------------------------------------------------------------------------

MasterEndpointOption = Annotated[
    str,
    typer.Option(
        "--master",
        help="Master profiling-admin endpoint host:port (the READY "
        "profiling-admin line of `edgeshard master serve`).",
    ),
]
ExperimentIdOption = Annotated[
    str, typer.Option("--experiment", help="Canonical experiment id (§7).")
]
OutputFormatOption = Annotated[
    Literal["json", "yaml"], typer.Option("--format", help="Output format.")
]
WorkerListOption = Annotated[
    list[str] | None,
    typer.Option(
        "--worker",
        help="Restrict to this worker id; repeatable. Default: every worker "
        "hosting the profiling service.",
    ),
]
DeviceListOption = Annotated[
    list[str],
    typer.Option(
        "--device",
        help="Device id the benchmarks lease (§39); repeatable, at least one.",
    ),
]
ModelIdOption = Annotated[
    str, typer.Option("--model", help="Model id resolvable in the workers' model stores (§13).")
]
DtypeOption = Annotated[
    str, typer.Option("--dtype", help="Declared measurement dtype (§17), e.g. fp32/bf16.")
]
RevisionOption = Annotated[
    str | None,
    typer.Option("--revision", help="Specific model revision; omit for the local snapshot."),
]
RequestedByOption = Annotated[
    str | None,
    typer.Option("--requested-by", help="Operator label recorded with the experiment (§8.1)."),
]
IncludeMeasuredOption = Annotated[
    bool,
    typer.Option(
        "--include-measured",
        help="Re-measure operator signatures that already have stored "
        "measurements; default reuses them (§28).",
    ),
]


def _fail(message: str) -> NoReturn:
    typer.secho(message, fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


def _render_payload(payload: object, output_format: Literal["json", "yaml"]) -> None:
    if output_format == "yaml":
        typer.echo(yaml.safe_dump(payload, sort_keys=False))
    else:
        typer.echo(json.dumps(payload, indent=2))


def _admin_call[T](
    endpoint: str, call: Callable[[ProfilingAdminClient], Awaitable[T]]
) -> T:
    """One admin RPC over a fresh channel; transport problems exit loudly."""

    async def invoke() -> T:
        async with ProfilingAdminClient(endpoint) as client:
            return await call(client)

    try:
        return asyncio.run(invoke())
    except grpc.aio.AioRpcError as exc:
        code = exc.code().name if exc.code() is not None else "unknown"
        _fail(f"master profiling-admin call failed ({code}): {exc.details()}")
    except ValueError as exc:
        _fail(f"invalid profiling request: {exc}")


def _build_intent(build: Callable[[], ProfilingRequest]) -> ProfilingRequest:
    try:
        return build()
    except ValueError as exc:
        _fail(f"invalid profiling request: {exc}")


def _start_intent(master: str, intent: ProfilingRequest) -> None:
    response = _admin_call(
        master,
        lambda client: client.start_experiment(
            profiling_mapper.StartExperimentRequest(request=intent)
        ),
    )
    if not response.accepted:
        _fail(f"experiment rejected: {response.detail}")
    typer.echo(f"experiment {response.experiment_id} started")


def _parse_path_class(value: str) -> NetworkPathClass:
    try:
        return NetworkPathClass(value)
    except ValueError:
        valid = ", ".join(member.value for member in NetworkPathClass)
        _fail(f"unknown path class {value!r} (valid: {valid})")


def _parse_pair(value: str) -> NetworkPair:
    parts = value.split(":")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        _fail(f"--pair must be SRC_WORKER:DST_WORKER, got {value!r}")
    return NetworkPair(source_worker_id=parts[0], destination_worker_id=parts[1])


@profile_model_app.command("inspect")
def profile_model_inspect(
    model: ModelIdOption,
    revision: RevisionOption = None,
    config: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            readable=True,
            help="Worker YAML config (spec 26) supplying the model store root.",
        ),
    ] = None,
    dtype: DtypeOption = "fp32",
    device: Annotated[
        str, typer.Option("--device", help="Torch device the characterization runs on.")
    ] = "cpu",
    output_format: OutputFormatOption = "json",
) -> None:
    """Characterize a local model checkpoint without any cluster (§47 step 1).

    Runs the same Worker-side loader `worker serve` uses and prints the
    ModelSessionFacts the Master-side strategy plans from. Requires the
    inference extra (torch + transformers); everything else in the CLI is
    torch-free.
    """
    try:
        worker_config = WorkerConfig.from_yaml(config) if config is not None else WorkerConfig()
    except (OSError, ValueError) as exc:
        _fail(f"invalid worker config: {exc}")

    try:
        import torch

        from edgeshard.control.worker.profiling_model_loader import (
            TorchModelSessionLoader,
            resolve_model_source,
        )
    except ImportError as exc:
        _fail(
            "model inspection requires the inference extra "
            f"(pip install 'edgeshard[inference]'): {exc}"
        )

    store = ModelStore(model_root=worker_config.model_store.root)
    request = ProfilingSessionRequest(
        kind=ProfilingSessionKind.MODEL,
        device_ids=(device,),
        model=ModelReference(model_id=model, revision=revision),
        dtype=dtype,
    )
    try:
        source = resolve_model_source(model, revision, scan_model_inventory(store), store)
        session = TorchModelSessionLoader(device=torch.device(device)).load(request, source)
    except (EdgeShardError, ValueError, RuntimeError, OSError) as exc:
        _fail(f"model inspection failed: {exc}")
    _render_payload(encode_payload(session.facts), output_format)


@profile_model_app.command("run")
def profile_model_run(
    master: MasterEndpointOption,
    model: ModelIdOption,
    dtype: DtypeOption,
    device: DeviceListOption,
    worker: WorkerListOption = None,
    revision: RevisionOption = None,
    include_measured: IncludeMeasuredOption = False,
    requested_by: RequestedByOption = None,
) -> None:
    """Plan and dispatch a full model profiling experiment (§47 steps 1-6).

    The Master inspects the model on each target worker, then dispatches
    operator/module/layer cases per the active strategy. Prints the canonical
    experiment id; follow it with `profile status`.
    """
    intent = _build_intent(
        lambda: ProfilingRequest(
            kind=ProfilingSessionKind.MODEL,
            model=ModelReference(model_id=model, revision=revision),
            dtype=dtype,
            worker_ids=tuple(worker or ()),
            device_ids=tuple(device),
            missing_only=not include_measured,
            requested_by=requested_by,
        )
    )
    _start_intent(master, intent)


@profile_operator_app.command("run")
def profile_operator_run(
    master: MasterEndpointOption,
    model: ModelIdOption,
    dtype: DtypeOption,
    device: DeviceListOption,
    worker: WorkerListOption = None,
    revision: RevisionOption = None,
    include_measured: IncludeMeasuredOption = False,
    requested_by: RequestedByOption = None,
) -> None:
    """Dispatch only the operator microbenchmarks a model's facts yield (§25).

    Same inspection as `profile model run`, but the experiment keeps
    OPERATOR-granularity cases — incremental by default (§28): signatures
    with stored measurements are reused, `--include-measured` re-runs them.
    """
    intent = _build_intent(
        lambda: ProfilingRequest(
            kind=ProfilingSessionKind.OPERATOR,
            model=ModelReference(model_id=model, revision=revision),
            dtype=dtype,
            worker_ids=tuple(worker or ()),
            device_ids=tuple(device),
            missing_only=not include_measured,
            requested_by=requested_by,
        )
    )
    _start_intent(master, intent)


@profile_network_app.command("rtt")
def profile_network_rtt(
    master: MasterEndpointOption,
    worker: WorkerListOption = None,
    requested_by: RequestedByOption = None,
) -> None:
    """Dispatch the dense cheap RTT matrix over the selected workers (§33)."""
    intent = _build_intent(
        lambda: ProfilingRequest(
            kind=ProfilingSessionKind.NETWORK,
            worker_ids=tuple(worker or ()),
            network_probe=ProbeKind.RTT,
            requested_by=requested_by,
        )
    )
    _start_intent(master, intent)


@profile_network_app.command("bandwidth")
def profile_network_bandwidth(
    master: MasterEndpointOption,
    worker: WorkerListOption = None,
    path_class: Annotated[
        list[str] | None,
        typer.Option(
            "--path-class",
            help="Only probe pairs of this class (§34); repeatable. "
            "Default: the strategy's sparse per-class selection.",
        ),
    ] = None,
    pair: Annotated[
        list[str] | None,
        typer.Option(
            "--pair",
            help="Explicit directed pair SRC:DST added regardless of the "
            "sparse selection (§34); repeatable.",
        ),
    ] = None,
    requested_by: RequestedByOption = None,
) -> None:
    """Dispatch iperf3-style bandwidth baselines in both flow directions (§34)."""
    classes = tuple(_parse_path_class(value) for value in path_class or ())
    pairs = tuple(_parse_pair(value) for value in pair or ())
    intent = _build_intent(
        lambda: ProfilingRequest(
            kind=ProfilingSessionKind.NETWORK,
            worker_ids=tuple(worker or ()),
            network_probe=ProbeKind.BANDWIDTH,
            bandwidth_path_classes=classes,
            extra_bandwidth_pairs=pairs,
            requested_by=requested_by,
        )
    )
    _start_intent(master, intent)


@profile_app.command("status")
def profile_status(
    master: MasterEndpointOption,
    experiment: ExperimentIdOption,
    output_format: OutputFormatOption = "json",
) -> None:
    """Print the Master-side lifecycle status of one experiment (§49).

    Unknown experiments exit non-zero: absence is reported, never guessed
    (§52.2). Typed case failures are shown when this Master process ran the
    experiment; the store keeps states only (§43).
    """
    response = _admin_call(
        master,
        lambda client: client.get_experiment(
            profiling_mapper.GetExperimentRequest(experiment_id=experiment)
        ),
    )
    if not response.found or response.status is None:
        _fail(f"unknown experiment {experiment!r}")
    _render_payload(encode_payload(response.status), output_format)


@profile_app.command("cancel")
def profile_cancel(
    master: MasterEndpointOption,
    experiment: ExperimentIdOption,
) -> None:
    """Cancel every non-terminal case of one experiment (§40).

    Terminal cases keep their history (§44); completed measurements are
    never rolled back.
    """
    response = _admin_call(
        master,
        lambda client: client.cancel_experiment(
            profiling_mapper.CancelExperimentRequest(experiment_id=experiment)
        ),
    )
    if not response.accepted:
        _fail(f"cancellation rejected: {response.detail}")
    typer.echo(response.detail)


@profile_app.command("snapshot")
def profile_snapshot(
    master: MasterEndpointOption,
    output_format: OutputFormatOption = "yaml",
) -> None:
    """Build and print the ProfileSnapshot Phase 3 consumes (§46).

    v1 snapshots the whole store; rendering (yaml/json) is CLI presentation
    and never a Master concern (§49).
    """
    response = _admin_call(
        master,
        lambda client: client.build_profile_snapshot(
            profiling_mapper.BuildProfileSnapshotRequest()
        ),
    )
    if not response.accepted or response.snapshot is None:
        _fail(f"snapshot build rejected: {response.detail}")
    _render_payload(encode_payload(response.snapshot), output_format)


async def _serve(config: ShardRuntimeConfig) -> None:
    from edgeshard.runtime.shard_server import ShardRuntimeServer  # torch (§49)

    server = await ShardRuntimeServer.create(config)
    print(f"READY runtime={config.runtime.runtime_id} endpoint={server.endpoint}", flush=True)
    try:
        await server.wait_for_termination()
    finally:
        await server.stop()


if __name__ == "__main__":
    app()
