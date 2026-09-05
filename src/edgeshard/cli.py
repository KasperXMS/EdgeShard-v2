"""Command-line entry point.

Phase 1 command groups (spec §42)::

    edgeshard runtime serve --config runtime.yaml
    edgeshard worker inspect [--config worker.yaml] [--format json|yaml]
    edgeshard worker serve --config worker.yaml
    edgeshard master serve --config master.yaml

The Phase 0 forms stay as compatibility aliases during Phase 1::

    edgeshard serve --config runtime.yaml        # == runtime serve
    edgeshard --config runtime.yaml              # historical bare form

``worker inspect`` works without any Master (spec §43); it is the primary
local validation tool for Phase 1 hardware discovery. ``worker serve``
(spec §44) registers with the Master and heartbeats until terminated;
``master serve`` (spec §45) runs the WorkerRegistryService, liveness
tracking and debug snapshots — no REST, dashboard or scheduler yet.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Annotated, Literal

import typer
import yaml

from edgeshard.control.master.config import MasterConfig, MasterServeConfig
from edgeshard.control.master.service import MasterService
from edgeshard.control.worker.agent import inspect_local_worker, to_plain_mapping
from edgeshard.control.worker.config import WorkerConfig
from edgeshard.control.worker.master_client import WorkerAgent
from edgeshard.model.errors import EdgeShardError
from edgeshard.protocol.control.grpc_server import start_control_server
from edgeshard.runtime.config import ShardRuntimeConfig
from edgeshard.runtime.shard_server import ShardRuntimeServer

logger = logging.getLogger("edgeshard.cli")

app = typer.Typer(
    help="EdgeShard: shard runtime and cluster control-plane CLI.",
    add_completion=False,
    no_args_is_help=False,
)
runtime_app = typer.Typer(help="Shard runtime commands.")
worker_app = typer.Typer(help="Worker Agent commands.")
master_app = typer.Typer(help="Master commands.")
app.add_typer(runtime_app, name="runtime")
app.add_typer(worker_app, name="worker")
app.add_typer(master_app, name="master")

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
    heartbeats and reconnect/backoff — but no scheduling, profiling or
    production runtimes yet (spec §44, §57).
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
        agent = WorkerAgent(worker_config)
        asyncio.run(agent.run())
    except EdgeShardError as exc:
        typer.secho(f"worker failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    except KeyboardInterrupt:
        typer.echo("worker stopped")


@master_app.command("serve")
def master_serve(config: MasterConfigPath) -> None:
    """Run the Master control plane until terminated (spec §45).

    WorkerRegistryService over gRPC, session/sequence handling, liveness
    tracking and debug snapshots — no REST, dashboard, database or
    scheduler yet (spec §45, §57).
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
        asyncio.run(_master_serve(serve_config, master_config))
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
    serve_config: MasterServeConfig, master_config: MasterConfig
) -> None:
    service = MasterService(master_config)
    server, port = await start_control_server(
        service, host=serve_config.master.host, port=serve_config.master.port
    )
    print(f"READY master endpoint={serve_config.master.host}:{port}", flush=True)
    await service.start()
    try:
        await server.wait_for_termination()
    finally:
        await service.stop()
        await server.stop(grace=None)


async def _serve(config: ShardRuntimeConfig) -> None:
    server = await ShardRuntimeServer.create(config)
    print(f"READY runtime={config.runtime.runtime_id} endpoint={server.endpoint}", flush=True)
    try:
        await server.wait_for_termination()
    finally:
        await server.stop()


if __name__ == "__main__":
    app()
