"""Command-line entry point.

Phase 1 command groups (spec §42)::

    edgeshard runtime serve --config runtime.yaml
    edgeshard worker inspect [--config worker.yaml] [--format json|yaml]

The Phase 0 forms stay as compatibility aliases during Phase 1::

    edgeshard serve --config runtime.yaml        # == runtime serve
    edgeshard --config runtime.yaml              # historical bare form

``worker inspect`` works without any Master (spec §43); it is the primary
local validation tool for Phase 1 hardware discovery.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Annotated, Literal

import typer
import yaml

from edgeshard.control.worker.agent import inspect_local_worker, to_plain_mapping
from edgeshard.control.worker.config import WorkerConfig
from edgeshard.model.errors import EdgeShardError
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
app.add_typer(runtime_app, name="runtime")
app.add_typer(worker_app, name="worker")

RuntimeConfigPath = Annotated[
    Path,
    typer.Option(exists=True, readable=True, help="Runtime YAML config (spec 19.1)."),
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


async def _serve(config: ShardRuntimeConfig) -> None:
    server = await ShardRuntimeServer.create(config)
    print(f"READY runtime={config.runtime.runtime_id} endpoint={server.endpoint}", flush=True)
    try:
        await server.wait_for_termination()
    finally:
        await server.stop()


if __name__ == "__main__":
    app()
