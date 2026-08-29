"""Command-line entry point (0F): run one shard runtime process.

Usage::

    python -m edgeshard --config runtime.yaml

The process loads its shard, binds the gRPC endpoint, prints a ``READY``
line, and serves until terminated. Errors exit non-zero with an explicit
message instead of degrading silently.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer

from edgeshard.model.errors import EdgeShardError
from edgeshard.runtime.config import ShardRuntimeConfig
from edgeshard.runtime.shard_server import ShardRuntimeServer

app = typer.Typer(
    help="EdgeShard shard runtime: run one pipeline stage.",
    add_completion=False,
    no_args_is_help=True,
)

ConfigPath = Annotated[
    Path,
    typer.Option(exists=True, readable=True, help="Runtime YAML config (spec 19.1)."),
]


@app.command()
def serve(config: ConfigPath) -> None:
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


async def _serve(config: ShardRuntimeConfig) -> None:
    server = await ShardRuntimeServer.create(config)
    print(f"READY runtime={config.runtime.runtime_id} endpoint={server.endpoint}", flush=True)
    try:
        await server.wait_for_termination()
    finally:
        await server.stop()


if __name__ == "__main__":
    app()
