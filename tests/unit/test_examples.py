"""Bundled examples stay parseable: configs and manifests from examples/.

Every shipped example must parse through the same strict models the
runtime and the Mock Master use, so documentation cannot drift from the
schemas (spec 28: examples are part of the Definition of Done).
"""

from __future__ import annotations

from pathlib import Path

from edgeshard.control.mock.manifest import DeploymentManifest
from edgeshard.runtime.config import ShardRuntimeConfig

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"


def test_all_example_runtime_configs_parse() -> None:
    paths = sorted((EXAMPLES_DIR / "configs").glob("*.yaml"))
    assert paths, "examples/configs/ should not be empty"
    for path in paths:
        ShardRuntimeConfig.from_yaml(path)


def test_all_example_manifests_parse() -> None:
    paths = sorted((EXAMPLES_DIR / "manifests").glob("*.yaml"))
    assert paths, "examples/manifests/ should not be empty"
    for path in paths:
        DeploymentManifest.from_yaml(path)
