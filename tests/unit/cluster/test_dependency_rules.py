"""Dependency-direction guard for the cluster domain (Phase 1 spec §8).

``edgeshard.cluster`` is a pure domain package: it may import only the
Python standard library, and only ``protocol.control`` / ``control.*`` may
depend on it. The inference and model layers must never reference cluster.
These structural rules are enforced by scanning imports, not by importing
the modules, so a violation fails even in environments where the offending
dependency happens to be installed.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import edgeshard
from edgeshard import cluster

SRC = Path(edgeshard.__file__).resolve().parent
CLUSTER_ROOT = Path(cluster.__file__).resolve().parent


def _imported_modules(path: Path) -> list[str]:
    """Full dotted module names imported by ``path``; relative imports skipped."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0 or node.module is None:
                continue  # relative imports inside the package are fine
            names.append(node.module)
    return names


def _cluster_module_files() -> list[Path]:
    return sorted(
        path
        for path in CLUSTER_ROOT.rglob("*.py")
        if "__pycache__" not in path.parts
    )


def test_cluster_imports_only_stdlib() -> None:
    """cluster may import the stdlib and itself, nothing else (spec §8)."""
    offenders = [
        f"{path.relative_to(CLUSTER_ROOT).as_posix()}: {name}"
        for path in _cluster_module_files()
        for name in _imported_modules(path)
        if name.split(".")[0] not in sys.stdlib_module_names
        and not name.startswith("edgeshard.cluster")
    ]
    assert offenders == [], f"cluster imports non-stdlib modules: {offenders}"


def test_cluster_module_files_exist() -> None:
    """Guard against the scan silently passing on an empty/renamed package."""
    stems = {path.stem for path in _cluster_module_files()}
    assert stems == {"__init__", "capability", "identity", "inventory", "snapshot", "state"}


def test_inference_and_model_do_not_import_cluster() -> None:
    offenders = [
        path.relative_to(SRC).as_posix()
        for package in ("inference", "model")
        for path in sorted((SRC / package).rglob("*.py"))
        if "__pycache__" not in path.parts and _imports_cluster(path)
    ]
    assert offenders == []


def _imports_cluster(path: Path) -> bool:
    return any(
        name == "edgeshard.cluster" or name.startswith("edgeshard.cluster.")
        for name in _imported_modules(path)
    )
