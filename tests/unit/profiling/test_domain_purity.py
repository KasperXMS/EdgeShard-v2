"""Dependency-direction guard for the profiling domain (P2A DoD).

``edgeshard.profiling.domain`` is a pure domain package: stdlib imports
only — no torch, gRPC, Docker, NVML, tegrastats, SQLite, and no other
``edgeshard`` package (the frozen cluster contract restricts its importers
to ``protocol.control``/``control.*``, so Phase 1 concepts are mirrored by
value-compatible declarations instead of imported). Enforced by scanning
imports, not by importing modules, so a violation fails even where the
offending dependency happens to be installed.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import edgeshard
from edgeshard.profiling import domain

SRC = Path(edgeshard.__file__).resolve().parent
DOMAIN_ROOT = Path(domain.__file__).resolve().parent


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


def _domain_module_files() -> list[Path]:
    return sorted(
        path
        for path in DOMAIN_ROOT.rglob("*.py")
        if "__pycache__" not in path.parts
    )


def test_profiling_domain_imports_only_stdlib() -> None:
    offenders = [
        f"{path.relative_to(DOMAIN_ROOT).as_posix()}: {name}"
        for path in _domain_module_files()
        for name in _imported_modules(path)
        if name.split(".")[0] not in sys.stdlib_module_names
        and not name.startswith("edgeshard.profiling.domain")
    ]
    assert offenders == [], f"profiling domain imports non-stdlib modules: {offenders}"


def test_profiling_domain_module_files_exist() -> None:
    """Guard against the scan silently passing on an empty/renamed package."""
    stems = {path.stem for path in _domain_module_files()}
    assert stems == {
        "__init__",
        "environment",
        "experiment",
        "hashing",
        "measurement",
        "model",
        "network",
        "session",
        "signature",
        "snapshot",
    }


def test_frozen_phase1_packages_do_not_import_profiling() -> None:
    """Phase 0/1 packages are frozen: profiling is additive, never inverted."""
    offenders = [
        path.relative_to(SRC).as_posix()
        for package in ("cluster", "inference", "model", "runtime", "protocol")
        for path in sorted((SRC / package).rglob("*.py"))
        if "__pycache__" not in path.parts and _imports_profiling(path)
    ]
    assert offenders == []


def _imports_profiling(path: Path) -> bool:
    return any(
        name == "edgeshard.profiling" or name.startswith("edgeshard.profiling.")
        for name in _imported_modules(path)
    )
