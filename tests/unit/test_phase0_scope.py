"""Scope guard: deferred work must remain absent.

Originated as the third clause of the 0K Phase 0 freeze gate, evolved for
Phase 1 (spec §57, §63) and now for Phase 2: the package tree contains
exactly the packages built so far, ``edgeshard.control`` holds only what the
current milestone allows, and future-phase components (scheduler, placement,
estimators, ...) do not exist anywhere. Any accidental introduction fails
in Tier 1.

Phase 2 status: the ``profiling`` package is admitted (Phase 2 spec §4 —
extensible profiling and workload characterization). Everything Phase 2
defers stays forbidden (§53): no scheduler, no placement, no cost/estimate
model — those are Phase 3/4 components and must not appear before their
time, exactly as ``edgeshard.profiling`` was forbidden during Phase 1.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import edgeshard

SRC = Path(edgeshard.__file__).resolve().parent

EXPECTED_PACKAGES = {"cluster", "control", "inference", "model", "profiling", "protocol", "runtime"}

# Later-phase components (Phase 0 spec 29 / Phase 1 spec §57 / Phase 2 spec
# §53) that must not exist yet. The production Worker/Master live under
# edgeshard.control.*, never as top-level packages.
FORBIDDEN_SUBMODULES = (
    "edgeshard.scheduler",
    "edgeshard.worker",
    "edgeshard.master",  # the production Master is edgeshard.control.master (P1F)
    "edgeshard.placement",
)
FORBIDDEN_FILE_STEMS = {
    "scheduler",
    "worker",
    "placement",
    "cost_model",
    "estimator",
}


def _module_files() -> list[Path]:
    return sorted(
        path
        for path in SRC.rglob("*.py")
        if "__pycache__" not in path.parts and "pb" not in path.parts
    )


def test_only_expected_packages_exist() -> None:
    packages = {
        path.name
        for path in SRC.iterdir()
        if path.is_dir() and path.name != "__pycache__"
    }
    assert packages == EXPECTED_PACKAGES


def test_control_contains_only_milestone_components() -> None:
    # P1B admits the production Worker Agent; P1F admits the production
    # Master state components (spec §33, §56); P1H adds SnapshotBuilder
    # inside the same master package. Scheduler-like modules stay deferred
    # (§57), so the control package gains no new sub-packages here.
    entries = {
        path.name for path in (SRC / "control").iterdir() if path.name != "__pycache__"
    }
    assert entries == {"__init__.py", "mock", "worker", "master"}


def test_no_future_phase_modules_are_importable() -> None:
    for name in FORBIDDEN_SUBMODULES:
        assert importlib.util.find_spec(name) is None, f"{name} must not exist in Phase 0"


def test_no_future_phase_source_files() -> None:
    offenders = [
        path.relative_to(SRC).as_posix()
        for path in _module_files()
        if path.stem in FORBIDDEN_FILE_STEMS
    ]
    assert offenders == []
