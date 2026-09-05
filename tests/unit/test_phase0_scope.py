"""Scope guard: deferred work must remain absent.

Originated as the third clause of the 0K Phase 0 freeze gate, now evolved
for Phase 1 (spec §57, §63): the package tree contains exactly the packages
built so far, ``edgeshard.control`` holds only what the current milestone
allows, and future-phase components (scheduler, profiling, placement,
production master, ...) do not exist anywhere. Any accidental introduction
fails in Tier 1.

Phase 1 status: the pure ``cluster`` domain package is admitted (spec §7);
``control`` holds the Mock Master, the production Worker Agent from
milestone P1B onward, and the production Master state components from
milestone P1F onward. Scheduler/profiling/placement remain deferred (§57).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import edgeshard

SRC = Path(edgeshard.__file__).resolve().parent

EXPECTED_PACKAGES = {"cluster", "control", "inference", "model", "protocol", "runtime"}

# Later-phase components (Phase 0 spec 29 / Phase 1 spec §57) that must not
# exist yet. The production Worker/Master live under edgeshard.control.*,
# never as top-level packages.
FORBIDDEN_SUBMODULES = (
    "edgeshard.scheduler",
    "edgeshard.worker",
    "edgeshard.profiling",
    "edgeshard.master",  # the production Master is edgeshard.control.master (P1F)
    "edgeshard.placement",
)
FORBIDDEN_FILE_STEMS = {
    "scheduler",
    "worker",
    "profiling",
    "profile",
    "placement",
    "cost_model",
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
    # Master state components (spec §33, §56). SnapshotBuilder joins the
    # master package in P1H; scheduler-like modules stay deferred (§57).
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
