"""0K scope freeze: no Phase 1+ functionality has leaked into Phase 0.

The third clause of the 0K gate, automated: the package tree contains
exactly the Phase 0 packages, ``edgeshard.control`` holds only the Mock
Master, and future-phase components (scheduler, worker, profiling,
production master, ...) do not exist anywhere. Any accidental
introduction fails in Tier 1.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import edgeshard

SRC = Path(edgeshard.__file__).resolve().parent

EXPECTED_PACKAGES = {"control", "inference", "model", "protocol", "runtime"}

# Later-phase components (spec 29) that must not exist in Phase 0.
FORBIDDEN_SUBMODULES = (
    "edgeshard.scheduler",
    "edgeshard.worker",
    "edgeshard.profiling",
    "edgeshard.master",  # production master; only control.mock exists
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


def test_only_phase0_packages_exist() -> None:
    packages = {
        path.name
        for path in SRC.iterdir()
        if path.is_dir() and path.name != "__pycache__"
    }
    assert packages == EXPECTED_PACKAGES


def test_control_contains_only_the_mock_master() -> None:
    entries = {
        path.name for path in (SRC / "control").iterdir() if path.name != "__pycache__"
    }
    assert entries == {"__init__.py", "mock"}


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
