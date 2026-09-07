"""Master-side torch-freedom guard (P2G.7).

torch is an *optional* dependency (``[project.optional-dependencies]
inference``) so the Master control plane stays lightweight, and the Master
never benchmarks (§40): the strategy layer, its protocol base, and the
Master ``ProfilingController`` must therefore import — and plan — with no
torch installed at all.

Enforced in a subprocess where ``import torch`` can only fail
(``sys.modules["torch"] = None`` halts any torch import with an
ImportError), so the guard cannot pass silently on a development machine
that happens to have torch installed. This complements the AST scan in
``test_domain_purity.py``: that one pins the *domain* to stdlib-only
imports statically; this one pins the *Master-side stack* to runtime
torch-freedom dynamically, catching transitive imports too (e.g. the lazy
torch table in ``profiling.dtypes``).
"""

from __future__ import annotations

import subprocess
import sys

MASTER_SIDE_MODULES = (
    "edgeshard.profiling.strategy.base",
    "edgeshard.profiling.strategy.default",
    "edgeshard.control.master.profiling_controller",
)
"""Every module the Master loads to plan and orchestrate profiling (§40)."""

_SCRIPT_HEADER = """\
import sys

sys.modules["torch"] = None  # any `import torch` now raises ImportError

"""

_SCRIPT_FOOTER = """
assert sys.modules["torch"] is None, "torch was imported"
print("torch-free")
"""


def _run_torch_blocked(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", _SCRIPT_HEADER + script + _SCRIPT_FOOTER],
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_master_side_modules_import_without_torch() -> None:
    for module in MASTER_SIDE_MODULES:
        result = _run_torch_blocked(f"import {module}\n")
        assert result.returncode == 0, (
            f"{module} is not importable without torch:\n{result.stderr}"
        )
        assert "torch-free" in result.stdout


def test_default_strategy_plans_without_torch() -> None:
    """Planning itself — not just importing — never touches torch (§46)."""
    script = """\
from edgeshard.profiling.network.classifier import InterfaceFacts, WorkerNetworkFacts
from edgeshard.profiling.network.classifier import classify_interface
from edgeshard.profiling.strategy.default import DefaultProfilingStrategy

def facts(worker_id, address):
    kind, overlay = classify_interface("eth0")
    return WorkerNetworkFacts(
        worker_id=worker_id,
        hostname=f"host-{worker_id}",
        interfaces=(InterfaceFacts(f"nic-{worker_id}", "eth0", kind, overlay, (address,), 1500),),
    )

plan = DefaultProfilingStrategy().plan_network_cases(
    facts=[facts("w-a", "192.168.1.10"), facts("w-b", "192.168.1.20")]
)
assert len(plan.cases) == 4, plan.cases
print("planned", len(plan.cases), "cases")
"""
    result = _run_torch_blocked(script)
    assert result.returncode == 0, (
        f"strategy planning requires torch:\n{result.stderr}"
    )
    assert "planned 4 cases" in result.stdout
    assert "torch-free" in result.stdout
