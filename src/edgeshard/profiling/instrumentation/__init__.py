"""Measurement instrumentation (Phase 2 spec §11, §14, §15).

Three independent instrumentation concerns, all producing domain metric
types and all honoring §52.2 — a metric that cannot be measured on this
host is ``None``, never a guessed or zero value:

- :mod:`~edgeshard.profiling.instrumentation.timing` — CUDA event timing
  (mandatory for GPU latency, §11.1) and wall-clock timing for CPU,
  network, and control-path measurements.
- :mod:`~edgeshard.profiling.instrumentation.memory` — PyTorch allocator
  view (§14.1) and the physical Phase 1 ``MemoryPool`` view (§14.2).
- :mod:`~edgeshard.profiling.instrumentation.telemetry` — contextual
  device observations plus configurable contamination checks (§15).

This package intentionally does NOT import ``edgeshard.control`` or
``edgeshard.cluster``: the dependency direction is
``profiling.domain <- profiling impl <- control/protocol`` (spec §56.5).
Bridges from Phase 1 telemetry probes to the protocols defined here live
on the control side (P2G runner).
"""
