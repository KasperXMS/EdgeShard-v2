"""Reusable benchmark lifecycle (Phase 2 spec §12-§13).

- :mod:`~edgeshard.profiling.benchmark.sampling` — the replaceable
  :class:`SamplingPolicy` protocol and the v1 default
  :class:`DurationSamplingPolicy` (≥3 warmups, 5-20 measured runs,
  ~1 s target accumulated duration; never a hard-coded "100 reps").
- :mod:`~edgeshard.profiling.benchmark.harness` — one
  :class:`BenchmarkHarness` implementation shared by layer, module, and
  operator profilers: prepare → validate environment → warmup → reset
  peaks → measurement loop → telemetry → summary → cleanup.

Advanced policies (convergence-based, confidence-interval stopping,
profiling budgets) are explicit future extension points, not v1 work
(§13).
"""
