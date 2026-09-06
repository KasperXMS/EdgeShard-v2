"""Operator extraction, normalization, and deduplication (spec §18-§20).

Pipeline (low-cost coverage, §1.4)::

    module + representative inputs
      → OperatorExtractor  (raw graph: identity, shapes, dtypes, order)
      → OperatorNormalizer (stable EdgeShard operator vocabulary)
      → unique_operator_signatures (dedup across models and layers)

Only unique missing signatures reach microprofiling (P2E) — this is the
mechanism that keeps model/device profiling from becoming a Cartesian
product (§20). The torch.export path is primary; the torch.profiler path
is a structural-discovery fallback whose timing is never authoritative
(§18.2, §52.6). Unknown operations are preserved as ``CUSTOM`` and never
dropped silently (§19).

Microprofiling (P2E, §25-§29) benchmarks signatures *without* model
execution: ``workloads`` materializes synthetic GEMM/attention/norm
workloads on the production-relevant backend path, ``registry`` maps the
operator vocabulary onto them (unregistered kinds fail typed), and
``profiler`` measures through the shared §12 harness, plans incremental
runs so a new model contributes only its missing shapes (§28), and gates
performance-class reuse on a very small verification suite (§29).
"""
