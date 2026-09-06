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
"""
