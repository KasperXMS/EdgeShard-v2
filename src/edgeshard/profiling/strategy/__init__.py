"""Replaceable profiling strategy layer (spec §47-§48).

A strategy is a *pure planner*: it turns facts the cluster already reports
(``ModelSessionFacts`` from a prepared Worker session, ``WorkerNetworkFacts``
from Phase 1 discovery) plus the reuse answer from the profile store into
``ProfilingCase`` tuples. It never benchmarks, never touches the store or
gRPC itself, and never produces composed performance estimates (§47 —
composition belongs to Phase 3); dispatch and persistence belong to the
Master ``ProfilingController`` (§40).

The layer stays replaceable through the ``ProfilingStrategy`` protocol
(:mod:`edgeshard.profiling.strategy.base`); ``DefaultProfilingStrategy``
(:mod:`edgeshard.profiling.strategy.default`) hard-codes the v1 policy
composition (§48 explicitly allows this — the domain, storage, and runner
layers never depend on it).
"""
