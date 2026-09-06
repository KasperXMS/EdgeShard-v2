"""Pure profiling domain model (Phase 2 spec §4-10, §30-32, §42, §46).

Immutable empirical facts and reusable workload identities: signatures,
model characterizations, device performance classes, environment
fingerprints, measurements, experiments/cases, typed failures, network
entities, and the ``ProfileSnapshot``.

This package imports only the Python standard library (P2A DoD): no torch,
gRPC, Docker, NVML, tegrastats, SQLite, and no other ``edgeshard`` package —
not even ``edgeshard.cluster``, whose frozen contract restricts its
importers to ``protocol.control``/``control.*``. Phase 1 concepts are
mirrored by value-compatible declarations (e.g. ``MemoryModel``) and bridged
only in the implementation layers above. Purity is enforced by
``tests/unit/profiling/test_domain_purity.py``.
"""
