"""Production Worker Agent (Phase 1 spec §7).

Local discovery milestone (P1B): persistent identity, configuration, host
capability discovery, psutil telemetry, and observational runtime/model
inventory - plus ``worker inspect`` as the Master-less validation tool
(spec §43). Networking (registration, heartbeats) lands in milestone P1G
on top of exactly these pieces.

Dependency direction (spec §8): this package may import ``cluster``
domain types and ``runtime`` seams (ModelStore, drivers' labels), never
the reverse.
"""
