"""Pure cluster domain model (Phase 1 spec §7-20, 38).

Immutable facts about Workers, Devices, memory pools, dynamic state,
inventories, and cluster snapshots. This package imports only the Python
standard library (spec §8): no gRPC, protobuf, Docker, torch, psutil,
pynvml, and no edgeshard package may depend on it except
``protocol.control`` and ``control.*``.
"""
