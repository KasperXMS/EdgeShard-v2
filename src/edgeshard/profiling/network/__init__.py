"""Network profiling (Phase 2 P2F, spec §30-§36).

Characterizes the cluster network from facts and measures it empirically:

- :mod:`edgeshard.profiling.network.classifier` reuses the Phase 1
  discovery facts (worker identity, interfaces, addresses, MTU) to build
  endpoint profiles, derives interface kinds and path classes from
  documented policies (§30-§31), enumerates directed pairs (§32), and
  applies the sparse per-class bandwidth selection (§34);
- :mod:`edgeshard.profiling.network.ping` runs bounded system-ping RTT
  probes with typed failures (§33);
- :mod:`edgeshard.profiling.network.iperf` runs JSON-only iperf3
  single-flow baselines with guaranteed server cleanup (§34-§35);
- :mod:`edgeshard.profiling.network.profiler` turns network cases into
  empirical measurement records — dense RTT matrices and sparse
  bandwidth representatives — with workload-derived payload sizes (§36).

Every unknown stays ``None`` (§52.2), every failure stays typed (§42),
and every throughput number is recorded as an idle single-flow baseline
(§35) — never as a guaranteed concurrent bandwidth (§52.7).
"""
