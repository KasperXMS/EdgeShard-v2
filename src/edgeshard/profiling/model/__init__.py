"""Model characterization and direct model profiling (spec §16-§17, §21-§24).

Static characterization (P2C) — architecture family, sizes, stage graph,
layer and module enumeration — without any performance benchmarking
(§17). Structural identity is consumed from the Phase 0
:class:`ModelLayout` ("later phases consume layouts; they never
re-inspect Hugging Face configs"); the HF config is read only for facts
the layout does not carry (vocab size, explicit head_dim, quantization
method).

Direct profiling (P2D) benchmarks the real checkpoint through adapter
enumeration and input building: :mod:`layer_profiler` (§21, plus the
§22 layer-position sanity check), :mod:`module_profiler` (§23), and the
shared :mod:`execution` workload/record mapping. Prefill only in v1 —
decode fails typed, never approximated (§24).
"""
