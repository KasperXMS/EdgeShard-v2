"""Model characterization for profiling (Phase 2 spec §16-§17).

Static characterization — architecture family, sizes, stage graph, layer
and module enumeration — without any performance benchmarking (§17).
Structural identity is consumed from the Phase 0 :class:`ModelLayout`
("later phases consume layouts; they never re-inspect Hugging Face
configs"); the HF config is read only for facts the layout does not
carry (vocab size, explicit head_dim, quantization method).

Layer/module profilers (P2D) build on the enumeration provided here.
"""
