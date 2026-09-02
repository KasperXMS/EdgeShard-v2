# Shard runtime images (Phase 0)

Three pinned Dockerfiles, one per target tier (spec 4.10). They build
**EdgeShard shard runtime** images. The independent vLLM runtime uses the
official pinned image untouched instead (specs 4.7, 20.2) — no EdgeShard
code is baked into it.

| File | Purpose | Torch source | Test tier |
| --- | --- | --- | --- |
| `Dockerfile.cpu` | deterministic correctness image (spec 21.2) | CPU wheel index, exact `uv.lock` set (`uv sync --frozen`) | Tier 1/2 |
| `Dockerfile.cuda` | x86 NVIDIA shard runtime | pinned official PyTorch CUDA runtime base (same torch version as the lock); other deps from a hashed `uv export` | Tier 2 |
| `Dockerfile.jetson` | Jetson/L4T shard runtime | pinned for the JetPack/L4T base (Python 3.10); validated on target hardware | Tier 3 |

Conventions every shard image follows:

- Models are mounted read-only at `/models` and never baked into the
  image; runtime containers never download models (spec 21.1).
- The runtime config mounts read-only at `/runtime/config/runtime.yaml`
  and uses container-side paths (`model.path` under `/models/...`,
  `server.listen_host: 0.0.0.0`).
- `ENTRYPOINT ["edgeshard"]` + `CMD ["--config", "/runtime/config/runtime.yaml"]`;
  the Mock Master passes its own generated config path (spec 22).
- Managed containers carry the spec 21.5 labels
  `io.edgeshard.managed`, `io.edgeshard.execution_id`,
  `io.edgeshard.runtime_id`, and `io.edgeshard.backend`, so orphaned
  containers of an execution can be found and cleaned.

Container tests (`tests/container/`) build the CPU image locally and skip
explicitly where no Docker daemon is reachable; the vLLM E2E additionally
requires `EDGESHARD_VLLM_IMAGE` on a GPU host.
