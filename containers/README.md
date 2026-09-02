# EdgeShard runtime containers (spec 21)

One container = one pipeline stage of one runtime backend. Phase 0 ships the
EdgeShard HF shard backend in three images; the vLLM backend (0J) uses the
official `vllm/vllm-openai` image unmodified.

| Dockerfile               | Purpose (spec)                          | Base                                            |
| ------------------------ | --------------------------------------- | ----------------------------------------------- |
| `hf/Dockerfile.cpu`      | CI / deterministic correctness (21.2)   | `python:3.12-slim` + CPU torch from `uv.lock`   |
| `hf/Dockerfile.cuda`     | x86 NVIDIA shard runtime (21.3)         | `pytorch/pytorch:2.13.0-cuda13.2-cudnn9-runtime` |
| `hf/Dockerfile.jetson`   | Jetson/L4T shard runtime (21.4)         | `dustynv/l4t-pytorch:r36.4.0`                   |

## Build

```bash
docker build -f containers/hf/Dockerfile.cpu   -t edgeshard/hf-shard:cpu    .
docker build -f containers/hf/Dockerfile.cuda  -t edgeshard/hf-shard:cuda   .
docker build -f containers/hf/Dockerfile.jetson -t edgeshard/hf-shard:jetson .  # on Jetson host
```

The CPU image installs the exact locked dependency set (`uv sync --frozen`),
so a pinned image + pinned config reproduces one runtime environment
(spec 4.10). The CUDA image shares every lock version except torch, which the
base image provides as the CUDA build (same torch version as the lock). The
Jetson Dockerfile is a pinned scaffold; validation happens on the target
JetPack/L4T environment (Tier 3, spec 26).

## Run

Models are mounted, never baked (spec 21.1). The runtime config is mounted
read-only at `/runtime/config/runtime.yaml` (the image `CMD`):

```bash
docker run --rm \
  -v "$EDGESHARD_MODEL_CACHE:/models:ro" \
  -v "$PWD/examples/configs/container-stage-cpu.yaml:/runtime/config/runtime.yaml:ro" \
  -p 9100:9100 \
  edgeshard/hf-shard:cpu
```

CUDA adds `--gpus all` (NVIDIA Container Toolkit); Jetson adds
`--runtime nvidia`. Driver-launched containers get the same wiring through
the Docker SDK: `EdgeShardShardRuntimeDriver` attaches an NVIDIA
`DeviceRequest` (`count=-1`, `[["gpu"]]` — the `--gpus all` equivalent)
whenever the mounted config sets `device.type: cuda`, and
`VLLMRuntimeDriver` always attaches it (plus `ipc_mode=host`), so GPU
launches behave exactly like the manually validated `docker run` forms.

Container-side config requirements (the driver rejects configs that violate
them before launch):

- `model.path` points under `/models/...`;
- `server.listen_host: 0.0.0.0` (loopback-only configs are unreachable from
  outside the container);
- the host publishes `server.listen_port`.

## Managed-container labels (spec 21.5)

Containers launched by `EdgeShardShardRuntimeDriver` carry:

```text
io.edgeshard.managed=true
io.edgeshard.execution_id=<id>
io.edgeshard.runtime_id=<id>
io.edgeshard.backend=<backend>
```

Orphan cleanup:

```bash
docker rm -f $(docker ps -aq --filter label=io.edgeshard.managed=true)
```
