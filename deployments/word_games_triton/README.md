# WORD_GAMES model: standalone Triton deployment

This configuration serves the existing Empero Qwen3.8-35B-A3B distillation,
Q4_K_M, using the validated `free_whole1024` runtime. It packages the inference
model independently of the WORD_GAMES bridge and its research clients.

From the repository root:

```bash
pixi run word-games-up       # Start Triton and load the model; also safe when already up
pixi run word-games-status   # Report container ownership, running state and model readiness
pixi run word-games-unload   # Release model GPU memory; keep Triton running
pixi run word-games-load     # Load the same model again
pixi run word-games-down     # Gracefully stop and remove this service's container
```

`up` verifies pinned assets before starting. These commands manage only
`word-games-triton`; they do not unload or restore unrelated models. The three
GPUs must have enough free memory. Qwen3-8B and OCR were already stopped for this
configuration. `down` leaves them in their existing state.

## Configuration and optimizations

| Setting | Pinned value |
| --- | --- |
| Triton model name | `word-games-qwen38-native` |
| HTTP / gRPC / metrics | Loopback ports 8020 / 8021 / 8022 |
| Weights | Original Q4_K_M GGUF; SHA-256 in `release.lock.json` |
| Native runtime | llama.cpp b11960 plus the sampling-output acquisition patch |
| GPUs | RTX 3060 12 GB, RTX 2060 SUPER 8 GB, RTX 2060 6 GB; UUIDs in Compose |
| Placement | Whole layers; 8 regular-2060 / 14 SUPER / 18 3060 trunk layers |
| Experts | All on GPU, zero CPU expert layers |
| Attention / KV | Flash Attention; Q8_0 K and V |
| Context / concurrent slots | 8,192 tokens / one slot |
| Logical batch / microbatch | 2,048 / 1,024 |
| CPU compute threads | One, for both decode and batch processing |
| Unused context checkpoints | Disabled |
| Speculation / prefix caching | Disabled |
| Triton adapter | One CPU backend instance owning the native three-GPU process |

The selected configuration previously measured about 67 full-request output
tokens/s on the frozen WORD_GAMES replay. This deployment does not establish new
throughput or task accuracy. Speculative decoding and prefix caching are excluded
because their earlier quality gates failed; the grammar-preview patch was slower.

`models/word-games-qwen38-native/` is a complete, pinned Triton model repository
entry: `config.pbtxt`, Python backend and native `runtime.json`. The snapshot is
intentional: changes to experimental adapter code do not silently change this
release. Native libraries and weights are mounted read-only.

The current Triton request contract remains JSON output, temperature 0, seed 73,
and at most 1,024 output tokens. Direct native chat is a separate client path.
This release does not add the V3/V4 bridge or expand its context budget.

## Verification and prerequisites

```bash
pixi run word-games-verify
docker compose -f deployments/word_games_triton/compose.yaml config --quiet
```

This is a reproducible configuration for the current machine. It relies on the
existing local artifacts, rather than embedding 22 GB of weights in Git:

- `/data/models/Empero-Qwen3.8-35B-A3B-Distill-GGUF/Qwen3.8-35B-A3B-Q4_K_M.gguf`
- `.pixi/word-games-sampling-v2/candidate/`: patched native binaries/libraries
- `.pixi/word-games-triton-native/sysroot/`: isolated native loader/libraries
- `/usr/local/cuda/lib64`: host CUDA libraries, and an NVIDIA container runtime
- The pinned local Triton Docker image named in Compose and the release lock

The lock records hashes for model repository files, native libraries and weights,
plus native-library symlink targets. The first verification hashes the complete
checkpoint; subsequent checks reuse that receipt only while its inode, size,
modification and change timestamps match. Changed assets fail verification.
Writable logs, lifecycle lock and verification receipt live under
`.pixi/word-games-triton-service/`; the CUDA cache retains its existing location.

For another host, transfer the pinned assets/image and explicitly adapt and
revalidate GPU placement. The pinned image is a local image ID, not a public
registry tag. `pull_policy: never` prevents substitution by a different image.

Compose can also be used directly after verification and creating the writable
directories, but the Pixi commands supply those preparation steps and ownership
checks. Model load/unload use Triton's standard explicit-control API:

```bash
curl -X POST http://127.0.0.1:8020/v2/repository/models/word-games-qwen38-native/unload -d '{}'
curl -X POST http://127.0.0.1:8020/v2/repository/models/word-games-qwen38-native/load -d '{}'
curl -f http://127.0.0.1:8020/v2/models/word-games-qwen38-native/ready
```

For the one-time conversion of the old research container, use
`pixi run python -m experiments.word_games_triton.service up --migrate-existing`.
This accepts only the matching old image, asset mounts and model repository,
drains the model, and preserves the old container for rollback until the new
service is ready. Ordinary subsequent starts need only `pixi run word-games-up`.
Use the new lifecycle commands after migration; the older experiment `deploy.py`
commands have different resident-service restoration behavior.
