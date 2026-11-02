# WORD_GAMES model in regular Triton

The optimized Empero Qwen3.8-35B-A3B Q4_K_M model is an ordinary model in the
existing Triton repository. The regular `triton` container serves it on HTTP
port **8000**, with gRPC on 8001 and metrics on 8002. Its model name is
`word-games-qwen38-native`.

## Model lifecycle

Use Triton's normal explicit-control API:

```bash
curl -X POST http://127.0.0.1:8000/v2/repository/models/word-games-qwen38-native/load -d '{}'
curl -f http://127.0.0.1:8000/v2/models/word-games-qwen38-native/ready
curl -X POST http://127.0.0.1:8000/v2/repository/models/word-games-qwen38-native/unload -d '{}'
```

Repository-root convenience commands call those same APIs:

```bash
pixi run word-games-load
pixi run word-games-unload
pixi run word-games-status
pixi run word-games-verify
```

`word-games-up` and `word-games-down` are aliases for model load/unload. They
**do not start, stop or recreate the shared Triton server**. Infrastructure owns
that server through `triton.service`. The server has explicit model control and
does not automatically load WORD_GAMES on startup. Unload waits for the native
child to exit, releasing its GPU allocations while Triton remains available.

The operator has reserved all three GPUs open-ended until explicit release;
60-second preemption is not authorized. This is an operator decision, not an
implementation of the planned reservation system. Do not load competing GPU
models alongside this model. Coordinate eventual release with PBX 201-1002;
infrastructure handles restoring its LLM/OCR workloads.

## Installed package

Versioned source: `regular/models/word-games-qwen38-native/`.
Installed model: `/data/models/_triton/word-games-qwen38-native/`.

```text
word-games-qwen38-native/
  config.pbtxt
  installation.json
  1/
    model.py
    runtime.json
    runtime-assets/
      native/    # pinned patched llama.cpp executables and libraries
      sysroot/   # isolated loader/libraries, used only by the native child
      cuda/      # pinned CUDA runtime and cuBLAS libraries
```

The regular server already mounts `/data/models/_triton` as `/models` and
`/data/models` as `/model-weights`, both read-only. No research workspace mounts
or separate WORD_GAMES container are required. The 22 GB GGUF is referenced at
its existing `/model-weights/Empero-Qwen3.8-35B-A3B-Distill-GGUF/` location;
it is not copied into the model repository.

`pixi run word-games-install` stages and verifies the model package atomically,
leaving other repository entries untouched. It copies about 1.1 GiB of pinned
runtime libraries from the existing validated build and host CUDA installation.
`regular/release.lock.json` pins those files, symlink targets, model sources,
weights and original image identity. Installation is idempotent for an unchanged
release and rejects a mismatched existing entry instead of overwriting it.

## Preserved runtime

- Original Q4_K_M weights; llama.cpp b11960 with the sampling-output acquisition patch.
- All experts on GPU; whole-layer placement across RTX 3060 / 2060 SUPER / 2060.
- Flash Attention, Q8_0 K/V, 8,192-token context, one native slot.
- Logical batch 2,048; microbatch 1,024; one CPU compute thread.
- Unused context checkpoints disabled; speculative decoding and prefix caching disabled.
- One CPU Triton backend instance owns the native process; no duplicate model per GPU.

The native child uses UUID-based GPU ordering, leaving the shared server's
mapping for other models unchanged. `/opt/infra/docker-compose.yml` exposes
host devices `[1,0,2]`, appending the SUPER to its prior `[1,0]` list. The exact
change is retained in `regular/triton-gpu-visibility.patch` for infrastructure's
canonical deployment source. Visibility alone does not load any models.

The inference contract is preserved: `text_input` plus optional
`sampling_parameters`, JSON output, temperature 0, seed 73, and up to 1,024
output tokens. Logs and CUDA cache live inside the regular container under
`/tmp/word-games-qwen38-native/`. Native HTTP stays private at port 18080.

The previous approximately 67 full-request tokens/s measurement applies to the
validated workload, not a new benchmark of this migration. This package supplies
the model runtime; V3/V4 WORD_GAMES orchestration remains separate.

Migration validation is recorded in `regular/validation.json`: 29 focused tests
passed, and six inference checks after load/reload exactly matched the three
baseline prompts. Unload kept the shared server healthy and released the native
model's GPU memory. The observed first load took 230.5 seconds, reload 80.3
seconds, and unload 2.1 seconds; allow several minutes for model loading.

The earlier standalone Compose, root `models/`, root lock and validation files
are retained for reproduction. Their lifecycle controller is now
`experiments.word_games_triton.standalone_service`; ordinary commands use the
regular service above. Do not run the historical standalone deployment alongside
the regular model.
