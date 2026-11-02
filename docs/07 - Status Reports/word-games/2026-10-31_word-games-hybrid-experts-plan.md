# Fine-grained CPU/GPU expert partition: exploratory implementation

Hypothesis `word-games-hybrid-experts-v1`, 2026-10-31. Written before the
implementation or outcomes. User requested the actual within-layer split.

Retain the same Empero Qwen3.8-35B-A3B Q4_K_M checkpoint, Q8 KV, Flash
Attention, 12 CPU threads, 8192 context, and frozen 36 WORD_GAMES requests
as `word-games-moe-placement-v1`. Attention stays on RTX 3060. Copy a
bounded subset of each of the first 26 layers' routed experts to the
RTX 2060 SUPER; compute the remaining selected experts on CPU. All top-8
experts contribute, using unchanged quantized weights and router weights.
The resident Qwen3-8B on RTX 2060 and OCR allocation on SUPER stay running.

Adapt the GPU compact-pool / CPU correction mechanism demonstrated in
https://github.com/bdrazn/blobpager at d7c9297ff054a3b54827c8f5eb2f77d01d8126a1
to llama.cpp b11960 and Qwen35MoE. This is implementation inspiration,
not a replication of that repository's reported speed. Initially enable
the split for single-token decode only; normal prefill remains unchanged.
Read the graph's actual final router weights, avoid reconstructing them.
Reuse CPU work graphs; attempt overlap of CPU misses with GPU hits and
explicitly identify whether overlap is implemented. Uniform expert IDs are
the first allocation; any later routing calibration must be declared.

Before task timing, validate GPU byte copies, finite CPU/GPU expert results,
and complete routed-expert accounting. Fail closed on missing correction.
Run feature-off and feature-on using the same custom native build. Include
the existing whole-bank `experts8` configuration as the practical speed
control (previously 31.02 end-to-end tok/s); historical measurements alone
do not establish a new-build gain. Pilot on the three existing pilot
requests, then run full 36 requests only if a split is competitive (within
10% of matched experts8 wall time) and returns valid complete JSON. A slow
pilot is a valid negative result; preserve it and diagnose overhead before
any explicitly recorded follow-up implementation.

Record source/binary/inputs/config hashes, CPU and GPU routed counts,
request wall time, native prefill/decode timings, output token counts,
memory telemetry and resident-service health. Stop the owned runtime if
GPU free memory drops below 384 MiB or the protected service fails twice.
Begin with 64 pins per CPU layer (roughly 3 GiB), bounded by observed
allocation; do not exceed 80 pins without a separate memory check.

Report speed separately from quality. Compare complete JSON, receiver
decisions/states and authorized semantics; changed wording is not by itself
quality loss. No claim of accuracy parity follows from this tiny pilot.
No training, precision reduction, expert pruning, or protected-service
unloading. Append-only dated artifacts under data/experiments/word_games;
source/build files under experiments and .pixi, execution from repo root
through Pixi. Preserve failed builds/runs and publish a conservative report
and NAL registry read-back for the stabilized outcome.
