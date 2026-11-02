# Qwen3.8-35B-A3B distillation on WORD_GAMES

Prospective exploratory protocol, 2026-10-31.
Hypothesis ID: `word-games-qwen38-moe-v1`.

The user requested Qwen3.8-35B-A3B. Public release discovery found
`empero-ai/Qwen3.8-35B-A3B-Distill-GGUF`, a community distillation into the
Qwen3.6-35B-A3B MoE backbone. The public Qwen organization search did not list
an official model of this name. Report the publisher and exact identity rather
than implying an official Qwen3.8 release. Use the publisher's recommended
Q4_K_M file, pinned by revision, size and SHA256 in `prepare_qwen38_moe.py`.

Question: how does this released model perform on the existing short structured
WORD_GAMES workload, and what latency/memory does it require on this machine?
No training, draft, lookup or new task prompts. Preserve the authorized-only
receiver view and model-selected metaphor. This is not a replication of the
publisher's long-thinking evaluation.

Use the same native llama.cpp b11960, temperature 0, seed 73, generic JSON-object
constraint, selector cap 256, other caps 1024, context 8192, Q8 KV and batch/ubatch
512. Disable thinking through the embedded template and require exact user
content modulo the template's surrounding-whitespace trim plus an empty closed
thinking prefix. If the template cannot provide this, stop and document the
incompatibility before altering the condition. The publisher recommends
reasoning with stochastic sampling and long outputs; report this departure.

The Q4 checkpoint is 21.7 GB and cannot fit in the RTX 3060's 12 GB VRAM. Use
explicit CPU expert offload for the first 26 of 40 layers, remaining layers and
non-expert computation on the 3060, 12 CPU threads for the host's 12 physical
cores. Native `--n-cpu-moe` fixes this placement; `--fit off` prevents silent
autotuning. This hybrid placement is deliberate and must appear beside all
speed results; it is not a matched all-GPU speed comparison. Preserve the
resident RTX 2060 Qwen3-8B and RTX 2060 SUPER services. Do not load a second
model onto either card or silently lower weight precision to fit.

Pre-inference configuration update: the user explicitly specified Q4_K_M,
Flash Attention, Q8 KV and 26 CPU MoE layers. This supersedes the initial
24-layer estimate before any model run. Flash Attention is explicitly on,
K and V are both Q8_0, and all 26 routed-expert placements will be checked from
native debug logs. Use `--load-mode none` to let native CPU tensor overrides
use their preferred buffers rather than mmap; the pinned loader recommends
this for CPU expert offload. Twelve CPU threads and verbosity 5 are fixed;
level 5 is required to record individual tensor placements.

Sequence and gates:

1. Verify release/checkpoint metadata and native architecture support. Run
   focused CPU contract tests. Start an isolated worker, verify actual expert
   placement and model identity, pass a JSON warm-up, then the same three frozen
   pilot requests. Check memory, valid complete JSON, timing and cleanup.
2. If the pilot passes, run all 36 frozen requests once. Report visible output
   tokens/full completion-request second, native decode-only rate, UTF-8 bytes/s,
   total request latency, and observed device/host memory. Reuse the completed
   Qwen2.5 and Bonsai 36-request controls; do not rerun completed evidence.
3. Run the same twelve live cases, reusing the Qwen2.5 baseline's synthetic
   context IDs only after authorization metadata equality checks. If the
   metaphor selector fails, stop before cases and report no case-accuracy
   measurement. Otherwise report sender, boundary, receiver and per-case joint
   structured success out of twelve, with canary/identifier disclosure checks.
4. Manually compare each receiver summary with its authorized packet, recording
   unsupported claims separately from structural success. This is assistant
   review, not an independent judge or exact-summary matching. Compare observed
   structural success with Qwen2.5 5/12, Bonsai 10/12 and historical Phi 10/12.
   No population-accuracy or 95% claim follows from this small synthetic suite.

Stop on bad identity, invalid/truncated pilot JSON, transport failure, OOM,
unsupported architecture or changed fingerprints. Infrastructure-only recovery
requires a dated amendment and a fresh root; preserve failures. Do not retry a
quality failure with changed prompts or sampling. If the pilot shows long
latency, retain it honestly; this is a feasibility test, not an optimization
sweep. Full replay requests retain the existing 240-second per-call timeout.

Use Pixi from the repository root; code and environments stay outside `data/`.
Freeze/hash sources, runtime, checkpoint, fixtures and plan before execution;
use fresh `data/experiments/word_games/20261031_qwen38_*` roots. Produce an
append-only aggregate, measured report, conservative typed registry record with
exact/semantic readback, and a selective source commit preserving user changes.

## Infrastructure amendment before pilot outcomes

The v1 loader logs confirm routed expert overrides for blocks 0-25 to
`CUDA_Host`, native page-locked CPU memory. The placement validator mistakenly
accepted only buffer names beginning with `CPU`. The owned worker was stopped
during startup, before warm-up or any pilot request, to correct this known
guard error. Preserve v1's raw startup/interruption and operator note. Accept
`CUDA_Host` explicitly while continuing to reject CUDA device buffers, missing
projections and missing/extra expert blocks. No runtime, prompt, weights,
precision, placement or quality gate changes; retry lifecycle under v2.
