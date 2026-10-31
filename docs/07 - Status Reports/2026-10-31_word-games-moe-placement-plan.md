# WORD_GAMES MoE placement and CPU tuning

Date: 2026-10-31. Hypothesis `word-games-moe-placement-v1`.
Prospective exploratory systems study; no training or deployment. The user
requested a test of using the RTX 2060 SUPER and accelerating CPU MoE work.
The available pinned **Empero**, not PrismML, 35B Q4_K_M is the hardware test
case. Publisher identity remains material to any model-quality claim.

## Controls and question

Can CPU tuning, resident layers on a second GPU, or exact expert caching reduce
completion-request latency on the same 36 frozen WORD_GAMES prompts?
Checkpoint SHA256 `196103269085bc54c9b8f49ed21e9f53e1b56b465e8b796c6d8e31e06f63cfa5`;
requests SHA256 `f9830c9480f8a4694630a06102cfb33e9e16517faec22e4f9f59888137fd53cb`.
Native b11960 and pinned container, Q4_K_M, Flash Attention, Q8 KV, 8192 context,
512 batch/ubatch, temperature 0, seed 73, no thinking/speculation/prompt reuse,
load mode none, original prompts and output caps remain fixed. No quantization
or expert dropping. Host CPU: 12-core / 24-thread Xeon E5-2697 v2, AVX/F16C,
no AVX2. GPU0 RTX 3060; GPU2 RTX 2060 SUPER with existing service preserved.
GPU1 RTX 2060 Qwen3-8B must remain untouched.

## Conditions, ordered gates, and selection

1. Unit tests for placement bounds, command generation, placement verification,
   data pairing and quality/speed classification.
2. Separate three-request lifecycle/screening run per configuration, IDs from
   the existing PILOT constant. These are not quality or final speed evidence.
   Control: 26 CPU expert blocks (0-25), 12 threads. CPU alternatives: 6, 8,
   24 threads; and 12 threads pinned to one logical CPU per physical core
   (0-11, strict affinity). Prompt threads track generation threads. No other
   activity is deliberately run on the test GPUs; resident services remain.
3. Second GPU pilot: first eight whole blocks on SUPER, blocks 8-25 expert
   weights on CPU, remaining blocks on 3060. Explicit device order and layer
   split must be verified from runtime logs; automatic fitting is disabled.
   If memory fails, a new six-block pilot is allowed, with original failure
   retained. At least 384 MiB free VRAM must remain on each test GPU.
4. Exact expert-cache pilot: baseline placement plus 2048 MiB GPU expert cache,
   with log evidence that a nonzero cache was allocated.
5. Select CPU configuration with lowest pilot summed completion time; choose
   control if improvement is less than 5%. Run two complete 36-request replays
   of control, selected CPU condition if different, resident-GPU condition,
   and expert-cache condition if they passed execution/memory gates. No
   conditional quality rescue. Use selected CPU setting for a combined GPU
   or cache follow-up only if separate gains justify it; preregister exact
   configuration in an amendment before that follow-up's outcomes.
6. Repeat control once after interventions to report drift. Selected arms
   must beat both beginning and ending controls to be called promising.

Primary timing: summed HTTP completion time and visible output tokens/s,
including prefill, excluding model hash/load/warmup/template application and
artifact I/O. Secondary: native decode rate, prefill seconds, output bytes,
GPU telemetry, CPU backend/affinity, host memory, cache logging. All measured
passes start after a real JSON warm-up; first SM75 JIT cost is startup, retained
separately. Continuous GPU/process observations are retained. No external
concurrent workloads are started. Because resident services are retained,
timing is an exploratory deployment measurement, not an isolated benchmark.

For a promising speed-and-parity result require >=5% lower completion time on
both repetitions versus both controls, all 36 valid untruncated JSON objects,
and exact parsed-JSON equality for all requests versus beginning control.
Raw text and token matches are also reported. Parsed equality is a finite-suite
regression check, not a general accuracy estimate. If outputs differ, timing is
still reported but no lossless quality claim is made; manually inspect changed
receiver outputs as a labeled diagnostic. The existing model's failed live
metaphor selection and unsupported approval summaries remain unresolved;
this study cannot establish a deployed bridge accuracy percentage.

## Integrity and stop rules

Fresh roots `data/experiments/word_games/20261031_moe_placement_*`; strict JSON,
full configuration, binary/checkpoint/input/source digests and frozen source.
No edits to producing dependencies while running. Code and runtime stay outside
data. Stop affected cell on wrong placement, invalid warm-up, missing cache,
OOM, process crash, service health failure or timeout. Retain failed artifacts;
no retry to obtain better quality. Record any infrastructure-only correction
in an amendment. Clean up only the owned container. Aggregate all scheduled
completed/failed cells, write analysis, conservative typed registry with exact
and semantic readback, focused tests and selective commit.

### Pre-inference guard correction

Pilot v1 was interrupted during initial checkpoint hashing, before source
freezing, model startup or any request. Review of a historical native log showed
layer device names are followed by commas. The new placement parser and its
fixture were corrected to handle that punctuation. Preserve v1's operator note;
launch v2 with unchanged runtime conditions and gates.

### Pilot-follow-up amendment

After pilot v2: CPU physical-core pinning used 11.66 seconds versus 15.58 control;
eight resident SUPER blocks used 11.82 seconds with changed output text. CPU
6/8/24 did not improve. The cache loaded 2046.77 MiB successfully, but the
controller rejected its capitalized `MoE cache size` log before any request.
Correct only that allocation parser, require a positive numeric allocation,
and retry cache under a new pilot root. Preserve the original failure.

Before further results, add two combination pilots: `gpu8pin` (same eight
blocks, 12 physically pinned CPU workers), and `gpu8pin_cache2048` (also 2048 MiB
expert cache). These use unchanged controls and precision. Advance the faster
valid combination to two full replays if it is at least 5% faster than the
fastest individual pilot. The original individual conditions remain in the
full comparison; selection is exploratory and all pilots remain reported.
