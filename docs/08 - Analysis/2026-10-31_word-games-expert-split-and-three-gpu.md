# Actual expert splitting and a 50 tok/s GPU placement

Status: completed exploratory tests. `word-games-hybrid-experts-v1` is
`negative_speed_pilot`; `word-games-three-gpu-experts-v1` is
`full_speedup_quality_unestablished`. Both remain `confirmed: false`.

**Measured 50.21 output tokens/s end to end and 71.22 decode tokens/s** on
the fixed 36-request WORD_GAMES suite, with 38 of 40 serving expert banks
on GPUs. This is a single full replay, not a guaranteed sustained rate or
proof of unchanged task accuracy. The fastest run still fails the metaphor
selector's distinct-role check. Qwen3-8B was restored after testing; OCR
remained resident and its server readiness was checked afterward.

Aggregate:
`data/experiments/word_games/20261031_hybrid_three_gpu_analysis_v2/campaign_results.json`

SHA256: `08605be1ef1f65a9fde1fed3e99a4b56d6df418ee9b3219bd0857e443459a870`.

## Full fixed-workload measurements

| Placement | CPU expert banks | End-to-end tok/s | Decode tok/s | Total request seconds | Prefill seconds | Decode seconds | Output tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Eight banks on SUPER; original remaining placement | 18 | 31.71 | 47.93 | 156.99 | 52.93 | 103.86 | 4,978 |
| Eight on SUPER, ten on regular 2060 | 8 | 38.93 | 56.81 | 123.63 | 38.70 | 84.72 | 4,813 |
| Nine on SUPER, ten on regular 2060, nineteen on 3060 | 2 | **50.21** | **71.22** | **95.17** | **27.89** | **67.08** | **4,778** |

Every full pass completed 36 JSON objects without truncation. The fastest
placement increased token throughput **58.3%** and reduced completion time
**39.4%** versus the eight-bank baseline. It increased throughput **29.0%**
over the intermediate three-GPU placement. Output lengths differ; the
fastest run generated 4.0% fewer tokens than the first baseline. Output-byte
throughput also improved, from 114.84 to 183.84 bytes/s. Do not interpret
the elapsed-time ratio alone as an equal-output speedup.

The first two full passes used the same campaign and runtime. The packed
follow-up used that immediately preceding full comparison and fresh
three-request 18-bank references before/after its intervention. Reference
times were 7.057 and 6.983 seconds (1.05% faster at the end), within the
prospective 5% drift bound. There is one full pass per placement and no
confidence interval. These are repeated frozen requests, not independent
task draws or trained-model replicates.

End-to-end here means visible output tokens divided by HTTP completion
wall time, including prefill. Model loading, checkpoint verification,
template application, JSON warm-up and artifact I/O are excluded. No
prompt reuse, draft model, speculation or model training was used.

## Exact configuration and placement

The checkpoint is **Empero** `Qwen3.8-35B-A3B-Distill-GGUF`, not a PrismML
checkpoint. File `Qwen3.8-35B-A3B-Q4_K_M.gguf`, 21,713,462,944 bytes,
SHA256 `196103269085bc54c9b8f49ed21e9f53e1b56b465e8b796c6d8e31e06f63cfa5`.
Architecture is Qwen35MoE, 40 serving layers plus an unused MTP layer,
256 experts per layer, top-8 routing. Same quantized weights and all routed
experts are retained in every arm.

Native llama.cpp b11960, Flash Attention, Q8_0 K/V, context 8192,
batch/ubatch 512, 12 unpinned CPU threads, temperature 0, seed 73,
thinking off, generic JSON-object output constraint. Input corpus SHA256:
`f9830c9480f8a4694630a06102cfb33e9e16517faec22e4f9f59888137fd53cb`.

| Device | Fastest placement |
| --- | --- |
| RTX 3060 12 GB, logical CUDA0 | All attention/recurrent/shared work; expert banks 18-22 and 26-39 |
| RTX 2060 SUPER 8 GB, logical CUDA1 | Expert banks 0-7 and 23; existing OCR allocation retained |
| RTX 2060 6 GB, logical CUDA2 | Expert banks 8-17; Qwen3-8B temporarily unloaded |
| CPU | Expert banks 24-25 |

All 42 generic native layer assignments were verified on CUDA0; tensor
overrides move only the three routed-expert projections. Maximum observed
whole-device use in the fastest full pass was 10,947 / 7,190 / 4,864 MiB
on 3060 / SUPER / regular 2060, respectively. Minimum free memory was
961 / 597 / 875 MiB. The guard remained 384 MiB. These are one-second
samples, not a guarantee that shorter peaks were observed.

The CPU is an AVX-only Xeon E5-2697 v2. The native Ivy Bridge backend was
retained. No AVX2-only replacement kernel, precision reduction or expert
pruning was introduced. The fastest result comes from moving more complete
expert banks onto GPUs, not from the custom individual-expert runtime.

## The requested individual-expert split was implemented and tested

`native_hybrid/llama-word-games-hybrid.cpp` implements compact GPU pools
with 64 uniformly spaced experts plus a zero slot in each of the first
26 layers. During single-token decoding, GPU hits use their unchanged
quantized weights; the other selected experts execute on CPU. A persistent
CPU worker and cached miss graphs overlap with GPU hit computation. The
graph's final router weights are read directly, and CPU/GPU contributions
join before the next layer. Batched prompt processing retains the original
path. The module is an exploratory single-model/context implementation.

The compact-pool/correction mechanism was informed by
[BlobPager E7b](https://github.com/bdrazn/blobpager/tree/d7c9297ff054a3b54827c8f5eb2f77d01d8126a1).
This is a port/reimplementation for b11960 and Qwen35MoE, not a claim to
replicate that repository's performance. Only `libllama.so.0.8.0` changed;
the server and CPU/CUDA backends were byte-identical to the original bundle.
Feature-off and whole-bank controls used the same custom libllama.

| Three-request pilot | End-to-end tok/s | Decode tok/s | Seconds | Tokens |
| --- | ---: | ---: | ---: | ---: |
| Feature off, original CPU placement | 24.60 | 40.43 | 11.665 | 287 |
| Split machinery with zero pins, all experts on CPU | 23.01 | 34.91 | 13.776 | 317 |
| Actual 64-pin CPU/GPU split | 24.27 | 38.00 | 13.059 | 317 |
| Whole-bank eight-layer SUPER control | 30.45 | 48.25 | 9.524 | 290 |

The split passed its operational checks but was **20.3% lower in token
throughput** and took **37.1% longer** than whole-bank placement. Its
competitive-speed gate failed, so no full split replay was run.

The split copied and verified 3,152,019,456 expert-weight bytes; its pool
occupied 3,052.99 MiB. Counters recorded 8,372 layer calls, 17,223 GPU
expert draws and 49,753 CPU draws: exactly eight contributions per call,
with a **25.7% GPU hit fraction**. Counters include warm-up. CPU work
totaled 3.891 seconds, join/writes 2.762 seconds, launch handling 0.374
seconds and GPU windows 1.519 seconds. These overlapping intervals are
not additive components of request time. Much of the CPU work still
remained on the critical path; the measured split did not repay its
coordination cost on this allocation.

All 26 tested layers had finite real-activation outputs. The zero-pin
all-CPU check had zero difference from the CPU reference; mixed execution's
largest relative L2 discrepancy was 0.0193921, below the frozen 0.03
shakedown bound. This is a wiring/numerics check, not semantic accuracy or
bitwise equivalence. CPU/GPU kernels and reductions can change output
tokens despite unchanged stored weights.

## Quality findings

All three full placements preserved the **12/12 receiver decisions and
occupied-slot states** relative to the frozen baseline. The fastest
placement matched 19/23 sender structures after excluding prose, and
6/36 complete parsed responses. Wording changes alone are not quality
failures; neither is parsed equality an accuracy metric.

The eight-bank and fastest packed placements both produced a selector
with `location=deadline=dock`, violating distinct metaphor-role labels.
The intermediate 38.93 tok/s placement produced a valid selector. Thus
the fastest result does **not** establish quality parity with the
intermediate placement and is not a deployment-ready bridge result.

Manual assistant review of the 12 receivers for each placement found the
same existing defect pattern: five applied placement/return summaries
assert an unsupported librarian approval/check, and the approved-annotation
summary does not clearly convey the authorized condition-check fact. Six
blocked summaries were consistent with their packets; these repeated
identical packets are not independent successes. Reviews bind each
request and response hash in `20261031_expert_split_semantics_v1`.
There was no independent judge and no new live bridge execution.

## Lifecycle, evidence and reproduction

The first three-GPU attempt returned from Triton's unload API before its
CUDA allocation had disappeared. The memory guard stopped it before request
timing. Qwen3-8B was restored. A prospective lifecycle amendment added two
consecutive free-memory samples before startup; the fresh v2 attempt and
packed follow-up passed. Each unload/reload status, original/restored
configuration, readiness result and memory sample is preserved. No OCR
model was unloaded. Initial CMake configuration failure and successful
build logs plus the tested custom library are retained separately.

Raw roots under `data/experiments/word_games/`:

- `20261031_hybrid_experts_pilot_v1`: four actual split/control pilots.
- `20261031_hybrid_build_provenance_v1`: build logs and frozen custom library.
- `20261031_three_gpu_experts_v1`: startup guard stop, zero timed requests,
  successful service restoration.
- `20261031_three_gpu_experts_v2`: two pilots and two full passes.
- `20261031_packed_experts_v1`: two opening pilots, packed full pass and
  ending reference pilot; service restored.
- `20261031_hybrid_three_gpu_analysis_v1`: intermediate immutable aggregate.
- `20261031_hybrid_three_gpu_analysis_v2`: final aggregate and registry receipt.

There were **135 completed timed requests** across successful pilots/full
passes, plus excluded JSON warm-ups. Sources, runtime binaries, inputs,
commands, responses and telemetry are hash-bound; producer snapshots are
preserved. The two typed records retain `confirmed: false` and explicitly
deny established accuracy noninferiority.

Run controllers through Pixi from the repository root, using fresh output
directories. Do not rerun completed roots:

```bash
pixi run --locked -e word-games python -m experiments.word_games.prepare_hybrid_runtime
pixi run --locked -e word-games python -m experiments.word_games.hybrid_experts --arms control pins0 pins64 experts8 --phase pilot --output <fresh-root>
pixi run --locked -e word-games python -m experiments.word_games.three_gpu_experts --output <fresh-root>
pixi run --locked -e word-games python -m experiments.neural_architecture_lab.run_word_games_packed_experts --reference <completed-three-gpu-root> --output <fresh-root>
```

Custom compilation requires the pinned b11960 source extracted under
`.pixi/word-games-hybrid/source` and the build image from
`experiments/word_games/Dockerfile.hybrid-build`. Build and runtime remain
outside `data/`; the binary archived under data is evidence only. The
three-GPU placements use the original native bundle and do not require
the custom library. Source SHA and archive SHA are in the hybrid producer.
