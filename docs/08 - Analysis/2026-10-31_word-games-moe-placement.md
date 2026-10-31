# WORD_GAMES MoE placement and CPU tuning

Completed exploratory study, 2026-10-31. Hypothesis
`word-games-moe-placement-v1`; classification `no_speedup_with_replay_parity`,
`confirmed: false`. **A speed gain was measured, but task-quality preservation
was not established.** No production configuration was changed.

Moving only the routed-expert banks for eight blocks onto the RTX 2060 SUPER,
while keeping attention and recurrent state on the RTX 3060, increased full-request
throughput from **26.17 to 31.02 tok/s (+18.5%)**. The same 36 input requests
finished in **160.49 versus 185.72 seconds (13.6% less elapsed time)**. Generated
answers differ, so these are equal-input workload comparisons, not identical-output
execution times. Whole-block placement, CPU pinning and the tested 2 GiB expert
cache did not improve full-request token throughput.

The strict parsed-output parity gate is separate from semantic accuracy. A
wording difference does not itself prove a quality regression. Manual inspection
nevertheless found unsupported approval claims and annotation-summary omissions;
this run does not establish high task accuracy or readiness to deploy the bridge.

[Prospective protocol](../07%20-%20Status%20Reports/2026-10-31_word-games-moe-placement-plan.md).
This uses the existing **Empero Qwen3.8-35B-A3B distillation**, not PrismML
weights. The parent model's live metaphor failure and unsupported approval
summaries are not repaired or reclassified by a hardware timing study.


Aggregate: `data/experiments/word_games/20261031_moe_placement_analysis_v1/campaign_results.json`.
SHA256: `5fa424e5397727601b695e37e63bf8e822b6cc4f7667c63113144d012b1bec61`.

## Full replay results

Eleven complete passes, **396/396 valid untruncated JSON responses**, on one
checkpoint. These are timing repetitions, not independent trained-model seeds.
The three control times were 185.60, 185.39 and 186.17 seconds; the ending
control therefore corroborates the measured expert-only gain.

| Configuration | Passes | Mean seconds | Full-request tok/s | Decode tok/s | Parsed answers matching control |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2048 MiB expert cache on 3060 | 2 | 182.19 | 26.14 | 39.79 | 16/36 |
| 26 CPU expert blocks, 12 threads | 3 | 185.72 | 26.17 | 41.67 | 36/36 |
| Same, CPU core pinning | 2 | 196.88 | 24.69 | 38.57 | 36/36 |
| Eight expert banks only on SUPER | 2 | 160.49 | 31.02 | 46.52 | 13/36 |
| Eight whole blocks on SUPER | 2 | 191.95 | 25.99 | 47.58 | 13/36 |

The expert-only passes took 160.79 and 160.19 seconds, both over 5% faster than
every control. They generated 4,978 tokens each, versus 4,861 for control.
Their mean prefill time was 53.30 seconds and decode time 107.00 seconds;
control means were 68.87 and 116.65 seconds. Whole-block placement increased
prefill to 86.92 seconds, offsetting its faster decoding. The two-cache-pass
mean was 182.19 seconds with 4,763 output tokens: about 2% less elapsed time,
but no token-throughput gain and a lower decode rate.

All conditions preserved the twelve frozen receiver decisions and occupied-slot
states. Sender structured responses matched control in 23/23 calls for CPU
pinning, 19/23 for whole-block placement, and 20/23 for cache and expert-only
placement. These differences can include retrieval choices, not just wrong
final proposals. Whole-block/cache selector outputs were structurally valid;
control, CPU-pinned and expert-only selectors still duplicated a role label.
No new live bridge workflow accuracy is claimed.

Manual assistant review of all twelve receivers in each changed full arm found
approval assertions unsupported by the packet in normal reserve, private note,
private title, changed due and cancellation. Approved-annotation summaries also
conflated a condition check with approval; cache and expert-only summaries omitted
the condition-recording fact. Whole-block cancellation wording could imply entity
deletion rather than releasing occupancy. The six blocked summaries were
consistent with their packets. Repeated packets/outputs are not independent
semantic examples, and no independent judge was used. Full review bindings are
in `20261031_moe_placement_review_v1/`; the aggregate checks their raw hashes.

## Expert-only follow-up and memory

The [prospective follow-up amendment](../07%20-%20Status%20Reports/2026-10-31_word-games-moe-experts-only-amendment.md)
was written after observing the first whole-block full pass. Its pilot took
9.85 seconds against 11.45 for the same three requests in the fresh controls,
passing the fixed 5% advancement threshold. Both full passes then ran.

This arm assigns all layers and KV/recurrent state to CUDA0, overrides only
routed-expert tensors in blocks 0-7 to CUDA1, and keeps expert tensors in blocks
8-25 on host. Runtime logs verify the exact sets and a **3852.00 MiB CUDA1
model buffer** plus a 92.51 MiB CUDA1 compute buffer. This is a bank-per-layer
placement, not a split of individual experts within each bank.

| Configuration | Observed 3060 peak MiB | Observed SUPER peak MiB | SUPER minimum free MiB |
| --- | ---: | ---: | ---: |
| control | 8962 | 2702 | 5085 |
| cpu12pin | 8962 | 2702 | 5085 |
| gpu8 | 8390 | 7192 | 595 |
| cache2048 | 11028 | 2702 | 5085 |
| experts8 | 8796 | 6772 | 1015 |
| control | 8962 | 2702 | 5085 |

Whole-device figures include resident allocations (291 MiB on 3060 and
2702 MiB on SUPER before testing). Host process swap remained zero. The
protected 6 GB 2060 stayed at 5646 MiB and its Qwen3-8B readiness endpoint
returned HTTP 200 after cleanup. Test workers were stopped. PCIe peer read
access reports `NS` between these GPUs; no direct peer-access benefit is assumed.

The 2 GiB cache reported 61.81% hit rate for batches up to eight tokens and
1,399,625.51 MiB uploaded on misses, approximately 1.33 TiB across the worker's
two full passes and warm-up. Larger-batch cache reads avoided some uploads,
but misses still produced substantial traffic. This is a software byte counter,
not a PCIe bandwidth measurement or proof that every cache budget will be slow.

## Fixed configuration

Same pinned Q4_K_M checkpoint, native llama.cpp b11960/container, Flash Attention,
Q8_0 K/V, context 8192, batch/ubatch 512, temperature 0, seed 73, thinking and
speculation disabled. Prompt reuse disabled. Same 36 frozen requests and caps.
Model hash, runtime hashes, producer hashes and source archives accompany each
campaign. No weights are retrained, requantized or removed.

Host: Xeon E5-2697 v2, 12 physical cores / 24 logical CPUs, AVX and F16C but
no AVX2. RTX 3060 12 GB is the primary GPU. The RTX 2060 SUPER has about 5 GiB
free around its resident service; the separate 6 GB RTX 2060 remains occupied
by Qwen3-8B and is never exposed to the test container. Existing services are
preserved. GPU links traverse the PCIe host bridge, with no NVLink; SUPER
negotiates an x8 link. Do not interpret its idle PCIe generation as a permanent
speed limit.

The baseline keeps expert weights for blocks 0-25 on host. The second-GPU arm
places whole blocks 0-7 on SUPER, host expert weights for blocks 8-25, and the
remaining model on 3060. Runtime assignment logs, not generic layer-offload
counts, verify this. The cache arm retains baseline placement and allocates
2046.77 MiB of GPU expert cache for a requested 2048 MiB. This preserves stored
weights but changes execution kernels; bit-identical generated text is not
assumed.

## CPU implementation audit

The runtime reports AVX, F16C, LLAMAFILE, OPENMP and REPACK build support.
However, build support does not mean the optimized repacking path executes:
the pinned [Q4_K x86 repack selector](https://github.com/ggml-org/llama.cpp/blob/b11960/ggml/src/ggml-cpu/repack.cpp)
requires AVX2. This processor cannot use that path. The actual host expert
buffers are `CUDA_Host` (pinned host memory), not CPU_REPACK.

[KTransformers' documented fallback](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/AVX2-Tutorial.md)
requires AVX2 and FMA, which this Xeon lacks. It is not an available accelerated
backend on this machine under its documented configuration. No unsupported
build was installed. The host has one NUMA node, so cross-node placement tuning
is not applicable. These observations do not rule out writing a new AVX/SSE
kernel; such a kernel was not implemented or measured here.

The pinned [MoE cache implementation](https://github.com/ggml-org/llama.cpp/blob/b11960/src/llama-moe-cache.cpp)
uses segmented LRU storage for selected experts. It copies their existing
quantized bytes into VRAM and uses the associated GPU kernels. The cached
execution path is limited to batches up to 32 tokens; larger prefill batches
use the other path. Consequently decode improvement need not translate into
the same improvement for full requests. Expert-weight caching across requests
is permitted in this arm; KV/prompt caching remains disabled.

## Pilot evidence and selection

These three-request runs are lifecycle/screening evidence, not final speed or
quality estimates. All successful pilots returned three valid, untruncated
JSON objects.

| Pilot | Completion seconds | Output tok/s | Decode tok/s |
| --- | ---: | ---: | ---: |
| Control, 12 unpinned threads | 15.58 | 18.42 | 27.72 |
| 6 threads | 18.29 | 15.69 | 22.10 |
| 8 threads | 15.95 | 17.99 | 26.75 |
| 24 threads | 192.41 | 1.49 | 1.55 |
| 12 threads, one logical CPU per physical core | 11.66 | 24.62 | 40.56 |
| 8 whole blocks on SUPER | 11.82 | 25.04 | 48.75 |
| 2048 MiB expert cache | 10.93 | 26.17 | 41.46 |
| SUPER plus CPU pinning | 11.91 | 24.86 | 47.91 |
| SUPER plus CPU pinning and cache | 10.68 | 26.32 | 56.37 |

The CPU selector chose 12 pinned threads. The three selected CPU-pinning pilot outputs
matched the control. GPU placement and cache changed generated text; their
selector outputs passed the role validator where the control reused `dock`.
This is a diagnostic change, not evidence of full live bridge success.

The fastest combination improved less than the predefined 5% margin over the
fastest individual pilot, so it did not advance. Two full passes each were
scheduled for control, pinned CPU, SUPER, and cache, followed by one ending
control to check drift.

## Measurement boundaries and integrity

Completion timing includes HTTP completion and prefill; it excludes checkpoint
hashing, loading, native/custom warm-up, template application and artifact I/O.
Native decode timing is reported separately. First-use SM75 compilation is
startup work, not steady-state inference. GPU telemetry samples run at roughly
one-second intervals; reported peaks are observed samples, not allocation
instrumentation. Resident services and background host activity remain, so this
is an exploratory deployment measurement, not an isolated hardware benchmark.

The speed-and-parity gate requires at least 5% lower completion time on both
full passes versus every beginning/ending control, plus all 36 valid complete
JSON answers and exact parsed-JSON equality with the initial control. This is
a finite-suite regression check, not semantic task accuracy. Structured state
agreement without prose agreement is explicitly a separate post-hoc diagnostic.

Pilot v1 was stopped during hashing, before any startup or request, to correct
comma handling in the placement-log parser. Pilot v2's cache arm stopped before
warm-up because its allocation guard did not recognize capitalized `MoE cache`;
actual allocation was successful. Corrected v3 recognizes a positive numeric
allocation. Both failures and the prospective amendments remain visible.
These corrections change observation guards, not inference settings.

Raw roots under `data/experiments/word_games/`:
`20261031_moe_placement_pilot_v1`, `20261031_moe_placement_pilot_v2`,
`20261031_moe_placement_pilot_v3`, `20261031_moe_placement_research_v1`,
`20261031_moe_placement_full_v1`. Additional roots are `20261031_moe_placement_experts_pilot_v1`,
`20261031_moe_placement_experts_full_v1` (includes the ending control),
`20261031_moe_placement_review_v1`, and `20261031_moe_placement_analysis_v1`.
There were ten successful three-request pilots, one preserved cache-guard startup
failure, one pre-start hashing interruption, and eleven successful full passes.
Code/runtime stay outside `data/`; evidence is append-only.


## Individual experts split between CPU and spare GPU

The user's later proposal is a finer execution split: keep selected individual
experts resident on SUPER, execute other routed experts on CPU, and combine
both weighted contributions while attention stays on 3060. No selected expert
should be dropped. Potential overlap must be implemented; partitioning storage
alone does not imply concurrent CPU/GPU execution or a speedup.

This is **not implemented or measured here**. Native b11960's cache assigns
banks to the layer device and uploads cache misses before GPU execution; it
does not expose the proposed spare-card GPU-hit/CPU-miss executor. Its tensor
overrides address whole expert tensors, not individual expert indices.
[KTransformers documents per-expert CPU/GPU scheduling and frequency placement](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/experts-sched-Tutorial.md),
but its documented CPU paths do not support this AVX-only host.
[BlobPager](https://github.com/bdrazn/blobpager) is a research reference implementing
GPU hits with CPU misses in llama.cpp. Its reported model/hardware differs;
no published speed claim is adopted as a measurement on our system.

A future implementation should profile representative routing, compare static
frequency placement with uniform placement at equal VRAM, retain CPU fallback
for every selected expert, measure synchronization and misses, and use separate
calibration/evaluation traffic. Our measured whole-bank expert-only result is
the relevant 31 tok/s control. CPU/GPU arithmetic and summation order can change
outputs even with unchanged stored weights, so semantic and live workflow
validation remain necessary.

## Reproduction and verification

From the repository root, using fresh output directories:

```sh
env CUDA_VISIBLE_DEVICES='' pixi run --locked -e word-games python -m experiments.word_games.moe_placement --phase pilot --configs control cpu6 cpu8 cpu24 cpu12pin gpu8 cache2048 --output data/experiments/word_games/NEW_PILOTS
env CUDA_VISIBLE_DEVICES='' pixi run --locked -e word-games python -m experiments.word_games.moe_placement --phase full --configs control cpu12pin gpu8 cache2048 --repeats 2 --output data/experiments/word_games/NEW_FULL
env CUDA_VISIBLE_DEVICES='' pixi run --locked -e word-games python -m experiments.word_games.moe_placement --phase pilot --configs experts8 --output data/experiments/word_games/NEW_EXPERT_PILOT
# Only after the pilot gate described in the amendment:
env CUDA_VISIBLE_DEVICES='' pixi run --locked -e word-games python -m experiments.word_games.moe_placement --phase full --configs experts8 --repeats 2 --ending-control --output data/experiments/word_games/NEW_EXPERT_FULL
```

Conformance profile: EXPLORATORY. Focused verification: 84 WORD_GAMES tests
passed, covering placement guards, cache allocation, matched output comparison,
conservative registration and expert-only overrides. The complete strict
speed-plus-parsed-parity gate did not pass, so registration stays unconfirmed;
the separate speed-only result is positive for `experts8`.
