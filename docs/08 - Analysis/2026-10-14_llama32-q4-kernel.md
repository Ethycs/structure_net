# Q4 kernel optimization: 4% throughput gain, speed gate unmet

Completed exploratory study `llama32-3b-q4-kernel-runtime-v1`.
Classification: `speed_target_unmet`; tested, underpowered, `confirmed:false`.
The optimized FP16-core/Q4-tail Llama 3B measured **64.17 output tokens/s**
against the unchanged packed implementation's **61.73**, a **3.96%** gain.
Complete-request time fell **3.81%**. Both passed 20/20 fresh source reviews,
with identical token sequences in all 100 paired timing cells. This misses
the prospective >=1.05x joint latency/token-rate gate. The fresh factorized
FP16 control remained faster at **80.35 tokens/s**.

Canonical aggregate:
`data/experiments/llama_q4_kernel/20261014_registered_v1/campaign_results.json`.
SHA-256:
`20e8a221ce03d481771fe380c398cbb65a58b5bf0ed2fd853d4ccfc167540885`.
The [plan](../07%20-%20Status%20Reports/2026-10-14_llama32-q4-kernel-plan.md),
[load-layout amendment](../07%20-%20Status%20Reports/2026-10-14_llama32-q4-kernel-layout-amendment.md)
and [epilogue amendment](../07%20-%20Status%20Reports/2026-10-14_llama32-q4-kernel-epilogue-amendment.md)
preceded model quality outcomes. Earlier operator evidence is retained;
implementation rounds are not pooled as independent model replications.

This changes execution of the fixed checkpoint from the
[FP16-core/Q4-tail study](2026-10-14_llama32-fp16-core-q4-tail.md), without
training, quantizer changes or weight changes. Its SHA-256 is
`8937d83adfa9c77e065f1c4918bea90a20ee9e1695ffc281561646e0f949a8ff`.
It remains the full 28-layer Llama-3.2-3B-Instruct summarizer, with a rank64
FP16 core and dense packed Q4 residual in each of 84 MLP projections.
Noise and neighborhood distillation describe its inherited training history;
no random noise is added during inference. The parent compression is
approximate, rather than an exact correction to the original teacher.

Attention, tied embedding/output weights, normalization and KV precision are
unchanged. Installing the wrappers verifies every persistent tensor's name,
storage pointer, dtype and shape. Both packed implementations retain exactly
**3,441,776,640 bytes** of persistent weight/buffer payload, plus the existing
256-byte nonpersistent FP32 RoPE frequency buffer. There is no additional
weight compression from this kernel work.

Decode preserves the original Q4 load layout and FP32 split-K partials, uses
four splits instead of eight, and fuses their reduction with the FP16 core's
second projection, tail rounding and final addition. The core's first
projection remains a cuBLAS operation. This removes intermediate operators
while retaining the intended FP16 representation boundaries.

For prefill, three families were tested: direct packed Q4 with FP32 tensor
accumulation; short FP16 tensor-core partial products accumulated across K
tiles in FP32; and coalesced transient expansion followed by cuBLAS. The
selected third path expands only the current matrix to contiguous FP16
`[K,N]` storage, computes the projection, and releases that tensor. It does
this on every invocation; it retains no dense tail in a module or between
requests. Its largest expansion is 50,331,648 bytes. Allocation and expansion
costs are included in timing and peak memory.

The first paired-nibble/fused-core attempt passed numerical checks but was
slower. The second preserved the old decode load layout and coalesced
transient writes. The third separated prefill core computation from the main
packed GEMM to reduce live state. The implementation draws on the official
[Triton grouped matrix multiplication tutorial](https://github.com/triton-lang/triton/blob/main/python/tutorials/03-matrix-multiplication.py)
and [NVIDIA Ampere tuning guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/).
Measured results, rather than theoretical precision throughput, select the
execution family.

The three operator roots contain 98, 82 and 94 measured cases, respectively,
with no resource exclusions. Each also retains twelve irregular-shape
controls. Both actual MLP shapes were tested at M=1/9 for decode and
M=512/1980/6300 for prefill; selected layouts were checked at M=513/6234.
Every v3 output met the finite, relative RMS <=0.002 and cosine >=0.9999
requirements against both old packed output and a dequantized FP16 reference.
The maximum relative RMS was 0.000694 and minimum cosine was 0.99999976.
These are multiplication controls, not summarization quality evidence.

At M=6300, transient full-projection times improved from 15.46 to 13.71 ms
for N8192/K3072 and from 15.18 to 13.31 ms for N3072/K8192. Decode improvements
were smaller and not uniform across shapes. Compiler inspection retained
the actual PTX: the selected packed prefill uses
`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` with 126 registers and
32 KiB shared memory; the staged variant uses
`mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16` with 234 registers and
48 KiB shared memory. FP16 partial accumulation was actually compiled, but
that family did not win the model-level development comparison.

Eleven focused CPU tests passed, covering nibble conversion, configuration,
storage identity, source/cell integrity and quality-first decision rules.
All three longest-prompt CUDA pilots passed graph/eager agreement, cache
numerics, request reset and memory checks. Maximum logit differences were
0.0625/0.03516/0.0625 and cosines exceeded 0.999997 for packed/staged/transient.
Transient's pilot allocation was 6,470,850,048 bytes, about 148 MiB above the
other pilots; these pilots retain extra numerical-control tensors and are
reported separately from the final timing blocks.

Development used the declared adaptive requests 22000–22007, three technical
repetitions and an excluded warmup per case. All candidates passed all eight
strict source reviews. The predefined lowest sum of request medians selected
transient prefill plus the shared optimized packed decode.

| Development family | Source pass | Sum of request medians, s | Output tokens/s |
| --- | ---: | ---: | ---: |
| Direct packed FP32 | 8/8 | 11.1861 | 63.47 |
| Staged FP16/FP32 | 8/8 | 11.0841 | 64.06 |
| Transient FP16 prefill | 8/8 | 11.0275 | 64.38 |

Fresh seeds 26000–26019 were generated before development outcomes, without
filtering or overlap with earlier prompt hashes. There were five cases each
of raw ~500 tokens, raw ~1980, eight raw fragments ~6220, and hierarchy ~1000.
The longest prompt was 6,233 tokens; fragments stayed within 2,048 tokens and
raw prompts within 6,300. Stored summaries stayed within 128 tokens, with
raw768/hierarchy512 generation caps and greedy EOS. The short-input bypass
is not tested. These synthetic final-decision fixtures, including copy-heavy
hierarchy cases, do not represent the full production merge distribution.

Final isolated blocks ran selected, old packed, then factorized FP16, with
five timed repetitions and one excluded warmup per case. All share the fixed
CUDA graph/FlashAttention runtime, width1/9 decoding, current-prompt ngrams1–4
lookup with at most eight draft tokens, and student verification. There is
no cross-request content or KV reuse. Complete-request timing includes CPU
lookup/indexing, prefill, KV transfer and generation through CPU output;
loading, compilation, graph setup and warmups are excluded.

| Fresh final condition | Strict/factual source pass | Payload, GB | Sum of request medians, s | Non-EOS output tokens | Output tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Optimized Q4 kernel | 20/20 | 3.442 | 27.6140 | 1,772 | 64.17 |
| Unchanged old Q4 kernel | 20/20 | 3.442 | 28.7068 | 1,772 | 61.73 |
| Optimized factorized FP16 | 20/20 | 4.135 | 22.0537 | 1,772 | 80.35 |

The judge was Codex reading final source quotations, aided by label,
cardinality, field, source-presence, token-limit and EOS checks. There was no
independent judge model. Every candidate summary quotes the correct final
decision without adding claims or preserving superseded proposals. Control
outputs were checked for exact identity to those reviewed texts and sources.
All conditions were stable across five repetitions, and optimized Q4 matched
both controls' complete token sequences in all 100 paired cells.

Both primary speed ratios versus old packed are **1.0395748x**, below 1.05x.
The quality gate passes; the speed gate fails. Both ratios versus FP16 are
about **0.7987x**: FP16 speed is not recovered. The broader 2x native-model
target was not tested in this study. No configuration was tuned or retried
after fresh final outcomes.

| Request family | Old/optimized complete-request time ratio |
| --- | ---: |
| Raw ~500 tokens | 0.9802x |
| Raw ~1980 tokens | 1.0238x |
| Eight raw fragments ~6220 tokens | 1.0519x |
| Hierarchy ~1000 tokens | 1.0013x |

Long batches gain most; short raw requests regress about 2%, and hierarchy
requests barely change. The sum of per-case median TTFT falls from 16.2566
to 15.3928 seconds; FP16 is 10.0808. Remaining time, derived by subtracting
these sums from request-time sums, is 12.4502/12.2212/11.9729 seconds for
old/optimized/FP16. These are aggregate timing components, not separately
measured operator FLOPs or attention/MLP attribution.

| Final block | Peak PyTorch allocated bytes | Peak reserved bytes | Graph setup, s |
| --- | ---: | ---: | ---: |
| Optimized Q4 | 5,740,232,704 | 6,192,889,856 | 2.9232 |
| Old Q4 | 5,689,901,056 | 6,188,695,552 | 2.9825 |
| FP16 | 6,280,773,632 | 6,775,898,112 | 0.7926 |

Final peak allocation increases exactly 48 MiB versus old Q4 and remains
8.6% below FP16. Weight payload is unchanged, and the parent's 16.8% payload
saving versus factorized FP16 is retained. Twenty excluded warmup requests
totaled 27.65/37.78/22.09 seconds for optimized/old/FP16; these steady-state
measurements are not cold-start latency claims.

Kernel improvements do not remove the dense residual's matrix work. The
parent's analytical MLP coefficient count is about 2.244x the rank1024 FP16
model's, despite its smaller stored weights. This remains a plausible reason
for the unrecovered FP16 speed, rather than a measured FLOP or bottleneck
attribution. A useful next study is Q4 packing of the remaining factor
matrices while retaining factorized execution, with neighborhood repair if
needed. That changes the representation and has not been tested here.

Integrity: 372/372 primary timing cells, comprising 72 development and 300
fresh final cells; zero failed, retried or reused timing cells. There were 84
excluded request warmups, plus separate operator and lifecycle controls.
One frozen trained checkpoint is the unit of model replication; input seeds
and timing repeats do not create independent training replicates. Physical
GPU0 alone was used, an RTX3060 12GB with Torch2.4.1+cu121, Triton3.1.0,
Transformers4.53.3 and FlashAttention2.6.3. GPU0 was released after completion;
the other reserved devices and unrelated worktree files were preserved.

The aggregate catalogs frozen producers, source snapshots, models/libraries,
all cell identities, source decisions, compiler artifacts, pilots and the
development coordinator. Additional runtime/source/fixture binding and
cross-control token readback is retained in
`data/experiments/llama_q4_kernel/20261014_runtime_readback_v1.json`, SHA-256
`3dd2e29d1f86aaaf366f8d5c3fe4ff9bd86ae12b4149e9dcb3f014e78714884b`.
This evidence does not establish production 95% accuracy, mobile speed,
energy savings or co-tenancy performance.

The local hypothesis index retains `tested:true`, `confirmed:false` and the
`speed_target_unmet` classification. Exact-ID, semantic and full-JSON
readback receipts are retained in the aggregate directory's
`registry_readback.json` and
`data/hypothesis_registry/llama32-3b-q4-kernel-runtime-v1.json`.

Reproduction uses `experiments.structure_net.campaign_llama_q4_kernel` for
fixtures/operators/pilots/timing, `inspect_llama_q4_kernel` for PTX inspection,
and `experiments.neural_architecture_lab.finalize_llama_q4_kernel` for audit,
source-review analysis, selection, finalization and conservative registry
readback. The selected wrapper is in `llama_q4_kernel_v3`; it shares decode
and transient expansion helpers from v2 and numerical/epilogue helpers from
v1. Parent runtime modules and checkpoint files remain unchanged. Exact
commands and configurations are in each `producer.json`; use new output roots
and select physical GPU0 before starting Python, never overwrite these runs.
