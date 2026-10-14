# FP16 tensor core with a Q4 residual: smaller, slower

Completed exploratory study `llama32-3b-fp16-core-q4-tail-v1`.
Classification: `smaller_speed_unmet`; tested, underpowered, `confirmed:false`.
The selected rank-64 model reduced weight payload by **16.8%**, retained
20/20 fresh source-reviewed summaries and produced the same token sequences
as FP16 in all 100 paired timing cells. Complete-request throughput fell from
**80.45 to 61.80 output tokens/s**. This implementation misses the speed goal.

Canonical aggregate:
`data/experiments/llama_fp16_core_q4_tail/20261014_registered_v1/campaign_results.json`.
SHA-256:
`ae9e27b4e5d213e99c9d71dc7d4530f714e6439b4913a418c37fbbe8301a3962`.

The [prospective plan](../07%20-%20Status%20Reports/2026-10-14_llama32-fp16-core-q4-tail-plan.md)
and [operator amendment](../07%20-%20Status%20Reports/2026-10-14_llama32-q4-tail-operator-amendment.md)
fix the decision order. This is one already-trained Llama-3.2-3B-Instruct
checkpoint, inherited from the
[noise and neighborhood-distillation study](2026-10-14_llama32-noise-tensor-continuation.md),
with zero new optimizer updates. It is a full 28-layer summarizer.

For each trained rank-1024 MLP projection `W*=L R`, preserve the first 64 or
128 factor components in FP16 and quantize the dense product of the remaining
components. Construct the tail in FP32 with TF32 disabled. Store signed uniform
Q4 weights with one FP16 scale per output channel and 64 input channels.
Execute `FP16-core(x) + Q4-tail(x)` with FP16 activations and FP32 accumulation.
The tail contains fixed trained weight components. “Noise” describes the
parent's training history and quantization error, rather than random inference
noise. This is approximate compression, not a lossless correction back to the
original pretrained teacher. The format differs from GGUF Q4_K_M.

Attention, embedding/output weights, normalization weights and KV use FP16;
RoPE retains its existing FP32 frequency buffer. The output embedding remains
tied. Packed tails execute directly through Triton, including one-position
decode and nine-position verification; no persistent dense FP16 tail is cached.
For decode, split-K materializes small FP32 partial output activations, which
are reduced and converted to FP16. These buffers contribute to runtime memory.

Five CPU storage/numerical tests, three decision-rule tests and two provenance
and cell-integrity tests passed. The initial eager operator pilot passed nine
numerical cases but was slower than dense FP16. Its immutable evidence remains
in `20261014_operator_pilot_v1`. Before any model quality result, the amendment
tested 44 fixed kernel configurations/cases under CUDA graphs, over both MLP
shapes and M=1/9/512/6300. Every case met the RMS <=0.002, cosine >=0.9999 and
finiteness checks. The selected decode layout was `(16,64,64,8,4,2)` and prefill
layout `(128,128,32,1,8,3)` for tile M/N/K, split-K, warps and stages.

The selected Q4 operation was approximately 1.8–2.1x faster than dense FP16 at
decode widths, but slower at prefill widths. That control tests multiplication
by the same dequantized weights; it does not test the faster factorized model.
Both full-model longest-prompt pilots passed cache numerical agreement,
graph/eager token agreement, fresh-request reset and memory checks. Their
maximum logit differences were 0.05640/0.03906 and cosines
0.9999967/0.9999986 for rank64/rank128. Independent physical checkpoint
readback verified all 84 core slices against the trained parent, 84 packed
tails and their 84 scale buffers, unique storage counts and embedding ties.

Development reused the declared adaptive requests 22000–22007, with three
technical repetitions per candidate. Both passed all eight source reviews;
rank64 won by the smallest sum of median complete-request times.

| Development candidate | Weight payload, bytes | Strict source pass | Sum of request medians, s | Output tokens/s |
| --- | ---: | ---: | ---: | ---: |
| FP16 core64 + Q4 tail | 3,441,776,640 | 8/8 | 11.5001 | 61.74 |
| FP16 core128 + Q4 tail | 3,562,887,168 | 8/8 | 11.6968 | 60.70 |

Fresh requests 25000–25019 were generated before development outcomes, with no
filtering and no overlap with earlier request hashes. Each family contributed
five cases: raw ~500 tokens, raw ~1980, eight raw fragments ~6220, and hierarchy
~1000. Maximum prompt length was 6,234 tokens. Input fragments stayed within
2,048 tokens, raw prompts within 6,300, output caps were raw768/hierarchy512,
and each stored summary stayed within 128 tokens. Short-source bypass is not
tested. These are synthetic final-decision summaries, including copy-heavy
hierarchy fixtures, rather than production raw or merge prompts.

Run five repetitions per case in isolated blocks: selected mixed precision,
optimized FP16, then native Q4 plus lookup8. Each student request had an
excluded per-case warmup. Native used its existing short warmup and five
excluded longest-request cells. All use greedy generation through EOS. Student
conditions share the fixed CUDA graph/FlashAttention/current-prompt lookup
runtime from the [runtime study](2026-10-14_llama32-tensor-runtime.md).
There is no cross-request content or KV reuse. Timing includes CPU prompt
lookup/indexing, prefill, KV transfer and generation through CPU output;
loading, hashing, compilation, graph setup and warmups are excluded.

| Fresh final condition | Strict source/format pass | Factual pass | Weight payload, GB | Sum of request medians, s | Non-EOS output tokens | Output tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP16 core64 + Q4 tail | 20/20 | 20/20 | 3.442 | 28.6106 | 1,768 | 61.80 |
| Optimized rank1024 FP16 | 20/20 | 20/20 | 4.135 | 21.9758 | 1,768 | 80.45 |
| Native Q4_K_M + lookup8 | 15/20 | 20/20 | 2.012 | 21.3341 | 1,676 | 78.56 |

Factual scores are the planned exploratory separation from strict formatting,
read from the saved source-review ledger. Native's five eight-fragment outputs
placed each ID on its own line, producing 16 nonempty lines instead of the
required eight labeled summary lines. All their final facts were correct.
Its concise hierarchy owner/budget/date listing also preserves the source
facts. The judge is Codex reading paired final source quotes, aided by label,
cardinality, field, token-limit and EOS checks. There is no independent judge.
Each condition was stable over its five repetitions; candidate and FP16 were
also token-identical in all 100 paired cells.

The >=19/20 quality gate and >=10% weight-payload saving gate pass. Both speed
ratios versus optimized FP16 are **0.76810x**, below the required 1.05x.
Request latency increased **30.2%**, while throughput decreased 23.2%.
Against native Q4, the latency ratio is **0.74567x** and token-throughput ratio
**0.78660x**, both below the broader 2x goal. Native emits shorter outputs, so
its token-rate and request-latency rankings differ; use both endpoints.

| Median-time sums, seconds | Core64/Q4 | Optimized FP16 | Native Q4 |
| --- | ---: | ---: | ---: |
| Time through first token | 16.2884 | 10.1160 | 12.2733 |
| Remaining request time, derived by subtraction | 12.3222 | 11.8598 | 9.0608 |

Most of the extra request time appears before the first token. The original
rank1024 MLP uses 968,884,224 multiply-accumulate coefficient applications per
token across its factored projections. The new dense tail plus rank64 core
uses 2,174,484,480, about **2.244x** as many. This is an analytic matrix-work
count, not a GPU FLOP measurement. Four-bit storage does not remove that dense
work, and split-K reduction/core addition adds operators. This explains why a
decode kernel advantage over dense FP16 did not yield a model speed advantage.

The serialized rank64 checkpoint is 3,441,905,090 bytes, including serialization
overhead; its unique persistent weight/buffer payload is 3,441,776,640 bytes.
Runtime adds a 256-byte nonpersistent RoPE frequency buffer. Rank128's file is
3,563,015,618 bytes with a 3,562,887,168-byte payload. These are weight-state
sizes, excluding tokenizer/configuration files and the software installation.
Peak PyTorch allocation in the final blocks was 5,689,937,920 bytes for
core64/Q4 and 6,280,810,496 for FP16, a 9.4% reduction. Native's after-load
NVML increment was 3,256,877,056 bytes, measured differently and unsuitable
for a direct peak-allocation comparison. Graph setup took 2.7144 versus
0.7580 seconds. Twenty excluded final warmup requests totaled 43.55 versus
21.97 seconds; compilation/cold-start cost is not a steady-state speed claim.

Campaign integrity: 348/348 primary cells, zero failed/retried/reused cells;
48 development and 300 final timing cells, with 56 excluded student request
warmups and five excluded native timed warmups. One trained model is the unit
of model replication; evaluation seeds select inputs, and timing repetitions
are technical repeats. GPU0 was an RTX3060 12GB with Torch2.4.1+cu121,
Triton3.1.0, Transformers4.53.3 and FlashAttention2.6.3. Native uses the pinned
existing runner/llama.cpp binary and Q4 model SHA
`6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff`.
Source snapshots, library/model readback, all cell identities, source reviews
and checkpoint audits are retained under the aggregate root and its artifact
catalog. Existing completed parent evidence and unrelated user files were
preserved.

This result establishes useful weight compression with retained quality on
this proxy, alongside a measured speed regression. It does not establish
production 95% accuracy, mobile performance, energy savings, co-tenancy or
quality outside the tested workload. A useful next comparison is Q4 packing
of the remaining factor matrices themselves, retaining factorized execution,
with neighborhood/quantization-aware repair if quality requires it. That
variant has not been tested here.

Reproduction uses the frozen modules
`experiments.structure_net.run_llama_fp16_core_q4_tail` for fixtures, packing,
pilot and evaluation, `experiments.structure_net.tune_llama_q4_tail` for the
pre-quality operator selection, and
`experiments.neural_architecture_lab.finalize_llama_fp16_core_q4_tail` for
analysis, selection, finalization and registry readback. Run on a fresh output
root with physical GPU0 selected before Python starts; exact producing
commands/configurations and hashes are in each `producer.json`.
