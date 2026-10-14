# Frozen tensor student: GPU replay recovers prototype speed

**Status:** completed exploratory systems study, 2026-10-14.  
**Hypothesis:** `llama32-3b-tensor-runtime-v1`.  
**Classification:** `prototype_speed_recovered_native_target_unmet`.  
**Design:** [prospective plan](../07%20-%20Status%20Reports/2026-10-14_llama32-tensor-runtime-plan.md)
and [native warmup amendment](../07%20-%20Status%20Reports/2026-10-14_llama32-tensor-runtime-warmup-amendment.md).  
**Aggregate:** `data/experiments/llama_tensor_runtime/20261014_registered_v1/campaign_results.json`.  
**SHA-256:** `cb9ec87769f9179368a6c248eb9e29778524d1f63aa2191c93b41795eea14397`.

GPU graph replay, fused KV updates and verified prompt lookup increased the
frozen tensor student's complete-summary throughput from **13.80 to 80.76
output tokens/s**, a **5.853×** gain. Both student runtimes passed all twenty
fresh synthetic requests and produced identical token sequences across all five
repetitions. Native Q4 plus lookup reached **83.28 tokens/s**. The student did
not meet the **2× native throughput and complete-request speed** gate.

This establishes a faster implementation of the existing quality-passing
student. It does not establish a further compression, fewer FLOPs, a benefit
from noise training, production application accuracy, or phone performance.

## Configuration and integrity

The frozen model is the full **28-layer Llama 3.2 3B Instruct** with all MLP
gate/up/down projections replaced by the trained rank-1024 factors. Attention,
embedding, normalization and residual connections retain the original weights.
The chosen checkpoint is all fourteen SHA-verified selected groups from
`data/experiments/llama_neighbourhood_continuation/20261014_noise_global_v2`.
There were **no optimizer updates or new noise perturbations** in this study.
The parent [noise continuation report](2026-10-14_llama32-noise-tensor-continuation.md)
remains unchanged.

Weights and activations use FP16. Execution uses physical **GPU0, RTX 3060
12 GB**, one worker, two CPU threads, Torch 2.4.1+cu121, Transformers 4.53.3 and
FlashAttention 2.6.3. GPUs1/2 were untouched. Original HF shards, configuration,
tokenizer manifest, selected factors, worker code, FlashAttention extension and
Llama implementation were hashed. Source snapshots and strict JSON survive
under each new dated root.

There are **396/396 analyzed timing cells**, zero failed/retried/reused cells,
and five prospectively excluded native longest-request warmup cells:

| Stage | Conditions | Requests | Timed repetitions | Cells |
| --- | ---: | ---: | ---: | ---: |
| Adaptive development | 4 | 8 | 3 | 96 |
| Fresh student comparison | 2 | 20 | 5 | 200 |
| Fresh native control | 1 | 20 | 5 | 100 |

These are technical timing repetitions, not independently trained models.
The study is exploratory and underpowered for training or population-accuracy
claims. Eight focused tests passed, including verification-prefix rollback,
EOS/cap handling, fragment attribution, worst-case repeated-output review,
source identity and the joint quality/speed gates.

The CUDA pilot used the longest prompt. Its nine-position teacher-forced
comparison had maximum absolute logit difference **0.03125** and cosine
**0.99999857**, passing the declared 0.25/0.999 thresholds. Captured and eager
custom outputs agreed, as did a repeated fresh request. Pilot peak allocated
memory was **7,012,414,976 bytes**. Pilot outputs are systems checks and are
excluded from model-quality evidence.

The fixture file was generated before hash instrumentation and the numerical
pilot width were expanded. The exact original source was recovered and its
SHA matches the fixture's declared producer hash; the fixture generator text
and parent generator hash are unchanged. This additional audit is retained in
`20261014_fixture_generator_readback_v1.json` with its source snapshot. No
fixture content, primary threshold or worker outcome was changed.

## Execution changes and lookup

The old reference executes HF DynamicCache, 512-token prefill chunks and one
Python-driven decode step per generated token. The custom implementation uses
full-prompt HF prefill and copies each layer's current-request KV to a bounded
static buffer. Decode executes the same factors directly and uses
[FlashAttention's fused KV update and rotary API](https://github.com/Dao-AILab/flash-attention/blob/v2.6.3/README.md).
Two CUDA graphs capture verification widths one and nine. GPU argmax and
finiteness checks return only a small vector to the CPU.

Lookup indexes token sequences in the **current prompt only**. It finds the
longest matching suffix of the generated output, from one to four tokens, and
proposes up to eight subsequent prompt tokens. The complete student verifies
`[pending token, proposed tokens]` causally. Matching candidates and the next
student-predicted token are emitted; the logical KV length rolls back past any
rejected candidates. Width-nine verification pads shorter proposals, and causal
masking excludes future padding from earlier predictions. No gold references,
teacher outputs or previous-request evidence are available to this mechanism.

This is an implementation of the
[prompt lookup decoding strategy](https://huggingface.co/docs/transformers/v4.53.3/generation_strategies),
with custom CUDA replay. Allocations and executable graphs persist between
requests; every prompt is processed again and its evidence KV is replaced.
The FP16 rounding difference in the fused rotary/attention path means this is
not a general bitwise-equivalence claim, although all measured student token
sequences agreed.

## Development selection

All four arms passed **8/8** explicit source reviews on the known adaptive
development examples. Three repetitions per request followed one excluded
longest-request warmup per arm. Repetition order reverses on odd repetitions.

| Runtime, same frozen weights | Summed per-request median seconds | Output tokens/s |
| --- | ---: | ---: |
| Previous HF prototype | 57.008 | 12.45 |
| Custom eager decode, full prefill | 42.215 | 16.82 |
| CUDA graph decode | 16.596 | 42.78 |
| CUDA graphs plus lookup8 | **8.782** | **80.85** |

The frozen rule selected **graph_lookup8**, the fastest eligible arm. Selection
and all development reviews were written before final execution.

## Fresh final quality and speed

Seeds **24000–24019** were generated prospectively with the unchanged parent
generator/instructions: five requests each of raw ~500 tokens, raw ~1980 tokens,
eight fragments ~6220 tokens and hierarchy ~1000 tokens. They do not overlap
train/development or the previously used 23000-series test. All inputs stayed
within 2048 tokens per fragment, eight fragments and a 6300-token prompt. Raw
and hierarchy generation caps were 768 and 512; summaries stayed <=128 tokens.

The judge is **Codex assistant source review**, aided by line-scoped field,
cardinality, EOS and token-limit checks. There is **no independent judge model**.
Every distinct output is linked to a text hash, source hash, final-decision
quotes and an explicit rationale. Repeated outputs were stable for all three
conditions. The student's 55 fragment summaries exactly matched the source
final decisions, with a maximum of **33 tokens per summary**.

| Runtime | Strict request passes | Source-factual passes* | Mean per-request median seconds | Output tokens/s |
| --- | ---: | ---: | ---: | ---: |
| Same student, old runtime | 20/20 | 20/20 | 6.425 | 13.80 |
| Same student, graphs + lookup8 | **20/20** | **20/20** | **1.098** | **80.76** |
| Original native Q4_K_M + lookup8 | 15/20 | 19/20 | 1.043 | 83.28 |

\* Source-factual counts separate content from formatting after inspecting the
outputs; they are exploratory and do not replace the preregistered strict gate.
All five native eight-fragment requests placed each ID on its own line and its
summary on another, producing sixteen nonempty lines instead of eight. Four
were factually correct. On `heldout_raw_24006_8_6220`, F04 also copied F06's
**$7,100 / November 14** into Project Aspen 4, whose source approves
**$5,800 / November 22**. Thus formatting accounts for most of the strict score
difference. The factual difference is one request, not evidence of a large
general accuracy gain.

Per-family sums below contain five per-request median latencies each:

| Family | Old student seconds | Optimized student seconds | Native Q4 seconds |
| --- | ---: | ---: | ---: |
| Raw ~500 | 11.279 | 1.226 | 1.215 |
| Raw ~1980 | 12.789 | 2.669 | 3.003 |
| Eight fragments ~6220 | 92.628 | 16.341 | 14.866 |
| Hierarchy ~1000 | 11.795 | 1.718 | 1.772 |

For each condition, throughput divides median output-token counts by summed
per-request median wall seconds, excluding EOS tokens. The student emitted
1773 output tokens; Q4 emitted 1737. Complete-request latency is therefore a
separate primary endpoint. Timing starts with CPU prompt IDs and ends with
CPU output IDs, excluding model load, graph capture and warmup. The historical
reference helper allocates its prompt tensor immediately before its internal
timer; the custom timer includes that small allocation and lookup indexing.
Native runs as a separate block on the same GPU after the interleaved student
comparison. It has its existing short warmup and the five declared excluded
longest-request repetitions. These implementation/order differences are
disclosed rather than treated as perfectly simultaneous measurements.

The quality gate passed: 20 >=19 and no fewer passes than either control.
Prototype complete-request speedup was **5.85297×**. Against native Q4,
complete-request speedup was **0.95004×**, and throughput ratio **0.96973×**.
Both native 2× endpoints failed. The registry therefore remains
`confirmed: false` with the declared partial-success classification.

## Work, memory and limits

For one development eight-fragment request, lookup used **81 verification
calls** rather than 257 single-token decode calls. It proposed 584 candidates
and accepted 176. Because nonempty proposals execute nine positions, this
implies **at least 665 verified input positions**, versus 257 for ordinary
greedy decode. Rejected/padded arithmetic increases; batching reduces launch
and repeated weight-read costs. No FLOP counter or energy measurement was run.
Faster wall time must not be described as less arithmetic or proven battery
savings.

Summed median time to first token was 10.104 seconds for the optimized student
and 12.241 for native Q4. Prefill is already faster in this comparison; native
Q4 retains better aggregate decode performance. The student's FP16 weight
payload remains **4,135,409,664 bytes**, versus **2,011,539,712 bytes** for native
Q4. This follow-up does not further shrink either model. Final student peak
Torch allocation was **6,275,624,960 bytes**, including static KV buffers;
graph setup took **0.776 seconds** and was excluded from steady-state timing.
Native load increased NVML residency by 3,256,877,056 bytes; its allocation
estimator differs, so these are not a matched allocator-memory benchmark.

No production summaries, independent judge, phone hardware, two-model
co-tenancy, below-192-token bypass path or general-language benchmark was tested.
The twenty cases are a narrow synthetic final-decision proxy; an observed
20/20 score does not guarantee >=95% application accuracy. The architecture
retains full attention and all twenty-eight layers, and this study does not
establish a MoBA attention speedup.

The next compression experiment should test hardware-supported quantization
of the factors and the large output projection, with recovery and the same
source-quality gates. Native support for factorized MLPs is a prerequisite to
claiming a portable quantized deployment. Those operators, lower ranks and
longer lookup proposals were not executed here.

## Artifacts and reproduction

All raw records are below `data/experiments/llama_tensor_runtime/`:

- `20261014_pilot_v1/`: numerical/cache controls and source snapshots.
- `20261014_development_v1/`: 96 cells and producer fingerprints.
- `20261014_selection_v1.json`: quality-gated runtime choice.
- `20261014_fixtures_v1.json`: fresh immutable inputs.
- `20261014_final_campaign_v1/`: serial coordinator, commands and logs.
- `20261014_heldout_v1/`: 200 student cells.
- `20261014_native_v1/`: native raw cells, including five excluded warmups.
- `20261014_development_reviews_v1.json` and `20261014_heldout_reviews_v1.json`:
  all 92 arm/request source decisions.
- `20261014_registered_v1/`: aggregate, preserved analysis source, registry
  readback and completion receipt.

The worker is `experiments/structure_net/llama_tensor_runtime.py`, with
independent audit/finalization under `experiments/neural_architecture_lab/`.
Use fresh output directories; completed evidence is append-only:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 \
LD_LIBRARY_PATH='' data/experiments/moba_paper/environment/bin/python \
  -m experiments.structure_net.llama_tensor_runtime run \
  --fixtures data/experiments/llama_tensor_runtime/20261014_fixtures_v1.json \
  --split heldout --arms reference,graph_lookup8 --repeats 5 \
  --output data/experiments/llama_tensor_runtime/NEW_REPRODUCTION_ROOT
```
