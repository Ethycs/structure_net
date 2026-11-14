# Shared task basis to compact, dense execution

Date: 2026-11-13. Study: **JSPACE-BENCH-TENSOR-01**. Completed exploratory
benchmark and implementation study on Qwen2.5-3B-Instruct. One checkpoint,
four task families, three modified MLP blocks, retained nonlinear residual.
The latest runtime is classified `cuda_graph_execution_measurement` and the
stacking stage `dense_stacked_core_engineering_measurement`;
all records retain `confirmed=false` because these are exploratory measurements.

**The compact polynomial core can run at approximately the latency of a dense
MLP.** Packing reduced affected-layer one-token latency from about 1.04 ms to
0.413 ms, approximately **2.52× faster** than the preceding implementation.
Combined task accuracy remains **209/256 (81.64%)**, versus original
**210/256 (82.03%)**. This is the small approximation loss the user accepted;
optimization did not reduce the overall correct count further.

**The final packed model with CUDA-graph decoding reaches 41.46 tokens/s versus
12.41 tokens/s through standard generation on the same model: 3.34× faster.**
Original dynamic prefill is preserved; all 256 task predictions and choice logits
match the packed model before graph optimization. The large whole-model gain is
from reducing execution overhead, combined with the densely packed polynomial
core. It is not evidence that basis compression alone supplies a 3.34× gain.

Latest aggregate: `data/experiments/jspace_bench_tensor/20261113_graph_v3/campaign_results.json`.
SHA-256: `6ca5d8ab006bc18f64db441811fca7f8e36d0b993b7f0019e6a6dedc3879ccff`.

The basis itself is unchanged during optimization. We removed the gate/up
computation of the same 73 channels, preserved the uncovered nonlinear
computation, and changed how the polynomial core and residual are executed.
Early deployment thresholds remain in the historical results; they do not decide
whether this research direction should continue.

## Latest combined-model measurements

The final generation comparison keeps the model, coefficients, precision, eager
attention, repetition penalty (1.05), EOS suppression and eight prompts fixed:

| Generation path | Tokens/s | Task accuracy |
| --- | ---: | ---: |
| Standard Hugging Face generation, packed model | 12.415 | 209/256 |
| Static decode without graph replay; original dynamic prefill | 12.900 | 209/256 |
| **CUDA graph; original dynamic prefill** | **41.463** | **209/256** |

Timing includes prefill, cache resets and copying its exact BF16 KV values into
the static decode cache. One-time graph setup/capture takes **2.42 seconds** and
is amortized across requests. The captured capacity is 416 positions; the test
generates 32 tokens per request. This is a reusable bounded-shape prototype, not
a benchmark of arbitrary long contexts. No equally graphed original-model speed
comparison was performed; the runtime improvement can also benefit dense models.

Graph replay and the identical uncaptured static algorithm match all 256 generated
tokens across the eight timing prompts. Against the dynamic-cache HF path,
134/256 generated token positions match; later autoregressive strings diverge
under the different attention reduction shape. Thus this is **unchanged measured
multiple-choice accuracy**, not an exact-output or long-generation semantic
equivalence claim. The first-answer choice logits are identical (choice KL 0).
Generated sequences are retained in the runtime result for inspection.

The previous static-prefill version reached 41.10 tok/s but scored 206/256. It
remains preserved as `20261113_graph_v2`. Restoring original dynamic prefill and
copying its KV state recovered all three answers in `graph_v3`; no weights,
polynomial coefficients or covered channels changed. The initially prepared
`graph_v1` never ran, because checkpoint inspection identified the need to match
the configured repetition penalty before launch.

The earlier stacking-only comparison follows.

All rows below use the same checkpoint, BF16 precision, eager attention, task
prompts, 256 held-out questions and RTX 3060 reservation. Generation timing is
batch one, eight development prompts, exactly 32 generated tokens each, including
prefill and excluding an initial warmup. It is not decode-only throughput.

| Implementation | Correct / 256 | Accuracy | Initial run tok/s | Resident tensor bytes |
| --- | ---: | ---: | ---: | ---: |
| Original | 210 | 82.03% | 12.06, mean of before/after | 6,171,877,632 |
| Previous separate polynomial branch + residual | 209 | 81.64% | 11.49 | 6,171,655,756 |
| Stacked dense original | 210 | 82.03% | 11.95 | 6,171,879,216 |
| **Stacked compact core + residual** | **209** | **81.64%** | **12.34** | **6,171,408,168** |

The original timing bracket was 12.1165 → 11.9940 tok/s. The initial packed
measurement is 7.40% faster than the previous polynomial implementation and
2.34% above the mean original rate. A matched stacked dense control separates
generic gate/up fusion from the incremental polynomial-core cost.

The counterbalanced repeat measured each implementation twice, in opposite
orders, for 512 generated tokens per implementation. It changes no weights,
accuracy, basis, kernel or prompts:

| Implementation | Two run rates (tok/s) | Pooled tok/s |
| --- | --- | ---: |
| Previous polynomial branch | 11.829, 11.638 | 11.733 |
| Stacked dense original | 11.522, 11.521 | 11.521 |
| Stacked compact core | 11.681, 11.987 | 11.832 |

The repeat gives **0.85%** gain over the previous polynomial implementation and
2.70% over the stacked dense control. Thus the initial 7.40% whole-model gain is
not a stable speedup estimate. The isolated-layer improvement below is clear;
the complete model is still dominated by other work. This motivated a separate
CUDA-graph execution extension with the same packed model.

| MLP block | Original, 1 token (µs) | Previous polynomial (µs) | Stacked dense (µs) | Stacked core (µs) |
| --- | ---: | ---: | ---: | ---: |
| 8 | 418.64 | 1,050.40 | 411.50 | 413.16 |
| 17 | 418.33 | 1,033.55 | 411.11 | 412.03 |
| 26 | 418.65 | 1,040.09 | 411.28 | 412.74 |

These CUDA-event measurements use ten warmups and 100 repetitions per cell. At
128 tokens, the packed core takes 577.6–579.7 µs versus 1,075.9–1,175.7 µs for the
previous polynomial branch, and 570.2–575.4 µs for stacked dense. The strongest
timing conclusion is that packing removes the large implementation overhead of
the small polynomial branch. A large whole-model compression advantage is a
different measurement from this successful implementation improvement.

Per-family correct counts, each out of 64:

| Implementation | ARC | BoolQ | HellaSwag | WinoGrande |
| --- | ---: | ---: | ---: | ---: |
| Original / stacked dense original | 54 | 57 | 57 | 42 |
| Previous polynomial + residual | 53 | 57 | 57 | 42 |
| Stacked compact core + residual | 54 | 57 | 56 | 42 |

The packed model agrees with the original on 255/256 labels. Although the total
accuracy is unchanged from the previous polynomial implementation, the one lost
answer changes family. Packing rounds reconstructed channels to BF16 before the
combined down projection; the previous implementation used a separate FP32
correction. This numerical change is measured, not described as bitwise equality
of the complete models. Stacked dense agrees with the original on all 256 labels.

## What was stacked

Each affected layer computes

`h_stack = [retained nonlinear channels, polynomial reconstruction p(z)]`

`output = [W_retained, W_covered] @ h_stack`.

The compact coefficient block for `p(z)` remains factored: 20, 34 or 19 covered
channels, each using the same 153-term degree-two feature library in 16 anchor
coordinates. No output-by-136 quadratic matrix is materialized. Shared relations
are stored once per layer, independently of how many tasks reuse them.

The implementation has one stacked gate/up GEMM, one Triton kernel that performs
SwiGLU and writes the packed residual/core activation, and one dense down GEMM.
The nonlinear channels and polynomial outputs occupy contiguous blocks. Layer
dependencies remain sequential. This is a dense execution layout, not a claim
that different transformer layers can be evaluated simultaneously.

Retained gate/up widths are padded to multiples of 16. The three padded widths
are 10,992, 10,976 and 10,992, compared with original 11,008. Thus 73 nonlinear
functions are removed, with nine zero padding rows added for alignment: 64 net
rows are saved in each of the gate and up projections. The down projection
retains its original width and receives reconstructed values in the covered
positions. No deleted gate/up functions are secretly recomputed.

The stacked model saves **469,464 resident tensor bytes** relative to the
original, versus 221,876 bytes for the preceding factored implementation. This
is a small 0.0076% whole-model saving because only 73 of 33,024 channels across
the three selected blocks are covered. The latest audited coefficient/weight
count, including derived buffers but excluding normalization/index buffers, is
3,085,687,713. A 405,336,690-byte replacement-module checkpoint is a delta; it
requires the pinned original checkpoint and fit metadata. It is not a standalone
405 MB language model.

## Cross-task redundancy: the requested research metric

A qualifying occurrence means that the **same polynomial generator in the same
layer chart** achieves development reconstruction NRMSE ≤0.5 for that task.
Anchor identities are excluded. This is approximate neural reuse of an identical
algebraic expression; it is not an exact identity of the pretrained network.

| Block | Qualifying task uses | Distinct generators | Repeated uses | Repeat fraction | Shared by ≥2 tasks | Shared by all 4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 694 | 555 | 139 | 20.03% | 85 | 20 |
| 17 | 1,872 | 1,556 | 316 | 16.88% | 207 | 34 |
| 26 | 1,049 | 825 | 224 | 21.35% | 153 | 19 |
| **Total** | **3,615** | **2,936** | **679** | **18.78%** | **445** | **73** |

The 2,936 distinct generators split into 2,491 qualifying on exactly one task,
284 on two, 88 on three and 73 on four. The compressed model uses the 73 that
qualify on all four. We have not established that the other generators are
unusable: task-conditional coverage, broader charts and residual correction are
separate opportunities. The all-task subset was an initial conservative slice,
not a limit on the idea.

![Basis reuse counts](../../../data/experiments/jspace_bench_tensor/20261113_v2/basis_redundancy.png)

Separate per-task fits in the same chart yielded no exact rational coefficient
matches to the common fit. That does not mean they have no approximately shared
structure, nor does it establish absence of pairwise equality among independent
task fits; that latter comparison was not performed. Counts are summed within
layers, with no assumption that a generator in block 8 equals one in block 17.
Redundancy counts are neither model accuracy nor independent mechanism counts.

## Mathematical construction and what it establishes

The pooled task-Jacobian bank contains final-token and sequence-mean gradients
of the correct-option margin for 16 calibration questions per task. At each
selected block, its leading 16 directions guide selection of 16 physical anchor
channels. SVD represents this J-space; it does not factorize the model weights.
Degree-two regression in the common anchor chart gives graph relations

`g_i = y_i - p_i(z)`.

Coefficients lie on the declared rational grid with denominator 2^20. Ordering
eliminated variables before anchors makes the distinct leading variables monic
and pairwise coprime. The product criterion gives a Gröbner basis for this graph
ideal, with representative exact SymPy normal-form checks. Substitution then
determines the coefficient tensor and its factors. The quotient is infinite
dimensional; the 153-term library is a bounded-degree slice, not a claimed
finite-dimensional full quotient.

The polynomial compiler distinguishes the exact identity

`W a = W_keep a_keep + W_covered p(z) + W_covered [a_covered-p(z)]`

from the implemented approximation, which omits the final relation-defect term
while retaining all uncovered nonlinear channels. General normal-form tests also
retain overflow and relation-defect terms explicitly. The implemented task
measurements quantify the effect of omitting the small covered-channel defect.

This is a concrete basis-to-tensor-to-model intervention with a residual, not
only a post-hoc probe. The graph construction guarantees a valid algebraic
surrogate for any fitted coefficients; the empirical questions are how well its
relations hold, how broadly tasks reuse them, and how much computation they can
replace. Semantic CAV identification, additional relations among anchors, and
cross-layer propagation are not established by these measurements.

Reference for the polynomial division interface:
[SymPy GröbnerBasis.reduce](https://docs.sympy.org/latest/modules/polys/reference.html#sympy.polys.polytools.GroebnerBasis.reduce).

## Benchmark and historical measurements

The pinned checkpoint is `Qwen/Qwen2.5-3B-Instruct`, revision
`aa8e72537993ba99e69dfaafa59ed015b17504d1`: 36 layers, hidden size 2,048 and
intermediate size 11,008. Only blocks 8, 17 and 26 undergo polynomial substitution.
Both stacked controls receive the same generic execution optimization elsewhere.

Official sources are [ARC](https://huggingface.co/datasets/allenai/ai2_arc),
[BoolQ](https://huggingface.co/datasets/google/boolq),
[HellaSwag](https://huggingface.co/datasets/Rowan/hellaswag), and
[WinoGrande](https://huggingface.co/datasets/allenai/winogrande). Revisions, source
files and hashes are pinned in the task manifest. Each family contributes 32
calibration, 16 development and 64 evaluation questions. The 448 question/options
identities are disjoint. Prompts longer than 384 tokens are filtered without
truncation. The score is first-assistant-token multiple-choice accuracy, not
summarization quality, free-generation accuracy or a general 95% capability claim.

The first budgeted experiment ranked channels using reconstruction error and
task-Jacobian sensitivity. None of its 128/512/1,024-channel-per-block candidates
met the original development rule. The 128 fallback scored 211/256 on evaluation,
but missed the development KL threshold (0.021506 versus 0.02) and the original
254/256 agreement threshold (253/256). Its immutable classification remains
`combined_quality_gate_failed`; this is the outcome of that bounded configuration,
not a rejection of shared-basis research. Its linear, zeroed-channel and explicit
polynomial controls scored 210, 208 and 209 respectively.

The next strict subset selected all 73 relations supported across all four tasks.
Its development score was 53/64 versus original 52/64, KL 0.002003, and agreement
63/64. Evaluation scored 209/256, with 255/256 original-label agreement. Its
classification was `covered_quality_preserved_no_speed_gate`: the point-estimate
quality tolerance passed, though one answer was lost, and throughput was 2.51%
lower. Expanded-tensor and zeroed-channel controls both scored 210/256. These
controls do not yet demonstrate a task-accuracy advantage of polynomial recovery
over simple removal on this tiny subset; they motivate testing a larger useful
covered component rather than calling the direction exhausted.

The user explicitly accepted the small loss and requested speed and compact
stacking. The latest implementation changes respond to that request without
rewriting the earlier records or searching new coefficients on evaluation data.

## Integrity, artifacts and reproduction

All GPU stages use `inference-ctl lease run` and root locked `ncm` Pixi, with
10,240 MiB reservations on the RTX 3060. No services were stopped for this study.
The first fit attempt failed only because CUDA integer matrix multiplication was
unsupported in overlap reporting. Revision 2 moved those small count operations
to CPU and reused frozen captures; the failed log and original source remain.

Focused CPU tests cover exact normal forms and defects, quotient multiplication,
projection substitution values and gradients, redundancy accounting, the CPU
count fix, vectorized features, dense packing, and registry scope. The final
suite total is 22 passing cases across the focused runs, including the membership
form of the configured repetition penalty against HF's reference processor. A direct `pytest`
invocation initially failed to import `experiments`; rerunning with
`pixi run --locked python -m pytest` passed. This was a collection/launch issue,
not experimental evidence. The fused CUDA shakedown matched its BF16 reference
exactly on all four tested cases (one and 37 tokens, with and without the core).

Artifact roots under `data/experiments/jspace_bench_tensor/`:

| Root | Scope | Fingerprint prefix |
| --- | --- | --- |
| `20261113_v1` | Original capture and preserved failed fit | `f37230622523d640` |
| `20261113_v2` | Reporting fix, shared basis and budgeted comparison | `bf8fad5b16ed6f47` |
| `20261113_covered_v1` | Strict 73-channel core | `f3f2317393b9b962` |
| `20261113_stacked_v1` | Dense packing, accuracy and first speed comparison | `56d3df5610cd0003` |
| `20261113_stacked_timing_v1` | Counterbalanced timing only | `ed66c3d40f43840c` |
| `20261113_graph_v1` | Prepared only; generation contract corrected before launch | `e2d1cf4320dac646` |
| `20261113_graph_v2` | Static prefill and graph decoding | `0c3755afc5bf04b1` |
| `20261113_graph_v3` | Original dynamic prefill and graph decoding | `76e0ad7a49fdadf9` |

Each root preserves `plan.json`, the relevant stage manifest, frozen producer
source and raw records. Completed analyses have `result.json` and matching
`campaign_results.json`; model variants have replacement checkpoints and byte
audits. The source snapshot and all completed artifacts are hashed in the stacked
root's `finalization.json`. The local NAL registry records the historical primary
result, named strict-coverage and engineering extensions, and a fresh-process
readback. These are one campaign on one checkpoint and reused evaluation pool,
not independent confirmations.

CPU preparation, analysis, audits and registration use `CUDA_VISIBLE_DEVICES=''`.
For a fresh compatible root, the engineering entry points are
`experiments.structure_net.jspace_bench_stacked` (`prepare`, `run`, `analyze`) and
`experiments.structure_net.jspace_bench_stacked_timing` (`prepare`, `run`). The
final graph runtime is `experiments.structure_net.jspace_stacked_graph_decode_v3`
(`prepare`, `run`); its class provides reusable capture and generation. The
`run` stages must be wrapped in an active reservation, for example:

```bash
/usr/bin/python3 /usr/local/bin/inference-ctl lease run \
  --allocation 3060=10240 --minutes 30 --reason 'J-space stacked core benchmark' -- \
  pixi run --locked -e ncm python -m experiments.structure_net.jspace_bench_stacked run
```

The completed default roots are append-only. Do not rerun a producer into them.
