# Full-model functional-coordinate and polynomial-core experiment

Status: completed exploratory campaign. Study: `JSPACE-FULL-COMPACT-01`.
`confirmed=false`. Latest root: `data/experiments/jspace_full_compact/20261113_affine_v2`.
Aggregate SHA-256: `cd25355eaf4b1a75ba7a769d98695bc5563aa5b94e09d6d53a9c5f761bb93fff`.

The corrected full model does not preserve native task accuracy at this budget. The full 36-layer pass was executed and adapted, including a
correction to an affine-value omission in the first implementation. All versions
remain recorded. The MLP-only branch is a separate, untrained follow-up selected
for measurement from development diagnostics; it is not the fully compressed
attention model and is not an independent confirmation.

## Measured results

| Model | Evaluation accuracy | Resident tensor GB | Graph tok/s |
| --- | ---: | ---: | ---: |
| Native | 210/256 (82.03%) | 6.1719 | 41.02 |
| Full-coordinate rotation | 209/256 (81.64%) | 6.1801 | — |
| v1 initialization; missing value mean | 89/256 (34.77%) | 5.5944 | — |
| v1 adapted | 103/256 (40.23%) | 5.5944 | 36.35 |
| Corrected v2 initialization | 93/256 (36.33%) | 5.5967 | — |
| Corrected v2 adapted | 88/256 (34.38%) | 5.5967 | 36.95 |
| All 36 MLPs compressed; native attention | 205/256 (80.08%) | 5.7501 | 41.01 |

GB is decimal. Throughput uses the same eight pilot prompts, batch one and 32
greedy generated tokens, including prefill. Both native and compressed models use
the same CUDA-graph runtime. Corrected v2's matched speed ratio is
**0.9007×**, and its tensor-byte reduction is
**9.32%**. The MLP-only speed ratio is
1.0028× against its contemporaneous native control;
that small difference is not a demonstrated speed improvement.
The MLP-only model gains 10 answers
and loses 15; the five-answer net
drop does not mean only five predictions changed.

Corrected v2's ordinary HF generation measured
6.51 tok/s versus
11.83 native.
Graph capture takes 0.71 seconds and
is amortized. The fast-path prefill scores
88/256 (34.38%). Fixed-length timing
does not establish free-generation or summarization quality.

| Task | Native | Corrected v2 adapted | MLP-only |
| --- | ---: | ---: | ---: |
| arc | 54/64 | 16/64 | 53/64 |
| boolq | 57/64 | 37/64 | 56/64 |
| hellaswag | 57/64 | 8/64 | 54/64 |
| winogrande | 42/64 | 27/64 | 42/64 |

Corrected progressive checkpoints: 12 layers: 102/256 (39.84%); 24 layers: 98/256 (38.28%); 36 layers: 88/256 (34.38%). These measurements did not
select a checkpoint or change the training schedule. The paired final comparison
has 11 gained answers and
133 lost answers; descriptive exact McNemar
two-sided p=9.127e-28 on this
reused evaluation pool.

## What was applied

The pinned Qwen2.5-3B-Instruct checkpoint has 36 layers and 3,085,938,688 native
parameters. A common orthogonal task-Jacobian/Ritz basis carries 1,024 functional
and 1,024 retained residual coordinates throughout the body. The Gramian uses
outer products of the correct-option-minus-mean-other-option margin gradient at
16 token positions per calibration case. This scalar task contrast is narrower
than the supplied recipe's full downstream-logit Jacobian; it does not measure
full-vocabulary generation sensitivity. RMS gains and
coordinate maps are folded into linear weights; the RMS denominator retains the
original 2,048-wide residual energy. Entry/exit rotations preserve the tied
vocabulary matrix. The complete-coordinate control is algebraically exact in
real arithmetic; its measured BF16 rounding effect is shown above.

Each attention block keeps one native GQA group and compiles the other eight
queries plus their shared K/V head. That group reads 1,024 dynamic input
coordinates with a mean anchor for the other coordinates, uses 48 RoPE frequency
pairs (96 coordinates), and retains the original 1/sqrt(128) score scale. All
causal token positions remain. Cache padding means this version saves no KV-cache
bytes. Corrected v2 restores the omitted value-mean output and packs both groups
into one attention call and one output projection. That fused call pads the
compact Q/K/V vectors back to 128 lanes, so it also forgoes the head-dimension
saving inside the attention matmuls; the incoming and outgoing projections still
use the compiled coefficients. This is an explicit runtime tradeoff, not a
demonstration of a fully 96-wide fused attention kernel.
Keeping the other native GQA group does not reconstruct the variable directions
discarded from the compiled group; those directions are mean-anchored here.

Each MLP removes 1,024 nonanchor nonlinear channels and compiles their contribution
through 153 degree-two normal-form features of 16 Jacobian-selected anchors.
The remaining 9,984 native nonlinear channels are retained. The monic graph
generators `y_i-p_i(z)` admit the relatively-prime-leading-term Gröbner criterion;
representative exact CAS checks accompany every layer. These certificates cover
the initialized rational surrogate. Relation defects in the neural activations
are measured approximations; adaptation learns the output coefficients. The
fixed 1,024-channel removal budget is separate from the cross-task eligibility
threshold: this run does not establish reliable coverage of every removed
channel. Retaining native channels is also not exact correction of every
discarded function.

Every layer receives 96 batch-one AdamW updates at 1e-4, for 3,456 updates per
variant. Only the current compact coefficients are trained. Its student prefix
is streamed from disk; future native layers remain frozen for continuation. Loss
is `0.7*T²*KL + 0.3*CE + 0.1*local_relative_MSE`, T=2, at three prompt positions
and the final option letter. This is a small adaptation experiment, not a large
language-model retraining campaign. The weighted local term dominates the
functional term on 63.0%
of corrected-v2 updates; this is a training concern, not proof of its causal role.

## Data, reuse and interpretation

Use 512 training, 64 development and 256 evaluation questions across pinned ARC,
BoolQ, HellaSwag and WinoGrande files, limited to 384 prompt tokens. Splits are
disjoint. The evaluation pool is reused from the earlier pilot. Calibration uses
32 training examples per task and 16 token positions each. Teacher capture has
78,115 prompt tokens across train and development and stores all 37 boundaries
as exact BF16 bits. No external teacher calls were used.

The common 1,024-coordinate core retains 92.01%
of pooled Jacobian trace energy. Reuse at per-task reconstruction NRMSE <=0.5 is
104,768 task uses, 51,325 unique layer-specific elements,
and 53,443 repeated uses (51.01%).
13,657 elements qualify on all four tasks. Anchor definitions are
excluded. These are common-fit eligibility counts, not identical independently
fitted coefficients or identified semantic CAVs. Layer one contributes much of
the reuse; the aggregate is not directly comparable to the earlier three-layer
pilot's percentage.

A CPU diagnostic on 128 calibration final-token states finds that the compact
input branch replaces 31.5%–
48.9% of squared deviation
from its anchor by the mean. High Jacobian trace retention therefore accompanies
a large finite input change. The diagnostic's pooled linearized quadratic-form
ratio is not a bound on the nonlinear model's finite error. The algebraic folding
control and the MLP-only result support continuing the retained-residual approach;
this global attention cut and short adaptation budget do not establish that the
same approximation remains valid through the entire stack.

## Failure localization and corrections

These development-only diagnostics concern v1, before the value-mean correction:

| Initialized component | Development accuracy |
| --- | ---: |
| Native | 48/64 |
| mlp_only | 46/64 (71.88%) |
| attention_only | 26/64 (40.62%) |
| attention_full_input | 20/64 (31.25%) |
| attention_full_head | 18/64 (28.12%) |

The full-rank, full-head split-attention control scores
210/256 (82.03%). Thus the split
implementation works at full width; the aggressive approximations are materially
different interventions. The value PCA in v1 omitted its affine mean correction,
dropping 70.87% and 81.89% of selected value-mean energy in its first two layers.
This was an implementation omission, not evidence against the supplied technique.
Corrected v2 restores `sum_h O_h (I-SS^T) mean(v)` and repeats the complete run.
Q/K frequency and input-coordinate omissions still require empirical validation.

The first fit also stopped on a transient reservation-readiness discrepancy.
Its retry changes only bounded readiness waiting and performs no GPU work while
waiting. The original root and failure record are preserved.

## Artifacts and reproduction

Seventeen CPU tests pass. The real GPU shakedowns include backward gradients,
frozen native weights, parameter reload and graph/uncaptured token agreement.
The corrected materialized checkpoint was loaded in a fresh reserved process;
all 256 evaluation records, including logits, match exactly.

The initial serializer copied the tied vocabulary weights twice. The separate
export restores that storage alias with every tensor unchanged. The corrected
export is 5,596,836,325 bytes, versus
6,219,163,218 bytes for its preserved raw training file.
Checkpoint SHA-256: `631e71d8d6660c79a36db932102b0a96a192fc84e4a9823ee718c3dd9da30976`.

- Corrected checkpoint: `data/experiments/jspace_full_compact/20261113_affine_export_v2/compact_model_tied.pt`.
- MLP-only checkpoint: `/home/rabbit/structure_net/data/experiments/jspace_full_compact/20261113_refinement_v1/mlp_runtime/model.pt`.
- Protocol and amendments: `docs/07 - Status Reports/jspace/2026-11-13_jspace-full-compact-plan.md`,
  `2026-11-13_jspace-full-compact-reservation-amendment.md`, and `2026-11-13_jspace-affine-correction.md`.
- Producer snapshots, source/data hashes, loss histories and per-case results are
  retained in every dated root. These are custom PyTorch research checkpoints,
  loaded by the corresponding experiment driver.

All GPU stages use root locked `ncm` Pixi under inference-ctl leases. To create
and verify a fresh identical export without retraining, choose an unused root:

```bash
CUDA_VISIBLE_DEVICES='' pixi run --locked python -m experiments.structure_net.jspace_full_export \
  prepare --parent data/experiments/jspace_full_compact/20261113_affine_v2 \
  --root data/experiments/jspace_full_compact/20261113_affine_recheck
CUDA_VISIBLE_DEVICES='' pixi run --locked python -m experiments.structure_net.jspace_full_export \
  export --root data/experiments/jspace_full_compact/20261113_affine_recheck
inference-ctl lease run --allocation 3060=11264 --minutes 60 \
  --reason "JSPACE corrected export verification" -- \
  pixi run --locked -e ncm python -m experiments.structure_net.jspace_full_affine_v2 \
  export_reload --root data/experiments/jspace_full_compact/20261113_affine_recheck
```

Stages are append-only: use a new prepared root to repeat a completed stage.
The combined plot is `full_compact_results.png`/`.svg` in the latest root.
