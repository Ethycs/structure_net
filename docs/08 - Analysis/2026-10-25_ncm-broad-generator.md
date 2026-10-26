# Broader generator training still scores 0/128 on complete-summary DEV accuracy

Completed exploratory study `ncm-broad-generator-v1`: **0% task accuracy
(0/128)** under the fixed Terra complete-summary rubric. All 128 judgments were
valid. Both source-disjoint cohorts scored 0/64; the independent aggregate
classifies the run as `DEV_semantic_quality_failed`. Its descriptive Wilson
95% interval is 0–2.91%, not evidence of proximity to the 95% target.

Aggregate: `data/experiments/ncm_paper_regime/20261025_broad_generator_results_v1/campaign_results.json`.
SHA-256: `2c0241923d497be9d92589fe4845d3e7013f7754e0d3588cf4c2094be4b77299`.

The [fixed plan](../07%20-%20Status%20Reports/2026-10-25_ncm-broad-generator-plan.md)
trained the existing generator and its source-attention modules on all 1,376
admitted complete-summary teacher pairs. It preserved the codec and word/source
attention exactly. This returned to the earlier generator parent, rather than
using the decoder-only adaptation that degraded reserved reconstruction.
It was one seed, without a compute-matched baseline: added data and added
updates cannot be separated causally. It did not use a discriminator or a
contrastive loss.

| Complete-summary endpoint | Valid judgments | Passing summaries | Accuracy |
| --- | ---: | ---: | ---: |
| Short-source DEV | 64/64 | 0/64 | 0% |
| Long-source DEV | 64/64 | 0/64 | 0% |
| Combined DEV | 128/128 | 0/128 | **0%** |

The judge was `codex_sdk/gpt-5.6-terra`. A pass requires the whole summary to be
faithful, salient and readable within the 128 stored-token limit. Every output
remained in the denominator. There were no empty outputs or token-cap failures;
the judge marked every output deficient on all three quality components.
Sol was not admitted because neither cohort reached the fixed 61/64 Terra
gate. No joint-judge accuracy is claimed. The sealed final 320 cases remain
unopened. This is development accuracy, not final-test accuracy.

All predictions were generated from source features alone, without target
fields, postprocessing repairs or output selection. Exact packets and raw
judgments are retained. The grade reused the four valid pilot first judgments,
added 124 new ones, and preserved all 3,795 historical first judgments; the
resulting cache contains 3,923. Legacy helper stdout used oracle-code metric
labels, but the final records and independently replayed packet bindings
correctly identify these as source-only predictions.

Training completed 4,096 optimizer updates and 16,384 example presentations,
11–12 per example. Half the presentations used the fixed 25%-to-5% code-history
corruption schedule, with clean target labels. An independent audit replayed
sampling, corruption, all updates, checkpoint/optimizer state, raw diagnostics,
and the identity of all 145 protected tensors. The clean reconstruction path
was preserved; this does not establish adequate generated summaries.

| TRAIN diagnostic | Measured result |
| --- | ---: |
| Correct code agreement with clean earlier codes | 793,677/882,016 (89.98%) |
| Correct code agreement with generated earlier codes | 353,222/882,016 (40.05%) |
| Generated active-token agreement | 19,091/96,910 (19.70%) |
| Mean equal-level teacher-forced cross-entropy | 0.5703 |

These are training diagnostics, not task accuracy. Active-token counts include
the native boundary tokens and exclude padding. On the same 16 fixed controls,
active-token agreement rose from 61/1,133 (5.38%) to 217/1,133 (19.15%); shuffled
source controls scored 85/1,133 (7.50%). This shows a training effect and some
source dependence, while the semantic endpoint still fails completely.

The saved per-level counts localize the training mismatch (levels below are
numbered starting at 1):

| Level | Agreement with clean histories | Agreement during rollout |
| --- | ---: | ---: |
| 1 | 100.00% | 100.00% |
| 2 | 99.58% | 99.58% |
| 3 | 79.32% | 79.12% |
| 4 | 78.17% | 66.04% |
| 8 | 93.52% | 42.09% |
| 12 | 95.85% | 36.04% |

The first level is not a constant-code shortcut: its TRAIN targets use 635
distinct codes. Error accumulation after early levels is a plausible mechanism,
but these measurements do not prove that correcting it alone will solve unseen
summarization. Generalization and factual decoding remain separate concerns.

An independently specified [residual-recoding diagnostic](../07%20-%20Status%20Reports/2026-10-25_ncm-residual-recoding-probe-plan.md)
tests whether the frozen later codebooks can correct an imperfect generated
prefix by recomputing residual targets. That diagnostic deliberately uses
TRAIN reference summaries and cannot count as source-only task accuracy.
Its results are separate from this completed failed study.

The run used locked Pixi ncm, physical GPU1 RTX2060, FP32 weights, FP16 training,
FP32 evaluation, TF32 disabled, and CPU FP32 AdamW. Training took 7,923.78 seconds
(132.1 minutes), excluding final diagnostics. Peak allocated GPU memory was
2,204,956,160 bytes. Lossless full FP32 source features streamed through a
256 MiB LRU, peaking at 268,431,360 resident bytes. No source or target clipping
was used. These resource measurements do not establish an inference speedup.

Checkpoint SHA-256:
`74cdce0eb14bb99c801b6ec0a025c092f95249663becc05561af8dfbad6040ff`.
Full model tensor hash:
`dbafe920914779d8099b6d5f56464f1ec41dd6a70a53d2022135cc1fd2968ac9`.

Focused training/parity checks, semantic contracts and finalizer gate tests
passed before their respective stages. CUDA lifecycle checks verified actual
updates, strict reload, protected tensors, packed/explicit gradient parity and
repeatable inference. The finalizer independently replayed all 128 output and
source bindings, exact judgment packets, first judgments, denominators and
admission decisions, and exited successfully.

The judge pilot's original terminal response was lost across context recovery;
its process exit remains explicitly unknown. Its completed artifacts were
independently replayed offline by
`experiments/structure_net/audit_ncm_broad_generator_judge_pilot.py`, which exited
0, made no judge calls and preserved all first judgments. The recovery receipt
states that distinction. The full grade and finalizer each have directly
observed successful process receipts. This lifecycle recovery did not rerun
or replace any judgment.

Reproduction entry points are the completed `ncm_broad_generator` stages,
`audit_ncm_broad_generator`, `ncm_broad_generator_development`, and
`experiments.neural_architecture_lab.finalize_ncm_broad_generator`. Completed
roots are append-only evidence; a new run requires new roots and a new producing
version. The final goal remains unmet, with no new speed claim.
