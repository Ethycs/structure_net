# TinyLLM Task-Quotient Contrast

**Status:** MAP-AWARE SUBCLAIM SUPPORTED; FULL HYPOTHESIS NOT CONFIRMED  
**Date:** 2026-08-05  
**Applies to:** `experiments/structure_net/tinyllm_task_quotient_contrast.py`  
**Hypothesis:** `tinyllm-task-topology-follows-quotient-v1`  
**Depends on:** `2026-08-05_tinyllm-semantic-quotient-circle.md`

## Measured verdict

Learned task-posterior topology followed the supervised quotient rather than merely preserving the periodic input loop. With identical periodic inputs, initialization, and minibatches, future-phase prediction produced an aligned degree-one circle while `cos(future phase)` prediction produced a highly correlated interval with almost no persistent H1. This result repeated across all three seeds in both d6 and d8.

The full preregistered hypothesis is nevertheless **not confirmed**. Six of seven criteria passed for each model class. The sole failed endpoint was mean exact-bin accuracy on the cosine-interval task: 57.23% for both d6 and d8 against a frozen 60% requirement. The continuous interval-map metric passed strongly at Pearson 0.993, but it does not replace the failed categorical endpoint.

The strongest supported statement is:

> Under this synthetic matched intervention, task-relative topological organization followed the target quotient in distribution. Robust invariant semantic-quotient recovery remains unestablished.

## Protocol

| Field | Value |
| --- | --- |
| Intervention | target only: future phase on `S1` versus `cos(future phase)` on `[-1, 1]` |
| Held matched | sensor inputs, TinyLLM initialization, minibatch indices, optimizer, and training budget |
| Models | d6: 29,956,224 parameters; d8: 50,964,992 parameters |
| Seeds | 7, 17, 29 |
| Conditions | trained and label shuffled |
| Training | 600 AdamW steps, batch 64, learning rate `3e-4` |
| Checkpoints analyzed | 0, 100, 300, 600 |
| Circle-map tests | phase alignment after rotation/orientation, ordered-grid winding degree |
| Interval-map tests | Pearson and Spearman correlation, RMSE, predicted range |
| Topology | Fisher--Rao posterior H1, target bottleneck, final-residual Euclidean/cosine, pullback Fisher |
| Fiber test | within-semantic-group / between-group distance ratio; lower is better |
| Stability | 20 posterior subsamples per final arm |
| Hardware | NVIDIA GeForce RTX 2060 SUPER; CUDA 12.1 PyTorch build |

The interval target used a Gaussian posterior over 16 centers on `[-1, 1]`. Its quotient explicitly identifies opposite phases with equal cosine. The nuisance-collapse set therefore grouped both branches of each cosine fiber, rather than merely testing repeated samples at the same phase.

## Final trained results

| Model | Quotient | Exact-bin accuracy | Map score | Posterior H1 | Posterior fiber ratio | Shifted accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| d6 | phase circle | **72.98%** | alignment **0.990**; degree **+1** | **0.775** | **0.116** | 25.72% |
| d6 | cosine interval | 57.23% | Pearson **0.993** | **0.003** | **0.255** | 19.60% |
| d8 | phase circle | **71.74%** | alignment **0.990**; degree **+1** | **0.816** | **0.113** | 28.19% |
| d8 | cosine interval | 57.23% | Pearson **0.993** | **0.011** | **0.237** | 22.53% |

Every trained circle run had alignment above 0.985 and winding degree +1. Every trained interval run had Pearson correlation above 0.988; its posterior normalized H1 ranged from 0 to 0.019. The known target geometries themselves separated as intended: the phase target had normalized H1 1.015, while the interval target had no H1 interval.

Label-shuffled accuracy averaged 6.15% across the circle arms and 8.82% across the interval arms. The circle labels are uniform, so their uniform-random top-one baseline is 6.25%. The interval labels inherit the arcsine density of `cos(phase)` and are not uniform: across its three deterministic evaluation sets, uniform random gives 6.25%, prediction sampled from the empirical class prior gives 7.60%, and the empirical majority class gives 13.02%. The shuffled interval result lies between the latter two baselines; it should not be described simply as 6.25% chance.

Shuffled posterior fiber ratios were 0.92--0.98, versus 0.11 for the trained circle and 0.24--0.25 for the trained interval. The learned posterior map therefore grouped the supervised semantic fibers in distribution; poor shifted accuracy shows that this grouping was not robustly invariant outside the training nuisance range. Final-residual Euclidean ratios also decreased, but this distance summary does not show that branch information was absent from the residual stream. Pullback-Fisher geometry is likewise decoder-conditioned rather than independent evidence of information erasure.

## Preregistered criteria

| Criterion | d6 | d8 |
| --- | --- | --- |
| Circle alignment at least 0.9 | pass | pass |
| Absolute circle degree within 0.1 of 1 | pass | pass |
| Interval Pearson correlation at least 0.9 | pass | pass |
| Interval normalized H1 at most 0.2 | pass | pass |
| Circle H1 exceeds interval H1 by at least 0.4 | pass | pass |
| Trained tasks beat label shuffling by at least 0.3 | pass | pass |
| Both trained tasks reach 60% exact-bin accuracy | **fail** | **fail** |

No criterion was changed after execution. The aggregate remains `confirmed: false`.

## Development through training

| Model | Quotient | Step | Accuracy | Posterior H1 | Map score |
| --- | --- | ---: | ---: | ---: | ---: |
| d6 | circle | 0 | 6.05% | 0.144 | 0.055 |
| d6 | circle | 100 | 48.89% | 0.728 | 0.969 |
| d6 | circle | 600 | 72.98% | 0.775 | 0.990 |
| d6 | interval | 0 | 6.18% | 0.144 | -0.004 |
| d6 | interval | 100 | 41.86% | 0.007 | 0.982 |
| d6 | interval | 600 | 57.23% | 0.003 | 0.993 |
| d8 | circle | 0 | 6.25% | 0.148 | 0.097 |
| d8 | circle | 100 | 47.40% | 0.784 | 0.968 |
| d8 | circle | 600 | 71.74% | 0.816 | 0.990 |
| d8 | interval | 0 | 5.53% | 0.148 | 0.238 |
| d8 | interval | 100 | 40.30% | 0.073 | 0.979 |
| d8 | interval | 600 | 57.23% | 0.011 | 0.993 |

By step 100, the map-aware scores were already above 0.96 in both tasks, but their topology had diverged: circle H1 grew sharply while interval H1 collapsed. This is ordinary checkpoint tracking, not zigzag persistence or proof of a topology-changing singularity.

## Why the tuple matters

No individual diagnostic is sufficient:

- label-shuffled circle runs sometimes had integer winding degrees from -5 to +2 despite map alignment below 0.27;
- an interval's empty target H1 diagram makes a small bottleneck distance easy to obtain even with a poor map;
- exact-bin accuracy understates a smooth interval decoder when probability mass lands in an adjacent bin;
- low in-distribution fiber ratio does not imply held-out nuisance invariance.

The defensible unit of evidence is the joint tuple of task accuracy, induced-map agreement, winding degree where applicable, target-relative persistence, and nuisance-fiber collapse. That tuple separates the trained tasks from their controls and prevents an accidental loop—or accidental absence of one—from being labeled semantic.

## Scope boundaries

**Supported:**

- topology follows the task quotient under a matched synthetic target intervention;
- the circle decoder is aligned with semantic phase and has degree +1;
- the interval decoder identifies the cosine quotient and has negligible H1;
- trained posterior maps collapse semantic fibers much more than shuffled controls; residual-stream information erasure remains untested here.

**Not established:**

- a circular coordinate derived independently from persistent cohomology; the tested map comes from the task decoder;
- true zigzag persistence or a Whitney rank/jet event;
- robust held-out nuisance invariance;
- transfer to real sensors, tokenizer-derived prompts, or pretrained weights;
- any benefit from feedback connections, structural growth, or sparsification.

## Artifacts and integrity

| Path | Contents |
| --- | --- |
| `data/experiments/tinyllm_task_quotient_contrast/20260805_d6_d8/results.json` | all 24 arms, trajectories, diagrams, controls, and aggregate verdict |
| `data/experiments/tinyllm_task_quotient_contrast/20260805_d6_d8/checkpoints/` | four self-describing seed-7 trained checkpoints |
| `data/meta_hypotheses/tinyllm-task-topology-follows-quotient-v1.json` | conservative meta-hypothesis and 24 linked evidence records |
| `data/chroma_db` | searchable hypothesis plus 24 linked experiment records |

The completed result and partial-resume envelope are byte-identical, with SHA-256 `129a7a47757d2ecdde7138ca7729c0052f4c4653d5afc38106949f477e87652d`. This hash includes the post-run class-prior baseline audit; no trained arm or primary metric changed. All four checkpoint state hashes reproduce exactly:

- d6 circle: `5f25c2bb659565df4a6b6e276e0c8057e16e10da299c2d5879ff92a8fb132137`;
- d6 interval: `2242dc6e746ff2c56f0014baf0b6ad4a47d2ca715659486387bb8795810d6237`;
- d8 circle: `d9b3d2fa1f76d04b9a301ce3e51a6d2b9b11dc3c61467c353e77115c6eea9722`;
- d8 interval: `bf20a98e242b72f14c186176951973970cb96f52353a6416c7b4db7b208d02fc`.

Direct Chroma inspection found the stable hypothesis ID and all 24 expected result hashes. Chroma's client emitted non-fatal telemetry and NumPy-2 consumer warnings during insertion; these warnings did not prevent persistence.

## Reproduction

```bash
env MPLCONFIGDIR=/tmp/structure-net-matplotlib \
  pixi run python experiments/structure_net/tinyllm_task_quotient_contrast.py \
    --presets d6,d8 \
    --seeds 7,17,29 \
    --quotients phase_circle,cosine_interval \
    --conditions trained,label_shuffled \
    --steps 600 \
    --checkpoints 0,100,300,600 \
    --batch-size 64 \
    --bootstrap-repeats 20 \
    --device cuda:auto \
    --output data/experiments/tinyllm_task_quotient_contrast/20260805_d6_d8

pixi run python \
  experiments/neural_architecture_lab/store_task_quotient_contrast_meta_hypothesis.py
```

The original focused experiment, analyzer, and ledger gate completed with **11 passed, 0 failed**; the expanded probe-and-baseline focused gate completed with **18 passed, 0 failed**. After the sublayer atlas extension, the final full repository gate completed with **350 passed, 1 skipped, 0 failed** and 23 warnings in 386.19 seconds.

## Follow-up result

The frozen layerwise test has now been run; see `2026-08-05_tinyllm-internal-quotient-probes.md`. Conditional branch and cross-decoding asymmetries support internal quotient formation in both retained d6/d8 seed-7 pairs, including on a disjoint nuisance family. Independent cohomology strongly recovers the phase-model circle. The result remains unconfirmed because other seed checkpoints were not retained and one d6 coordinate-alignment criterion missed by 0.003.

## Next experiment

Freeze the matched phase and cosine models, probe every residual layer for the cosine branch bit on exact cosine-matched pairs, and cross-decode cosine and phase on a disjoint nuisance family. In parallel, derive the circle coordinate from persistent cohomology rather than the supervised decoder. These tests distinguish output-map collapse from internal information erasure and separate map discovery from map verification.
