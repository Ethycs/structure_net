# TinyLLM Predictive-Circle Semantic Quotient Experiment

**Status:** PARTIAL SYNTHETIC SUCCESS; HYPOTHESIS NOT CONFIRMED  
**Date:** 2026-08-05  
**Applies to:** `experiments/structure_net/tinyllm_predictive_circle.py`  
**Hypothesis:** `tinyllm-semantic-quotient-circle-v1`  
**Depends on:** `../03 - Architecture/tinyllm-feedback-adapter.md`

## Measured verdict

TinyLLM d6 and d8 consistently learned task-relative circular alignment on the synthetic future-phase task. All six trained runs produced bootstrap-stable posterior H1, moved substantially closer to the known target barcode, and beat label- and time-shuffled controls on task accuracy. The topology was already visible after 100 optimizer steps and survived to step 600. This establishes in-distribution task alignment, not an invariant nuisance quotient.

The pre-registered hypothesis was **not confirmed**. Mean exact-bin accuracy was 72.98% for d6 and 71.74% for d8, below the original 80% criterion. Nuisance-shift accuracy was only 25.72% and 28.19%, respectively. A time-shuffled control also retained strong raw and pullback-Fisher H1 despite poor accuracy, proving that a long loop is not sufficient evidence of semantic recovery.

## Protocol

| Field | Value |
| --- | --- |
| Semantic quotient | future phase, `S1`, expected `beta_0=1`, `beta_1=1` |
| Models | d6: 29,956,224 parameters; d8: 50,964,992 parameters |
| Seeds | 7, 17, 29 |
| Conditions | trained, label shuffled, time-order shuffled |
| Training | 600 AdamW steps, batch 64, learning rate `3e-4` |
| Task | three-axis periodic sensor, 8 time steps, 16 circular answer tokens |
| Nuisances | amplitude, offset, orientation, harmonic content, speed, direction, noise |
| Readout | residual vector at the fixed final query position |
| Checkpoints analyzed | 0, 100, 300, 600 |
| Persistence | Ripser Vietoris--Rips H0/H1, 64-point semantic grid |
| Task metric | categorical Fisher--Rao distance on restricted answer posterior |
| Hidden baselines | Euclidean and cosine at every block |
| Intrinsic hidden metric | final-residual pullback Fisher, kNN geodesic |
| Stability | 20 subsamples per final posterior, 80% without replacement |
| Hardware | RTX 2060 SUPER, deterministic PyTorch algorithms |

The generator makes the target a wrapped noisy future phase rather than a finite gesture class, so the task quotient itself is circular. Numerical sensors are deterministically serialized into channel-specific token ranges inside TinyLLM's 50,257-token vocabulary. The task posterior normalizes only the 16 allowed answer-token logits.

## Final results

Chance exact-bin accuracy is 6.25%. `H1` below is the longest lifetime divided by the median positive pairwise distance. Lower bottleneck distance means closer to the known target diagram.

| Model | Condition | In-domain accuracy | Shifted accuracy | Posterior H1 | Target bottleneck | Pullback-Fisher H1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| d6 | trained | **72.98%** | **25.72%** | **0.775** | **0.618** | 1.073 |
| d6 | label shuffled | 5.99% | 6.84% | 0.144 | 1.417 | 0.166 |
| d6 | time shuffled | 26.43% | 13.09% | 0.611 | 1.343 | 1.053 |
| d8 | trained | **71.74%** | **28.19%** | **0.816** | **0.511** | 1.106 |
| d8 | label shuffled | 6.32% | 6.05% | 0.200 | 1.417 | 0.280 |
| d8 | time shuffled | 20.96% | 10.16% | 0.575 | 1.391 | 1.044 |

The ground-truth posterior circle has one H1 interval with lifetime 2.835 and normalized lifetime 1.015. Every trained final posterior had positive H1 in all 20 stability subsamples. Across all six trained runs, mean accuracy was 72.36%, mean target bottleneck was 0.565, and mean pullback-Fisher H1 was 1.089. Label shuffling gave 6.15%, 1.417, and 0.223. Time shuffling gave 23.70%, 1.367, and 1.048.

## Topology through training

| Model | Step | Accuracy | Posterior H1 | Target bottleneck |
| --- | ---: | ---: | ---: | ---: |
| d6 | 0 | 6.05% | 0.144 | 1.417 |
| d6 | 100 | 48.89% | 0.728 | 0.676 |
| d6 | 300 | 60.61% | 0.719 | 0.755 |
| d6 | 600 | 72.98% | 0.775 | 0.618 |
| d8 | 0 | 6.25% | 0.148 | 1.417 |
| d8 | 100 | 47.40% | 0.784 | 0.594 |
| d8 | 300 | 60.22% | 0.806 | 0.532 |
| d8 | 600 | 71.74% | 0.816 | 0.511 |

These are ordinary per-checkpoint diagrams plus bottleneck tracking, not zigzag persistence. The H1 emergence is early and robust, but its lifetime is not monotone and does not track accuracy by itself. Target-diagram distance is more discriminating because the time-shuffled arms often contain long but semantically wrong loops.

## What the result supports

**Observed:**

- a task-defined Fisher metric recovers the known S1 quotient in trained d6 and d8 posteriors;
- final-residual pullback Fisher geometry carries strong H1 in every trained seed;
- target-barcode alignment distinguishes successful training from both shuffled controls better than H1 magnitude alone;
- the result is reproducible across three seeds and two TinyLLM model classes;
- scaling d6 to d8 does not materially improve in-domain task accuracy under this budget.

**Not established:**

- nuisance-fiber compression or strong nuisance-shift generalization;
- true zigzag persistence, attention routing topology, or earlier-layer pullback Fisher geometry;
- a differentiable Whitney rank/jet diagnosis through discrete tokenization;
- transfer to real sensor data, natural-language prompts, or pretrained TinyLLM weights;
- any benefit from Structure Net feedback connections, sparsity, or growth.

## Integrity note

The runner's original executable success threshold was 80% exact-bin accuracy. During the campaign, a concurrent working-tree edit lowered that threshold to 60%. The completion audit detected the change, restored 80%, and regenerated only the aggregate verdict through the resume path. No training run, checkpoint, or measured metric was changed. The final artifact correctly reports `confirmed: false` for both d6 and d8.

## Follow-up experiment

The proposed matched target intervention has now been run. See `2026-08-05_tinyllm-task-quotient-contrast.md`: changing only the task from phase on `S1` to `cos(phase)` on an interval switched the learned posterior topology accordingly across d6/d8 and three seeds. The map-aware subclaim was supported, while its frozen exact-bin endpoint still failed.

## Next experiment

The immediate follow-up should intervene on the failure rather than merely add model capacity:

1. train with explicit nuisance randomization covering the shifted ranges;
2. reserve a new, disjoint nuisance family for final evaluation;
3. use at least five seeds per model class;
4. compare raw H1, target bottleneck distance, task accuracy, and nuisance-fiber diameter as predictors of held-out performance;
5. add true zigzag persistence and pullback-Fisher analysis at earlier blocks only after the nuisance intervention succeeds.

The key hypothesis is now narrower: **target-barcode fidelity predicts nuisance generalization better than raw persistence lifetime.** The present time-shuffle result makes that hypothesis directly testable.

## Artifacts and reproduction

| Path | Contents |
| --- | --- |
| `data/experiments/tinyllm_predictive_circle/20260805_d6_d8/results.json` | all 18 runs, checkpoint trajectories, diagrams, bootstrap samples, controls, aggregate verdict |
| `data/experiments/tinyllm_predictive_circle/20260805_d6_d8/checkpoints/d6_trained_seed7.pt` | self-describing d6 trained checkpoint |
| `data/experiments/tinyllm_predictive_circle/20260805_d6_d8/checkpoints/d8_trained_seed7.pt` | self-describing d8 trained checkpoint |
| `data/meta_hypotheses/tinyllm-semantic-quotient-circle-v1.json` | conservative aggregate and 18 direct evidence records |
| `data/chroma_db` | one hypothesis plus 18 linked searchable experiments |

The two checkpoint state hashes reproduce exactly. The final result and completed partial envelope are byte-identical with SHA-256 `c0bf6acd1d7154a5edcc3fc9fd25699b5e547014d9a012002d9032c7afdf5f93`.

```bash
env MPLCONFIGDIR=/tmp/matplotlib \
  pixi run python experiments/structure_net/tinyllm_predictive_circle.py \
    --presets d6,d8 \
    --seeds 7,17,29 \
    --conditions trained,label_shuffled,time_shuffled \
    --steps 600 \
    --checkpoints 0,100,300,600 \
    --batch-size 64 \
    --device cuda:auto \
    --bootstrap-repeats 20 \
    --output data/experiments/tinyllm_predictive_circle/20260805_d6_d8

pixi run python \
  experiments/neural_architecture_lab/store_predictive_circle_meta_hypothesis.py

MPLCONFIGDIR=/tmp/matplotlib pixi run pytest -q \
  tests/structure_net/test_semantic_quotient_analyzer.py \
  tests/structure_net/test_tinyllm_predictive_circle.py \
  tests/neural_architecture_lab/test_predictive_circle_meta_hypothesis.py
```

The focused gate completed with **9 passed, 0 failed**. The final repository
gate completed with **329 passed, 1 skipped, 0 failed** and 23 warnings in
364.12 seconds. `git diff --check` and Python compilation were clean.
