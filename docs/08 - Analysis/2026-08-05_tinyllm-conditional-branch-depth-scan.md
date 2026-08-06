# TinyLLM Conditional Branch-Information Depth Scan

**Status:** INTERNAL QUOTIENT SUPPORTED IN DISTRIBUTION; NUISANCE-ROBUST HYPOTHESIS NOT CONFIRMED  
**Date:** 2026-08-05  
**Applies to:** `tinyllm_conditional_branch_depth_scan.py`  
**Hypothesis:** `tinyllm-joint-depth-internal-cosine-quotient-v1`  
**Depends on:** `../03 - Architecture/depth-graded-transformer.md`

## Measured verdict

The cosine quotient is not merely imposed by the shared decoder. Across five
seeds and all three training arms, frozen full-depth cosine residuals retained
cosine while nonlinear conditional probes recovered the phase branch only at
chance. Full-depth mean branch accuracy was **51.33%** ordinary, **52.42%**
discrete multi-exit, and **50.55%** continuous-gate. The matched seed-7 phase
controls scored **99.90–100%**, validating probe sensitivity.

Joint-depth training moved the median in-distribution residual quotient front
from **0.020** ordinary to **0.005** in both joint arms. The stronger decoder-
supported median moved from **1.5** ordinary to **0.005**. This revises the old
ordinary cosine front at 1.85: that was a mature decoder/paired-geometry front,
not the onset of residual branch erasure.

The full preregistered hypothesis is nevertheless **not confirmed**. No cosine
run passed the nuisance-robust front because full-depth disjoint-family cosine
correlation was only 0.736 ordinary, 0.759 discrete, and 0.732 continuous. A
chance branch probe is not quotient evidence when the shifted representation
also loses the retained task coordinate.

## Direct depth result

| Arm | ID residual fronts by seed 7/17/29/41/53 | Median | ID decoder-supported median | Full ID branch | Full shifted branch | Full shifted cosine |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Ordinary final | .005 / .005 / .020 / .100 / .050 | **.020** | 1.500 | 51.33% | 49.98% | .736 |
| Discrete multi-exit | .005 / .005 / .005 / .010 / .005 | **.005** | .005 | 52.42% | 50.10% | **.759** |
| Continuous gate | .005 / .005 / .005 / .005 / .005 | **.005** | .005 | 50.55% | 49.57% | .732 |

Every ID residual front required:

- nonlinear balanced branch accuracy at most 55% on exact cosine pairs;
- conditional log-loss gain over `p(branch | cosine)` at most 0.02;
- residual cosine Pearson correlation at least 0.9;
- a one-component finite-fiber proxy.

Every decoder-supported front additionally required decoder cosine correlation
at least 0.9. The nuisance-robust front added the same retained-cosine and branch
conditions on the disjoint nuisance family; none passed.

## Block-1 mechanism

The corrected pre/attention/MLP localization used the ID residual front rather
than the failed shifted front.

| Arm | Query branch | Post-attention branch | Post-MLP branch | Post-attention conditional log-loss gain |
| --- | ---: | ---: | ---: | ---: |
| Ordinary final | 50.00% | **60.88%** | 51.48% | **.0541** |
| Discrete multi-exit | 50.00% | 53.69% | 51.97% | .0019 |
| Continuous gate | 50.00% | 51.89% | 51.17% | -.0008 |

Ordinary block-1 attention made the phase branch nonlinearly readable in all
five seeds (58.1–64.2%), while its MLP removed the signal. The corresponding
linear probe averaged only 52.71%, so the transient was nonlinear. Discrete
supervision attenuated it; continuous-gate supervision nearly eliminated it.
This supports a mechanistic interpretation: joint-depth supervision changes
the first block's routing/cancellation, not merely the final head.

The finite-fiber MST proxy reported one component at every listed cut, including
ordinary post-attention where the nonlinear probe detected branch information.
It therefore did not resolve the transient and cannot be promoted to Reeb-
component agreement. It remains a coarse finite-sample geometric proxy.

## Controls and topology

- Nonlinear shuffled-label controls averaged 49.96%, 50.18%, and 50.28% across
  ordinary, discrete, and continuous depth cells; their maxima were at most
  53.32%.
- Full-depth conditional log-loss gains were approximately zero in every cosine
  arm (-.0017, .0015, and -.0027).
- Full-depth posterior normalized H1 was low in every cosine arm (.0089, .0075,
  and .0087), agreeing with interval-like output geometry but remaining decoder-
  conditioned.
- Seed-7 phase controls retained full-depth branch information at 100%, 100%,
  and 99.90% ID and 95.61%, 94.63%, and 97.66% shifted.
- Phase controls already exposed branch information at depth .005, showing that
  the same probe family can recover it from extremely shallow residual changes.

## Protocol

| Field | Value |
| --- | --- |
| Model | d8 TinyLLM, 8 blocks, 50,964,992 parameters |
| Cosine seeds | 7, 17, 29, 41, 53 |
| Phase controls | seed 7, all three training arms |
| Checkpoints | 18 frozen final models |
| Training | 600 AdamW updates; original ordinary/discrete/continuous schedules |
| Main depths | 0, .005, .01, .02, .05, .1, .25, .5, 1, 1.5, 2, 3, 4, 8 |
| Ordinary refinement | 1.70, 1.75, 1.80, 1.825, 1.85, 1.875, 1.90, 1.95 |
| Probe data | 2,048 train; 512 validation; 1,024 per held-out family |
| Probes | linear and 2-layer width-128 nonlinear; 240-update cap; validation early stopping |
| Fiber check | 5 interior cosine values, 12 nuisance replicates per branch |
| Posterior topology | 64-phase Fisher–Rao H1 at every selected depth |
| Hardware | NVIDIA GeForce RTX 2060 SUPER; Torch 2.5.1+cu121 |

The 12 missing cosine checkpoints took 1,139 seconds to train; the 18 original
probe analyses took 1,887 seconds. All weights are retained. Runs were executed
sequentially on one GPU to avoid contention and preserve deterministic schedules.

## Claim boundaries

**Supported:**

- cosine full-depth residuals discard the conditional phase branch in all 15
  independently trained cosine cells;
- this is an internal residual result, not only decoder collapse;
- joint-depth training moves the median ID residual and decoder-supported fronts
  earlier than ordinary training;
- ordinary first-block attention creates a nonlinear branch transient that its
  MLP removes;
- linear/nonlinear, shuffled-label, phase-task, and shifted-branch controls rule
  out a generally powerless branch probe.

**Not supported or not established:**

- the preregistered nuisance-robust quotient claim;
- a certified conditional mutual-information value;
- a Reeb graph, Reeb cosheaf, or certified fiber component merger;
- causal absence of branch information outside the tested probe families;
- phase-control repeatability beyond seed 7;
- transfer beyond the synthetic generator.

## Artifacts and reproduction

| Path | Contents |
| --- | --- |
| `data/experiments/tinyllm_conditional_branch_depth_scan/20260805_d8_five_seed/results.json` | all 18 profiles, controls, topology, fiber records, and block-1 cuts |
| `…/checkpoints/` | 12 newly trained cosine checkpoints for seeds 17/29/41/53 |
| `data/experiments/tinyllm_depth_graded_quotient/20260805_d8_seed7/checkpoints/` | six reused seed-7 checkpoints |
| `data/meta_hypotheses/tinyllm-joint-depth-internal-cosine-quotient-v1.json` | conservative NAL aggregate and linked experiments |

```bash
pixi run python \
  experiments/structure_net/tinyllm_conditional_branch_depth_scan.py \
  --device cuda:auto \
  --seeds 7,17,29,41,53 \
  --training-steps 600 \
  --probe-steps 240 \
  --train-samples 2048 \
  --validation-samples 512 \
  --test-samples 1024 \
  --output data/experiments/tinyllm_conditional_branch_depth_scan/20260805_d8_five_seed

pixi run python \
  experiments/structure_net/tinyllm_conditional_branch_depth_scan.py \
  --device cuda:auto \
  --output data/experiments/tinyllm_conditional_branch_depth_scan/20260805_d8_five_seed \
  --refresh-existing

pixi run python \
  experiments/neural_architecture_lab/store_conditional_branch_depth_meta_hypothesis.py
```

The strict-JSON result and partial envelopes are byte-identical with SHA-256
`fd99760f7cf7f54574869688fef1e6b59656c6b3d83035d9380be5e9f1eeb966` before
NAL storage metadata is written.
