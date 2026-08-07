# TinyLLM calibration-orientation noise preregistration

**Status:** COMPLETED — REFERENCE PRECISION CRITICAL; QUOTIENT SURVIVES LONGER THAN TASK CALIBRATION  
**Date:** 2026-08-07  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, existing-checkpoint intervention  
**Hypothesis:** `tinyllm-calibrated-orientation-noise-radius-v1`  
**Schema:** `nal.tinyllm-calibrated-orientation-noise.v1`

**Pre-primary lifecycle amendment:** a reduced one-seed CUDA systems run under
`20260807_shakedown_cuda` exposed no scientific conclusion but correctly
failed the clean byte-replay contract: the initial implementation recomputed
`cos(alpha), sin(alpha)` even at `sigma=0`, changing float32 packet entries by
at most `5.96e-8`. Before primary execution, zero noise is repaired to return
an exact clone of the stored packet under a new lifecycle root. The nonzero
construction, sources, seeds, grid, metrics, thresholds, controls,
classification, and stop rules are unchanged.

## Decision question

The calibrated identifiability experiment established that clean analytic and
learned symmetry-respecting front ends form a stable cosine quotient in five
of five d8 checkpoints. The frozen-writer sidecar branch is now closed. Test
the next constructive question without retraining:

> How much error in the observed gauge-fixing orientation reference can each
> validated front end tolerate before cosine retention and conditional branch
> contraction cease to coexist?

This is a causal intervention on the reference, not a new observational probe.
It estimates a robustness radius for the mechanism already shown to work.

## Locked source

Use only the analytic and learned structured arms from:

```text
data/experiments/tinyllm_calibrated_frontend_causal/
    20260806_d8_preregistered/campaign_results.json
SHA-256 80623a7283fed3d902a1cbf9fb58afe2a2fef82ab86406212f907de40a96c501
implementation 73e152b9b6c7ab51ad08e6a80786c32a71566f9d4fb410ff2cf3df53761faf77
```

The five checkpoint seeds `7, 17, 29, 41, 53` are replication units. Reload
each frozen TinyLLM, scalar embedding, and learned encoder exactly. No TinyLLM,
front end, embedding, or calibration estimator is trained or fine-tuned.
Conditional diagnostic probes are newly fit and are not part of the deployed
mechanism.

## Intervention

Let the observed orientation record be the unit vector
`o=(cos(alpha), sin(alpha))`. For each analysis example draw one deterministic
standard-normal value `e` from its regime, split, and fiber identifier. Reuse
the same `e` for both C2 sheets and every noise level. Replace only the
orientation record by

```text
o_sigma = (cos(alpha + sigma e), sin(alpha + sigma e)).
```

The corruption is independent of phase, branch, target, sensor values, and
checkpoint. Amplitude, signed speed, offset, and drift remain exact. Pair-shared
noise prevents the intervention itself from encoding sheet identity. Common
random numbers make changes across levels paired and interpretable.

Before execution, lock the deterministic draw precisely: sort the observed
`fiber_id` values within each split, draw one `float64` standard normal per
unique fiber from NumPy `default_rng(6_700_019 + split_seed)`, and map it back
through the fiber identifier. This base tensor is stored once at campaign
scope and hashed; every arm, checkpoint, and registered level reuses it.

The locked standard-deviation grid, in radians, is:

```text
0, 0.035, 0.087, 0.175, 0.349, 0.524, 0.785
```

These correspond approximately to `0, 2, 5, 10, 20, 30, 45` degrees. No level
may be added, removed, or selected after outcome inspection.

## Fresh analysis cohorts

Use `analysis_seed=19083` and the predecessor's fixed offsets:

| Split | Seed | Samples | Use |
| --- | ---: | ---: | --- |
| train | `19184` | `2048` | fit diagnostic probes only |
| validation | `19294` | `512` | probe selection only |
| in distribution | `19390` | `1024` | descriptive |
| composition | `20399` | `1024` | primary |
| extrapolation | `21408` | `1024` | primary |

The same underlying examples are reused across the noise grid. Neither model
parameters nor probe parameters transfer between levels; each level receives
the same probe capacity and schedule as the predecessor (`240` maximum steps,
width `128`, deterministic early stopping).

## Measurements and gates

At `frontend` and `full`, report:

1. Pearson correlation of the fitted cosine readout with target cosine;
2. nonlinear conditional branch balanced accuracy given cosine;
3. conditional branch log-loss gain over the cosine-only null; and
4. full-depth task exact-bin accuracy and circular error.

A cut/regime cell passes only when all three representation conditions hold:

```text
cosine correlation >= 0.90
conditional branch balanced accuracy <= 0.55
conditional log-loss gain <= 0.02
```

A checkpoint-level noise gate additionally requires:

- all four primary cells (`frontend/full` by `composition/extrapolation`) pass;
- full-depth exact-bin accuracy in each primary regime falls by no more than
  `0.03` absolute from that checkpoint and arm's clean value on the same fresh
  cohort; and
- all provenance, checkpoint, feature-width, finite-value, pair-shared-noise,
  and clean-replay contracts pass.

A population level passes in at least four of five checkpoint seeds. The
robustness radius is the largest registered sigma whose population gate passes.
If a lower level fails after a higher level passes, report
`nonmonotone_no_single_radius`; do not smooth, interpolate, or select a prettier
threshold. The analytic arm is the positive-control mechanism. The learned
arm is the primary architectural result.

## Validity and controls

Before interpreting the curve, require:

1. source aggregate, result, model, and front-end hashes replay;
2. clean (`sigma=0`) fresh analysis passes the complete joint gate in at least
   four of five seeds for both arms;
3. orientation vectors retain unit norm within `1e-6`;
4. paired sheets receive bit-identical angular errors;
5. corruption tensors are identical across arms and checkpoints for a split;
6. each nonzero level equals its registered sigma times the locked base noise;
7. every recorded feature and metric is finite; and
8. exact resume preserves completed JSON and array bytes.

A clean replay failure invalidates the campaign. A learned-arm failure at zero
is not a noise-radius result. The analytic and learned arms are evaluated on
identical corruptions; raw TinyLLM is omitted because it failed the clean
quotient gate in `0/5` predecessor seeds and cannot define a robustness radius.

## Classification

| Outcome | Classification | Decision |
| --- | --- | --- |
| validity or clean replay fails | `invalid` | repair lifecycle only under a new root |
| learned radius equals or exceeds analytic radius | `learned_matches_analytic_radius` | test missing calibration fields next |
| learned radius is positive but smaller | `learned_brittle_relative_to_analytic` | train with declared calibration noise, then fresh-confirm |
| learned fails at first nonzero level while analytic is robust | `learned_clean_only_equivariance` | treat exact equivariance as insufficient for reference robustness |
| both fail at first nonzero level | `reference_precision_critical` | calibration accuracy, not model capacity, is the next engineering bottleneck |
| pass/fail curve is nonmonotone | `nonmonotone_no_single_radius` | inspect probe variance with a preregistered replication, not threshold tuning |

No topology scan, residual penalty, wider writer, sidecar observer, or TinyLLM
retraining follows before this frozen causal curve is resolved.

## Fixed artifacts

- runner: `experiments/structure_net/tinyllm_calibration_orientation_noise.py`
- tests: `tests/structure_net/test_tinyllm_calibration_orientation_noise.py`
- result root:
  `data/experiments/tinyllm_calibration_orientation_noise/20260807_d8_preregistered`
- report:
  `docs/08 - Analysis/2026-08-07_tinyllm-calibration-orientation-noise.md`
- meta hypothesis: `tinyllm-calibrated-orientation-noise-radius-v1`

## Scope boundary

This measures test-time orientation-reference error on one synthetic nuisance
family and five already successful checkpoints. It does not estimate real
instrument calibration cost, robustness to missing reference fields, or
population prevalence beyond the selected checkpoint cohort.
