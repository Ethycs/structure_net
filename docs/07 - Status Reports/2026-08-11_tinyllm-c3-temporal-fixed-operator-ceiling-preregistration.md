# TinyLLM C3 temporal fixed-operator ceiling preregistration

**Status:** FROZEN BEFORE NEW COHORT EVALUATION

**Date:** 2026-08-11

**Conformance:** NAL-STD-EXPERIMENT `PROSPECTIVE / NO-TRAINING GROUP-OPERATOR COMPARISON`

**Hypothesis:** `tinyllm-c3-temporal-fixed-operator-ceiling-v1`

## Decision question

The exact-`C3` sensor-only campaign succeeded in `5/5` seeds, and the following
causal decomposition showed that its solution is carried by the affine
identity character. TinyLLM was absent from both positive results. Before
training a typed continuation, ask the cheaper question:

> Does the noiseless observable sequence contain a material temporal correction
> beyond the current two-frame group operator, or can a fixed all-frame group
> estimator already consume it?

This decides whether TinyLLM has any same-task continuation work to perform.
It evaluates new data cohorts but trains no model, loads no checkpoint, fits no
coefficient, and changes no parameter.

## Frozen sources

| Source | SHA-256 |
| --- | --- |
| observable `C3` generator and analytic carrier | `812ac0a574bab812ff9fbbea7a0997c71e8ed0c50f98b270b8344234956ec118` |
| fixed interval likelihood | `b027eb55d971b87f598bd600131b33ead82463e3e73766fcdce38fdc3a5497c6` |
| five-seed sensor-only campaign | `4d023d9f9d77f64b1fe75970acd63461cfd5a64aa1de60c7954a2c671c427012` |
| affine sensor-mechanism result | `c3dbfecd7a6381c2129e4d99f135557f003ad8a225fa2b4d3f4fa0cb429f669b` |

The two learned-result sources motivate this question but do not supply its
outcomes. The primary cells use new generator seeds.

## Cohorts

Use independent replicate seeds

```text
7, 17, 29, 41, 53
```

with `4,096` examples per shift and replicate. Derive generator seeds as:

```text
composition:   531000 + replicate seed
extrapolation: 533000 + replicate seed.
```

Retain the registered composition and outside-speed ranges, eight time steps,
three observed channels, 1,024-bin quantization, amplitude/offset/drift
calibration packet, random deck action, and target

```text
T = cos(3 * (phase + 8 * speed)).
```

Require zero quantizer saturation, exact stored deck construction, finite
arrays, and deterministic tensor hashes in every cell.

## Frozen operators

Let the analytic invariant carrier sequence be

```text
q_t in C, |q_t| = 1, t = 0,...,7.
```

### Registered last-increment baseline

Retain the existing operator exactly:

```text
d_7 = q_7 * conjugate(q_6)
u_last = Re(q_7 * d_7).
```

### Fixed all-increment operator

Compute all seven adjacent group increments and their unweighted circular mean:

```text
d_t = q_t * conjugate(q_(t-1)),  t = 1,...,7
d_mean = sum_t d_t / max(|sum_t d_t|, 1e-12)
u_mean = Re(q_7 * d_mean).
```

No increment is dropped, weighted, clipped, selected, or fitted. This is the
unique new deployable arm. Do not add a weighted mean, phase-unwrapping rule,
robust estimator, learned coefficient, or post-outcome subset.

### Exact continuous positive control

Construct the unquantized carrier directly from generator phase and speed and
verify that both operators reproduce the exact target to maximum absolute error
`1e-12`. This validates constant-speed group algebra; it is not a deployable
result.

## Invariance and specificity controls

For both deployable operators:

- apply each nonidentity channel roll to the quantized observations and require
  scalar prediction error at most `2e-12`;
- deterministically derange targets within each cell using seed
  `631000 + generator seed`, require zero fixed points, absolute prediction/
  shuffled-target correlation at most `.10`, and shuffled RMSE at least `.80`;
- apply no target, phase, speed, or held-out statistic to either operator.

## Endpoints

Measure temporal scalar correlation and RMSE, followed by the unchanged
sixteen-bin physical likelihood. Retain the complete fixed-task gates:

| Endpoint | Composition | Extrapolation |
| --- | ---: | ---: |
| posterior-mean correlation | `>=.90` | `>=.90` |
| exact-bin accuracy | `>=.50` | `>=.35` |
| target cross-entropy | `<=1.80` | `<=2.20` |
| predicted-bin coverage | `>=14` | `>=12` |

For each seed and shift define:

```text
rmse_ratio = RMSE(u_mean, T) / RMSE(u_last, T)
accuracy_delta = accuracy_mean - accuracy_last
cross_entropy_delta = CE_mean - CE_last.
```

The all-increment operator materially dominates a cell only if all hold:

```text
complete mean-operator task gate passes
rmse_ratio <= .75
accuracy_delta >= -.005
cross_entropy_delta <= .005.
```

A seed passes material dominance only when both composition and extrapolation
cells pass. The baseline is at the declared fixed-operator ceiling for a seed
when its temporal RMSE is at most `.01` on both shifts.

## Locked population classifications

| Outcome | Classification | Decision |
| --- | --- | --- |
| all-increment material dominance in `>=4/5` seeds | `fixed_multistep_group_operator_dominates_last_step` | use the fixed all-frame operator; the same-task continuation has a known cheaper solution, so TinyLLM training is not licensed |
| dominance `<4/5`, baseline ceiling in `>=4/5` seeds | `last_step_already_at_declared_fixed_operator_ceiling` | the declared fixed path already exhausts material same-task work; TinyLLM training is not licensed |
| neither population rule passes | `fixed_operator_family_leaves_typed_residual_scope` | only a new no-training capacity/gradient preflight for a metric-typed residual continuation is licensed; unrestricted TinyLLM remains closed |
| any source, algebra, data, action, shuffle, finiteness, or determinism contract fails | `invalid_fixed_operator_ceiling_contract` | repair infrastructure only and draw no scientific conclusion |

No outcome licenses the stopped raw/learned predecessor cells, an unrestricted
TinyLLM continuation, or a sweep over temporal estimators. The negative branch
licenses only a separately registered typed residual function-class preflight.

## Accounting and expected artifact

```text
optimizer steps:             0
parameters changed:          0
checkpoints loaded:          0
TinyLLM models instantiated: 0
target-using fits:           0
```

Expected artifact:

```text
data/experiments/tinyllm_c3_temporal_fixed_operator_ceiling/
  20260811_preregistered/result.json
```
