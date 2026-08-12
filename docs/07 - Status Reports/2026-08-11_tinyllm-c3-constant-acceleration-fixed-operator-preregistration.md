# TinyLLM C3 constant-acceleration fixed-operator preregistration

**Status:** FROZEN BEFORE PRIMARY DATA GENERATION

**Date:** 2026-08-11

**Conformance:** NAL-STD-EXPERIMENT `PROSPECTIVE NO-TRAINING DYNAMICS-SCOPE PREFLIGHT`

**Hypothesis:** `tinyllm-c3-constant-acceleration-fixed-operator-v1`

## Decision question

The calibrated, noiseless, constant-speed `C3` task is closed by a fixed
all-increment circular mean, and the frozen TinyLLM continuation cannot use the
corresponding projected trajectory. The smallest licensed scope change is a
nonconstant temporal law:

> Does constant angular acceleration create a genuine learned-temporal-model
> opportunity, or does one fixed degree-2 group-polynomial operator still solve
> the declared task without TinyLLM?

This is an analytic-ceiling preflight. It instantiates no model, loads no
checkpoint, fits no coefficient, and performs no optimization. Passing the
fixed ceiling closes constant-acceleration TinyLLM training before it begins.
Failing it licenses only a new estimator/function-class preflight; it does not
by itself license an unrestricted transformer.

## Scope change and retained contracts

Change only the latent temporal law. Retain:

- the exact observable three-channel cyclic action;
- eight quantized observations and prediction of time `8`;
- calibrated amplitude, offset, and linear sensor drift;
- the sixteen-bin physical interval posterior;
- composition and outside-range extrapolation families;
- five independent dataset seeds and joint both-shift decisions.

For `t=0,...,8`, define

```text
theta_t = phase + velocity * t + acceleration * t(t-1)/2
q_t     = exp(i * 3 * theta_t).
```

Then the group increment and group acceleration are

```text
d_t = q_t conjugate(q_(t-1))
a_t = d_t conjugate(d_(t-1)),
```

and every exact constant-acceleration sequence satisfies

```text
a_t = exp(i * 3 * acceleration)
q_8 = q_7 d_7 a_t.
```

The target remains `Re(q_8)`. Deck transformations shift latent phase by a
third-turn and preserve the target exactly.

## Frozen generator

Use seeds

```text
107, 127, 149, 173, 197
```

with `4,096` examples per seed and shift. Draw phase uniformly on `[0,2pi)`.
Draw independent signs for velocity and acceleration, then draw magnitudes
from:

| Family | `|velocity|` | `|acceleration|` | amplitude | offset | drift |
| --- | ---: | ---: | ---: | ---: | ---: |
| composition | `[.04,.12]` | `[.010,.025]` | `[.7,1.8]` | `[-.4,.4]` | `[-.06,.06]` |
| extrapolation | `[.13,.20]` | `[.026,.050]` | `[.5,2.2]` | `[-.7,.7]` | `[-.10,.10]` |

Quantize with the retained `1,024` bins on `[-4,4]`. Saturation must be zero.
Dataset seed streams are `811107 + seed` for composition and `813107 + seed`
for extrapolation. Target-derangement streams are `821107 + seed` and
`823107 + seed` respectively.

The two shift families are disjoint in both velocity and acceleration
magnitude. No cohort, threshold, or candidate may be changed after primary
generation.

Pre-primary integration tests may use `64` examples from disjoint pilot seed
`991`. Pilot metrics are systems-lifecycle evidence only, are not interpreted
against the scientific gates, and are never pooled with the ten primary cells.

## Fixed operator arms

Compute the analytic carrier from observed tokens and calibration. Compare
exactly three preregistered operators.

### `constant_speed_mean`

The inherited misspecified comparator averages all seven increments:

```text
d_mean = normalize(sum_(t=1..7) d_t)
q_8_hat = q_7 d_mean.
```

### `recent_degree2`

The minimal last-three-frame degree-2 operator is

```text
a_7 = d_7 conjugate(d_6)
q_8_hat = q_7 d_7 a_7.
```

### `all_frame_degree2`

The primary operator averages all six observable group accelerations and all
seven implied estimates of the final increment:

```text
a_mean = normalize(sum_(t=2..7) d_t conjugate(d_(t-1)))
d_7_t  = d_t a_mean^(7-t)
d_7_mean = normalize(sum_(t=1..7) d_7_t)
q_8_hat = q_7 d_7_mean a_mean.
```

This operator is fixed by group algebra. It has no learned weight, target
access, unwrap heuristic, or fitted parameter.

## Primary endpoints

Evaluate each operator with:

- scalar correlation and RMSE against `Re(q_8)`;
- exact-bin accuracy, posterior-mean correlation/RMSE, cross-entropy, and
  predicted-bin coverage through the fixed sixteen-bin interval decoder;
- exact continuous-law and deck-action errors;
- target-deranged scalar and task controls.

The complete task gate remains:

| Endpoint | Composition | Extrapolation |
| --- | ---: | ---: |
| posterior-mean correlation | `>=.90` | `>=.90` |
| exact-bin accuracy | `>=.50` | `>=.35` |
| target cross-entropy | `<=1.80` | `<=2.20` |
| predicted-bin coverage | `>=14` | `>=12` |

The stronger fixed-ceiling gate requires, in each shift:

```text
scalar RMSE <= .020
exact-bin accuracy >= .90
complete task gate passes.
```

An operator passes a seed only when the gate passes both shifts. The population
threshold is `4/5` seeds.

The all-frame operator materially dominates the recent operator in a seed only
when both shifts satisfy:

```text
scalar RMSE ratio <= .75
exact-bin accuracy delta >= -.005
cross-entropy delta <= .005.
```

## Controls and validity

- Continuous unquantized `recent_degree2` and `all_frame_degree2` predictions
  must reproduce the exact complex `q_8` within `1e-12`.
- Identity, composition, order-three, stored-action, and independently
  regenerated deck-action token errors must all be zero.
- Maximum target change under a deck action must be `<=2e-12`.
- Every dataset must be deterministic under exact regeneration and have zero
  quantizer saturation.
- Every target derangement must have zero fixed points.
- For every operator and cell, shuffled-target absolute scalar correlation must
  be `<=.10` and shuffled scalar RMSE must be `>=.80`.
- No shuffled-target complete task gate may pass.
- All recorded values must be finite and strict JSON.
- Optimizer steps, parameters changed, models, checkpoints, and target-using
  fits must all be zero.

Any failed validity contract classifies the campaign as invalid and prevents a
scientific conclusion.

## Locked classifications

Use the first matching valid row:

| Outcome | Classification | Program decision |
| --- | --- | --- |
| all-frame fixed ceiling `>=4/5` and material dominance `>=4/5` | `fixed_all_frame_degree2_closes_constant_acceleration` | promote the all-frame degree-2 operator; do not train TinyLLM on this law |
| recent or all-frame fixed ceiling `>=4/5`, without all-frame material dominance | `fixed_degree2_closes_without_multistep_dominance` | use the cheaper passing degree-2 operator; do not train TinyLLM on this law |
| neither degree-2 operator reaches `4/5` | `registered_fixed_degree2_family_insufficient` | inspect a robust typed estimator/function class; unrestricted TinyLLM remains unlicensed |
| any validity failure | `invalid_constant_acceleration_preflight` | repair infrastructure only |

The misspecified constant-speed arm is descriptive and cannot determine the
classification.

## Frozen sources

| Source | SHA-256 |
| --- | --- |
| retained C3 generator/action | `812ac0a574bab812ff9fbbea7a0997c71e8ed0c50f98b270b8344234956ec118` |
| physical interval decoder | `b027eb55d971b87f598bd600131b33ead82463e3e73766fcdce38fdc3a5497c6` |
| constant-speed fixed-operator result | `9b03a0cf19ef820586e2a031d80a694d498655de151f70f245c3c82b0abb853a` |
| constant-speed fixed-operator runner | `9471ffe7f319d3d53234fd795fc48ce39a7717970d0e191ae2882343ad6d3b37` |
| frozen feature-trajectory result | `a0ac3315b03aa65df273539a24d8c08f51f12e8ceb702859cc16a282886ddf27` |

The implementation must pin and revalidate these sources plus this
preregistration before generating primary data.

## Expected artifact and accounting

```text
data/experiments/tinyllm_c3_constant_acceleration_fixed_operator/
  20260811_preregistered/result.json
```

```text
new evaluation examples: 40,960
optimizer steps:          0
parameters changed:       0
models/checkpoints:       0 / 0
target-using fits:        0
```
