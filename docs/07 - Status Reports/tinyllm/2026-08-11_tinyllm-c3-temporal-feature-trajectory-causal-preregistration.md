# TinyLLM C3 temporal feature-trajectory causal preregistration

**Status:** FROZEN BEFORE CHECKPOINT INTERVENTION

**Date:** 2026-08-11

**Conformance:** NAL-STD-EXPERIMENT `REGISTERED POST-OUTCOME / ARTIFACT-ONLY CAUSAL DIAGNOSTIC`

**Hypothesis:** `tinyllm-c3-temporal-feature-trajectory-causal-v1`

## Decision question

The fixed all-increment `C3` operator halves temporal error and materially
improves the task in `5/5` new replicates, while the original analytic-sensor
TinyLLM population passes natural utility in only `2/5`. Before training or
changing architecture, patch the newly identified computation into the five
frozen continuations at their existing analytic feature interface:

> Can the failed TinyLLM continuation use a denoised constant-speed carrier
> trajectory when it is supplied causally, or does the failure remain downstream
> of temporal feature construction?

This diagnostic loads the five analytic d6 checkpoints but performs no fit,
optimization, or parameter change.

## Frozen sources

| Source | SHA-256 |
| --- | --- |
| analytic d6 campaign | `e46710de358a3c9c5d30ddd4a19e36182b94a50cabc699014f79d58d3d3de7cc` |
| five-result manifest | `7dfdcf1ff80fe20a975fe6a7d1311dc92e3ff1a396a6da9550c91835a568a0ff` |
| fifteen-artifact manifest | `a0b90484863346cf2a5e0ef8be65cac3a221cfa50a373f88c9ead07a0cd351a1` |
| source campaign runner | `9b2cd0e3ce3752b7eea80d5859c11880a9d3732fb48b58306e34eab4f080d5ec` |
| source analysis | `89dacc60d02707678e689c6ce1e8f9c963889af352565a227bb90ed8e367e6a3` |
| source training/system | `dbc934190b5f725185ff9a99690409ac63fc4f16bd04686f870e7ef21cba03a6` |
| frozen loader/readout diagnostic | `3ccb922b8e0fb5119cc8c327024f6b9e1e957bb34b959dcb9e523518ac41746e` |
| fixed-operator ceiling result | `9b03a0cf19ef820586e2a031d80a694d498655de151f70f245c3c82b0abb853a` |

Revalidate the campaign, all source results and artifacts, dataset hashes,
checkpoint/frontend hashes, source state digests, and known `2/5` natural-task
population before intervention.

## Frozen systems and cohorts

Use analytic d6 seeds

```text
7, 17, 29, 41, 53
```

and the unchanged `1,024`-example composition and extrapolation cohorts from
the sealed source campaign. Run on `cuda:0` with batch size `256`. Recomputed
source metrics must match the sealed source result within `2e-6`, including
exact bin coverage.

## Feature interventions

Let the natural analytic feature sequence be

```text
q_0,...,q_7 in C, |q_t| = 1.
```

Patch the two real coordinates of the following sequences immediately before
the frozen learned sequence embedding. Leave positional embeddings, TinyLLM,
layer normalization, tied answer rows, and every parameter unchanged.

### `source`

Use the natural analytic carrier sequence. This must reproduce the frozen
checkpoint output.

### `last_consistent`

Anchor at the observed final carrier and reconstruct a constant-speed sequence
using only the last increment:

```text
d_last = q_7 * conjugate(q_6)
q_t_last = q_7 * d_last^(t-7).
```

Require `q_6` and `q_7` to match the source features within `2e-6`. This changes
only the first six frames apart from numerical roundoff and tests whether a
generic constant-trajectory projection is sufficient.

### `mean_consistent`

Use the preregistered all-increment circular mean:

```text
d_mean = normalize(sum_(t=1..7) q_t * conjugate(q_(t-1)))
q_t_mean = q_7 * d_mean^(t-7).
```

Require `q_7` to match within `2e-6` and require the last-to-next temporal scalar
of the projected trajectory to equal the fixed all-increment scalar within
`2e-6`. This is the primary causal repair.

### `early_deranged`

Use a deterministic Sattolo derangement seeded by `741000` for composition and
`743000` for extrapolation. Replace `q_0,...,q_5` with those from the deranged
example while retaining each example's own `q_6,q_7` exactly. Require zero
fixed points. This is a descriptive causal sensitivity stress, not a candidate
repair and not part of the population success gate.

No feature arm may use the target, phase, speed, source outcome, or a fitted
coefficient.

## Measurements

For every arm and shift record the complete frozen posterior metrics:

- posterior-mean correlation and RMSE;
- exact-bin accuracy;
- target cross-entropy;
- predicted-bin coverage.

Retain the source task gates:

| Endpoint | Composition | Extrapolation |
| --- | ---: | ---: |
| posterior-mean correlation | `>=.90` | `>=.90` |
| exact-bin accuracy | `>=.50` | `>=.35` |
| target cross-entropy | `<=1.80` | `<=2.20` |
| predicted-bin coverage | `>=14` | `>=12` |

An arm passes a seed only when the complete gate passes both shifts.

For each intervention relative to `source`, also record:

```text
maximum posterior change
mean absolute posterior change
mean Jensen-Shannon divergence
posterior-mean RMS change
argmax-change fraction.
```

These quantify causal early-frame use but do not replace task preservation.

## Controls and validity

- The fixed all-increment scalar plus fixed physical decoder must pass both
  shifts in all five seeds; it is the positive-control computation.
- Evaluate each arm against a deterministic Sattolo target derangement seeded
  by `751000 + checkpoint seed` for composition and
  `753000 + checkpoint seed` for extrapolation. Require zero fixed points and
  at most one `mean_consistent` shuffled seed to pass both shifts.
- Every source state digest must remain unchanged.
- All features, posteriors, metrics, and divergences must be finite.
- Source posterior/metric replay, feature algebra, source manifests, and dataset
  hashes are validity contracts, not adjustable endpoints.

## Locked classifications

| Outcome | Classification | Decision |
| --- | --- | --- |
| `mean_consistent >=4/5`, `last_consistent <4/5`, shuffled mean `<=1/5` | `all_frame_projection_repairs_frozen_continuation` | the continuation can use the correct group statistic when supplied; any successor must construct this typed trajectory before TinyLLM |
| `mean_consistent >=4/5`, `last_consistent >=4/5`, shuffled mean `<=1/5` | `constant_trajectory_projection_repairs_without_all_frame_specificity` | trajectory regularization repairs the interface, but the all-frame mean is not the unique causal ingredient |
| `mean_consistent <4/5`, fixed bypass `5/5`, shuffled mean `<=1/5` | `fixed_operator_available_but_frozen_continuation_cannot_use_projected_trajectory` | failure remains in continuation/readout use after a correct temporal trajectory is supplied; close same-task TinyLLM repair |
| shuffled mean `>1/5` | `feature_trajectory_specificity_failed` | do not interpret a true repair as task-specific |
| any validity contract fails | `invalid_feature_trajectory_causal_contract` | repair infrastructure only and draw no mechanistic conclusion |

No outcome licenses the stopped raw or learned predecessor cells, a loss sweep,
or unrestricted TinyLLM retraining. The result is a boundary localization for
the existing five analytic checkpoints.

## Accounting and expected artifact

```text
optimizer steps:             0
parameters changed:          0
checkpoints loaded:          5
TinyLLM models instantiated: 5 frozen source models
target-using fits:           0
```

Expected artifact:

```text
data/experiments/tinyllm_c3_temporal_feature_trajectory_causal/
  20260811_d6_preregistered/result.json
```
