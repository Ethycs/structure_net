# TinyLLM C3 single-frame corruption fixed-estimator preregistration

**Status:** FROZEN BEFORE PRIMARY CORRUPTION GENERATION

**Date:** 2026-08-11

**Conformance:** NAL-STD-EXPERIMENT `PROSPECTIVE NO-TRAINING OBSERVATION-SCOPE PREFLIGHT`

**Hypothesis:** `tinyllm-c3-single-frame-corruption-fixed-estimator-v1`

## Decision question

Known constant-speed and constant-acceleration laws are already closed by fixed
group operators. The smallest unresolved scope is uncertainty in the observed
trajectory:

> Does one unmarked gross frame substitution create work for a learned temporal
> model, or does a fixed robust estimator still recover the constant-acceleration
> target under composition and extrapolation?

This is an analytic-ceiling preflight. It trains no model, loads no checkpoint,
and uses no target or label to estimate a trajectory. Passing the robust ceiling
closes TinyLLM training on this corruption law. Failure can license a learned
sensor comparison only when the oracle deletion control shows that the target
remains identifiable.

## Retained law and frozen cohorts

Reuse the five deterministic constant-acceleration base cohorts and all of their
generator, action, calibration, quantization, target, and shift contracts:

```text
seeds:                 107, 127, 149, 173, 197
examples per shift:    4,096
shifts:                composition, extrapolation
observed times:        0,...,7
target time:           8
target:                cos(3 theta_8)
```

The composition and extrapolation ranges remain exactly those in the frozen
constant-acceleration study. Regenerating a base cohort must reproduce its
deterministic hash and all exact `C3` action contracts before corruption.

Pilot integration may use only `64` examples from the already reserved seed
`991`. Pilot values are lifecycle evidence, are not compared with scientific
gates, and are never pooled with the ten primary cells.

## Frozen corruption intervention

For every sequence independently, draw one index uniformly from `0,...,7`.
Replace all three quantized sensor tokens at that time with the same-time frame
from another example in the cell. Donors come from a deterministic Sattolo
derangement, so no example donates to itself. The frame index and donor identity
are not supplied to the primary estimator.

This is a gross but marginally matched corruption: the replacement is an
observed frame from the same shift and time, not an arbitrary out-of-vocabulary
token. Calibration and target remain those of the receiving example. The donor
frame is copied only after the clean dataset has been generated and deck acted.

Use independent streams:

| Shift | donor stream | frame-index stream |
| --- | ---: | ---: |
| composition | `831107 + seed` | `841107 + seed` |
| extrapolation | `833107 + seed` | `843107 + seed` |

Each of the eight frame positions must occur at least `400` times in every
primary cell. Donor fixed points must be zero. Corruption generation must be
bitwise deterministic.

## Fixed estimator arms

Extract the invariant complex carrier `q_t` from tokens and observed calibration
using the frozen analytic `C3` carrier.

### `clean_all_frame_degree2`

Apply the frozen all-frame degree-2 group operator to the uncorrupted carrier.
This is a positive ceiling and provenance replay, not a deployable corruption
arm.

### `corrupted_all_frame_degree2`

Apply the same operator directly to all eight corrupted frames. This is the
registered naive comparator and tests whether the new scope is material.

### `oracle_drop_one_quadratic`

Delete the known corrupted frame, unwrap the remaining carrier phases in time
order, fit the fixed quadratic chart

```text
arg(q_t) = beta_0 + beta_1 t + beta_2 t(t-1)/2
```

by its closed-form pseudoinverse, and evaluate it at time `8`. This arm uses the
corruption index but never the target. It is the identifiability and chart
positive control.

### `robust_drop_one_quadratic`

For each candidate deletion `j=0,...,7`, perform the same target-free quadratic
fit on the other seven frames and compute mean squared phase residual on those
seven frames. Choose the candidate with the smallest residual, breaking exact
ties toward the lower index, and return its time-8 prediction. No coefficient,
threshold, frame subset, or branch is tuned from primary outcomes.

The quadratic design uses columns `1`, `t`, and `t(t-1)/2`. Unwrapping uses
principal consecutive phase increments after deletion. Over the declared
ranges, the largest clean two-time-step carrier increment must remain below
`pi - .20`; violating this chart-margin contract invalidates the cell.

Closed-form per-example state estimation is part of the fixed operator. It is
not a dataset-level fit and does not alter a reusable parameter.

## Primary endpoints and gates

Evaluate all four arms with the frozen scalar and sixteen-bin physical task
metrics. Retain the complete task gate:

| Endpoint | Composition | Extrapolation |
| --- | ---: | ---: |
| posterior-mean correlation | `>=.90` | `>=.90` |
| exact-bin accuracy | `>=.50` | `>=.35` |
| target cross-entropy | `<=1.80` | `<=2.20` |
| predicted-bin coverage | `>=14` | `>=12` |

The strong fixed ceiling additionally requires, in each shift:

```text
scalar RMSE <= .020
exact-bin accuracy >= .90
complete task gate passes.
```

An arm passes a seed only when it passes both shifts. The population threshold
is `4/5` seeds.

The corruption is material in a seed only when the clean positive control
passes both shifts and the corrupted naive arm fails the fixed ceiling in both
shifts. Materiality must hold in `>=4/5` seeds.

The robust estimator materially repairs a cell only when:

```text
robust fixed ceiling passes
robust / corrupted-naive scalar RMSE ratio <= .50
robust exact-bin accuracy delta >= .20
robust target cross-entropy delta <= -.10.
```

It is oracle-faithful in a cell only when:

```text
robust scalar RMSE <= oracle scalar RMSE + .002
robust exact-bin accuracy >= oracle accuracy - .01
robust target cross-entropy <= oracle cross-entropy + .005.
```

Repair and oracle-fidelity population gates require every condition in both
shifts in `>=4/5` seeds. Corruption-index recovery is a secondary diagnostic,
not a primary gate, because a donor frame can accidentally be trajectory
compatible without harming prediction.

## Controls and validity

- Base dataset hashes, saturation, deterministic regeneration, and exact `C3`
  action contracts must replay.
- Corruption donors must be deranged, frame-position coverage must pass, and
  corrupted tensor regeneration must be bitwise exact.
- Corrupting globally deck-transformed source and donor frames must equal
  globally deck-transforming the corrupted tensor, with zero token errors.
- Every arm must be invariant to global deck action within `2e-12`.
- On continuous carriers with the identical donor/index intervention, oracle
  and robust predictions must recover the exact complex time-8 state within
  `1e-10`; the robust deletion must recover every corruption index.
- Clean retained time gaps after true deletion must satisfy the `.20` chart
  margin.
- A fixed-point-free target derangement must yield absolute scalar correlation
  `<=.10`, scalar RMSE `>=.80`, and zero complete task-gate passes for every arm
  and cell.
- All values must be finite strict JSON.
- Model instances, checkpoints, optimizer steps, changed parameters, and
  target-using fits must all be zero.

Any validity failure prevents scientific classification.

## Locked classifications

Use the first matching valid row:

| Outcome | Classification | Program decision |
| --- | --- | --- |
| corruption material `>=4/5`, robust ceiling `>=4/5`, repair `>=4/5`, oracle fidelity `>=4/5` | `fixed_robust_estimator_closes_single_frame_corruption` | promote the robust operator; do not train TinyLLM on this corruption law |
| corrupted naive ceiling `>=4/5` | `single_frame_corruption_not_material` | scope is too weak; do not train TinyLLM |
| corruption material `>=4/5`, robust ceiling `<4/5`, oracle ceiling `>=4/5` | `recoverable_corruption_exceeds_registered_fixed_estimator` | license a matched compact robust/equivariant sensor comparison, not unrestricted TinyLLM |
| oracle ceiling `<4/5` | `declared_corruption_not_recoverable_at_required_ceiling` | repair observation/target identifiability; do not train |
| any other valid combination | `inconclusive_single_frame_corruption_preflight` | do not train; inspect the failed joint gate without retuning on primary cells |
| any validity failure | `invalid_single_frame_corruption_preflight` | repair infrastructure only |

No result licenses tuning this estimator on the ten primary cells.

## Frozen sources

| Source | SHA-256 |
| --- | --- |
| constant-acceleration runner | `6ea952f386b82b12355c3aa2e9552af6bf73e03e7cd47310fec764ce49d0d5e2` |
| constant-acceleration result | `b04a5574efc658ec1ed73f70fa494041ad16c0ae1342423cdde32925c1c7bc53` |
| retained C3 generator/action | `812ac0a574bab812ff9fbbea7a0997c71e8ed0c50f98b270b8344234956ec118` |
| physical interval decoder | `b027eb55d971b87f598bd600131b33ead82463e3e73766fcdce38fdc3a5497c6` |

The implementation must pin and revalidate these four sources plus this
preregistration before generating primary corruptions.

## Expected artifact and accounting

```text
data/experiments/tinyllm_c3_single_frame_corruption_fixed_estimator/
  20260811_preregistered/result.json
```

```text
new base evaluation examples:          0 (matched deterministic replay)
new corrupted evaluations:             40,960
closed-form observation-only fits:      368,640
optimizer steps:                        0
parameters changed:                     0
models/checkpoints:                     0 / 0
target-using fits:                      0
```
