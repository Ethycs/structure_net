# TinyLLM C3 hidden-order corruption fixed-estimator preregistration

**Status:** FROZEN BEFORE CROSS-LAW CORRUPTION GENERATION

**Date:** 2026-08-11

**Conformance:** NAL-STD-EXPERIMENT `PROSPECTIVE NO-TRAINING JOINT-SCOPE PREFLIGHT`

**Hypothesis:** `tinyllm-c3-hidden-order-corruption-fixed-estimator-v1`

## Decision question

The previous fixed estimators separately close known constant speed, known
constant acceleration, and one unmarked corrupted frame under constant
acceleration. A proposed successor would hide whether each sequence has degree
one or degree two dynamics and add the same observation corruption.

Before training a selector, test the nesting relation:

> Does hidden constant-speed versus constant-acceleration order plus one
> unmarked frame substitution exceed the single frozen robust quadratic
> estimator, or is the apparent model-selection problem already closed because
> constant speed is the `beta_2=0` subfamily of the quadratic chart?

This experiment performs no learned or adaptive law selection. One estimator,
without a law label, is applied unchanged to both laws and their actual 50:50
mixture. Passing closes this proposed TinyLLM branch before model construction.

## Algebraic inclusion contract

For the invariant carrier, both laws lie in

```text
arg(q_t) = beta_0 + beta_1 t + beta_2 t(t-1)/2.
```

Constant speed is exactly `beta_2=0`; constant acceleration permits nonzero
`beta_2`. Therefore an unknown mixture of the two laws does not enlarge the
declared function class. The empirical question is only whether quantization,
unwrapping, and hidden corruption make the common robust estimator fail its
registered ceiling.

The runner must assert the exact continuous inclusion and may not read, infer,
or branch on a law label when producing the primary robust prediction.

## Frozen matched populations

Pair the five frozen constant-speed cohorts with the five frozen
constant-acceleration cohorts by replication position:

| Replicate | Constant-speed seed | Constant-acceleration seed |
| ---: | ---: | ---: |
| 1 | `7` | `107` |
| 2 | `17` | `127` |
| 3 | `29` | `149` |
| 4 | `41` | `173` |
| 5 | `53` | `197` |

Each law contributes `4,096` examples to composition and `4,096` to
extrapolation. Regenerate every base cohort from its frozen runner and require
its predecessor dataset hash to match exactly.

For every replicate and shift, the mixed population is the concatenation of
the full constant-speed and constant-acceleration cohorts: `8,192` examples at
an exact 50:50 law mixture. No sample is selected by outcome.

Disjoint lifecycle tests may use `64` examples per law from seed `991`. Pilot
values are never compared with scientific gates or pooled with primary cells.

## Frozen corruption

Apply the already sealed single-frame corruption intervention independently to
each law family:

- one time index sampled uniformly from `0,...,7` per sequence;
- all three quantized channels replaced by the same-time frame from a Sattolo-
  deranged donor in the same family and shift;
- donor identity and frame index hidden from the primary estimator;
- calibration and target retained from the receiving example.

Use the frozen corruption streams and implementation. Because paired law seeds
are distinct, their donor permutations and frame-index draws are independent.
Each frame position must occur at least `400` times per primary family/cell.

Use independent target-derangement streams:

| Family | Composition | Extrapolation |
| --- | ---: | ---: |
| constant speed | `861107 + seed` | `863107 + seed` |
| constant acceleration | `865107 + seed` | `867107 + seed` |

The mixed shuffled target concatenates the two within-family derangements so
the 50:50 law proportion is preserved.

## Fixed arms

Use four arms per law and in the concatenated mixture.

### `clean_law_specific`

Apply the frozen all-increment mean to clean constant-speed carriers and the
frozen all-frame degree-2 group operator to clean acceleration carriers. This
is a positive ceiling only.

### `corrupted_law_specific`

Apply those same law-specific operators naively to all eight corrupted frames.
This tests whether the combined corruption is material; it is not the common
candidate.

### `oracle_drop_one_quadratic`

Apply the frozen quadratic chart after deleting the true corrupted frame. The
law label remains unused. This is the observation-identifiability control.

### `robust_drop_one_quadratic`

Apply the exact sealed estimator from the single-frame-corruption study:
enumerate every candidate deletion, fit the fixed quadratic phase chart to the
other seven frames, choose minimum retained residual, and predict time `8`.
Use identical code and weights—none—on both laws. Do not add a degree selector,
law classifier, threshold, or fitted coefficient shared across examples.

## Primary endpoints

Evaluate each arm separately on constant speed, constant acceleration, and the
actual 50:50 mixture with the frozen scalar and sixteen-bin physical metrics.
Retain the complete task gate:

| Endpoint | Composition | Extrapolation |
| --- | ---: | ---: |
| posterior-mean correlation | `>=.90` | `>=.90` |
| exact-bin accuracy | `>=.50` | `>=.35` |
| target cross-entropy | `<=1.80` | `<=2.20` |
| predicted-bin coverage | `>=14` | `>=12` |

The strong fixed ceiling additionally requires:

```text
scalar RMSE <= .020
exact-bin accuracy >= .90
complete task gate passes.
```

A replicate passes the common-estimator endpoint only if the robust arm clears
the strong ceiling for both law families and the mixture under both shifts.
Require at least `4/5` paired replicates.

For every family/mixture cell, corruption is material only when the clean arm
passes and the corrupted naive arm fails the strong ceiling. Robust repair and
oracle fidelity retain the sealed thresholds:

```text
repair:
  robust fixed ceiling passes
  robust / corrupted-naive RMSE <= .50
  robust accuracy delta >= .20
  robust cross-entropy delta <= -.10

oracle fidelity:
  robust RMSE <= oracle RMSE + .002
  robust accuracy >= oracle accuracy - .01
  robust cross-entropy <= oracle cross-entropy + .005.
```

Materiality, repair, and oracle fidelity must each pass all three populations
and both shifts in at least `4/5` paired replicates.

## Controls and validity

- Replayed base dataset hashes must match all twenty predecessor cells.
- Base saturation, deterministic generation, and exact observable `C3` action
  contracts must pass.
- Corruption donors must be deranged, frame coverage must pass, and corruption
  regeneration must be bitwise exact.
- Corruption must commute exactly with global deck action; every prediction arm
  must be deck invariant within `2e-12`.
- The minimum clean post-deletion phase-chart margin must be at least `.20`.
- With continuous carriers and the same donor/index intervention, oracle and
  robust predictions must recover exact time `8` within `1e-10`, and robust
  deletion must identify every corrupted index.
- A fixed-point-free target derangement must give absolute scalar correlation
  `<=.10`, scalar RMSE `>=.80`, and zero complete task passes in every family
  and mixture arm.
- All values must be finite strict JSON.
- Law-label reads, adaptive selector decisions, learned parameters, models,
  checkpoints, optimizer steps, changed parameters, and target-using fits must
  all be zero.

Any failed validity contract prevents a scientific conclusion.

## Locked classifications

Use the first matching valid row:

| Outcome | Classification | Program decision |
| --- | --- | --- |
| common robust endpoint, materiality, repair, and fidelity each `>=4/5` | `single_robust_quadratic_closes_hidden_order_and_corruption` | reject selector and TinyLLM training; the family is nested and fixed-estimator closed |
| corrupted naive mixture ceiling `>=4/5` | `hidden_order_corruption_not_material` | scope is too weak; do not train |
| oracle endpoint `>=4/5` but common robust endpoint `<4/5` | `recoverable_hidden_order_corruption_exceeds_common_fixed_estimator` | license a compact typed selector comparison, not unrestricted TinyLLM |
| oracle endpoint `<4/5` | `hidden_order_corruption_not_recoverable_at_required_ceiling` | repair observation/target identifiability; do not train |
| any other valid combination | `inconclusive_hidden_order_corruption_preflight` | inspect the joint gate without tuning on primary cells |
| any validity failure | `invalid_hidden_order_corruption_preflight` | repair infrastructure only |

## Frozen sources

| Source | SHA-256 |
| --- | --- |
| constant-speed fixed-operator runner | `9471ffe7f319d3d53234fd795fc48ce39a7717970d0e191ae2882343ad6d3b37` |
| constant-speed fixed-operator result | `9b03a0cf19ef820586e2a031d80a694d498655de151f70f245c3c82b0abb853a` |
| single-frame-corruption runner | `8600081fc813ae6da5a49c39222bf3a94286127d7238899d61ec9acd1c6d31f8` |
| single-frame-corruption result | `59681f2764b988f05b0916965898b87d5b233b2165151fe19e0c97391fe467b9` |

Those sources transitively pin the generator, action, calibration, interval
decoder, acceleration runner/result, corruption law, estimator code, and all
thresholds. The implementation must validate the four direct hashes plus this
preregistration before generating a primary corruption.

## Expected artifact and accounting

```text
data/experiments/tinyllm_c3_hidden_order_corruption_fixed_estimator/
  20260811_preregistered/result.json
```

```text
matched base examples replayed:       81,920
new corrupted evaluations:            81,920
closed-form observation-only fits:   737,280
law-label reads:                            0
adaptive selector decisions:               0
optimizer steps:                            0
parameters changed:                         0
models/checkpoints:                         0 / 0
target-using fits:                          0
```
