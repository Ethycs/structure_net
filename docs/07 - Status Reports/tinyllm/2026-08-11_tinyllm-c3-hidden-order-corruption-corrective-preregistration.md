# TinyLLM C3 hidden-order corruption corrective preregistration

**Status:** FROZEN BEFORE FRESH COHORT GENERATION

**Date:** 2026-08-11

**Conformance:** NAL-STD-EXPERIMENT `PROSPECTIVE NO-TRAINING CORRECTIVE REPLICATION`

**Hypothesis:** `tinyllm-c3-hidden-order-corruption-corrective-v1`

## Decision question

The first hidden-order experiment was valid but inconclusive. One law-blind
robust quadratic estimator passed every absolute and oracle endpoint on
constant speed, constant acceleration, and their actual mixture. Its full
repair conjunction failed because constant-speed accuracy improved by less
than the borrowed `.20` effect-size threshold, despite strong absolute success
and more than `98%` RMSE repair.

This corrective asks on entirely fresh data:

> Does the same frozen law-blind robust quadratic estimator Pareto-repair one
> hidden corrupted frame across both nested laws and their mixture, without
> sacrificing accuracy, and still clear the absolute and oracle gates?

No old primary outcome is pooled or reclassified. The previous result remains
`inconclusive_hidden_order_corruption_preflight` regardless of this outcome.

## Corrective rationale fixed independently of the failed value

Replace only the redundant `.20` accuracy-gain clause with the pre-existing
nondegradation guard used by the constant-speed and constant-acceleration fixed-
operator studies before the hidden-order outcome existed:

```text
accuracy delta >= -.005.
```

The absolute robust ceiling already requires accuracy `>=.90`; materiality
already requires the naive corrupted arm to fail that ceiling. The repair gate
should reject a scalar/likelihood tradeoff, not demand that every baseline have
the same initial accuracy damage.

Retain the more stringent corruption repair requirements for RMSE and
cross-entropy unchanged.

## Fresh paired populations

Use five new paired seeds, disjoint from every earlier primary and pilot cell:

| Replicate | Constant-speed seed | Constant-acceleration seed |
| ---: | ---: | ---: |
| 1 | `71` | `211` |
| 2 | `83` | `229` |
| 3 | `97` | `251` |
| 4 | `109` | `277` |
| 5 | `131` | `307` |

Generate `4,096` examples per family, replicate, and shift with the exact
predecessor distributions, quantizer, calibration, action, and target:

| Family | Composition base stream | Extrapolation base stream |
| --- | ---: | ---: |
| constant speed | `911107 + seed` | `913107 + seed` |
| constant acceleration | `915107 + seed` | `917107 + seed` |

The mixed cell is the full concatenation of the two `4,096`-example family
cells, giving an actual 50:50 `8,192`-example mixture. Do not subsample or
select by outcome.

Disjoint lifecycle tests may use `64` examples per law from seed `997` with
separate pilot streams. Pilot measurements are never interpreted against
scientific gates.

## Frozen corruption and target controls

Use the sealed unmarked one-frame donor substitution unchanged. For each
sequence, sample one frame uniformly from `0,...,7` and replace all three
quantized channels with the same-time frame from a Sattolo-deranged donor in
the same family and shift. Do not reveal the donor or frame to the primary
estimator.

Use the frozen corruption streams:

| Purpose | Composition | Extrapolation |
| --- | ---: | ---: |
| donor | `831107 + family seed` | `833107 + family seed` |
| frame index | `841107 + family seed` | `843107 + family seed` |

Use new target-derangement streams:

| Family | Composition | Extrapolation |
| --- | ---: | ---: |
| constant speed | `921107 + seed` | `923107 + seed` |
| constant acceleration | `925107 + seed` | `927107 + seed` |

The mixed shuffled target concatenates the two within-family derangements.
Every primary frame-position count must be at least `400`.

## Frozen arms

Retain exactly the four predecessor arms:

1. `clean_law_specific`: all-increment speed mean or all-frame degree-2
   acceleration operator on clean observations;
2. `corrupted_law_specific`: the same law-specific operator naively applied to
   all corrupted observations;
3. `oracle_drop_one_quadratic`: the common quadratic chart after deleting the
   true corruption index;
4. `robust_drop_one_quadratic`: the sealed exhaustive minimum-residual delete-
   one quadratic estimator.

The robust arm must use the same function for both laws and may not read or
select a law label. No old prediction, threshold fit, or learned coefficient is
transferred; only the frozen algorithm is reused.

## Absolute and corrective endpoints

Retain the complete physical task gate:

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

For every speed, acceleration, and mixture cell, require:

```text
materiality:
  clean fixed ceiling passes
  corrupted naive fixed ceiling fails

Pareto repair:
  robust fixed ceiling passes
  robust / corrupted-naive RMSE <= .50
  robust accuracy delta >= -.005
  robust cross-entropy delta <= -.10

oracle fidelity:
  robust RMSE <= oracle RMSE + .002
  robust accuracy >= oracle accuracy - .01
  robust cross-entropy <= oracle cross-entropy + .005.
```

A paired replicate passes an endpoint only if every population passes under
both shifts. Require at least `4/5` paired replicates for the common absolute,
materiality, Pareto repair, and oracle-fidelity endpoints simultaneously.

## Controls and validity

- Every fresh base and corrupted cohort must regenerate bitwise exactly.
- Saturation must be zero; exact observable `C3` identity, composition, order,
  stored-action, latent-regeneration, and target-invariance contracts must pass.
- Donors and target shuffles must have zero fixed points; every frame-position
  count must pass.
- Corruption must commute exactly with global deck action, and every prediction
  arm must be deck invariant within `2e-12`.
- The minimum clean post-deletion phase-chart margin must be at least `.20`.
- Continuous degree-1/degree-2 inclusion error must be `<=1e-10`.
- On continuous corrupted carriers, oracle and robust predictions must recover
  exact time `8` within `1e-10`, and robust deletion must recover every hidden
  corruption index.
- Every shuffled-target arm must have absolute scalar correlation `<=.10`,
  scalar RMSE `>=.80`, and fail the complete task gate.
- All recorded values must be finite strict JSON.
- Old primary examples pooled, law-label reads, selector decisions, models,
  checkpoints, optimizer steps, changed parameters, and target-using fits must
  all be zero.

Any validity failure prevents a scientific conclusion.

## Locked classifications

Use the first matching valid row:

| Outcome | Classification | Program decision |
| --- | --- | --- |
| common absolute, materiality, Pareto repair, and oracle fidelity each `>=4/5` | `fresh_corrective_confirms_common_robust_nested_law_closure` | the original stays inconclusive, but fresh evidence closes selector/TinyLLM work on the nested family |
| corrupted naive mixture ceiling `>=4/5` | `fresh_hidden_order_corruption_not_material` | scope is too weak; do not train |
| oracle endpoint `>=4/5` but common absolute endpoint `<4/5` | `fresh_recoverable_scope_exceeds_common_fixed_estimator` | license a compact typed selector comparison, not unrestricted TinyLLM |
| oracle endpoint `<4/5` | `fresh_scope_not_recoverable_at_required_ceiling` | repair identifiability; do not train |
| any other valid combination | `inconclusive_hidden_order_corruption_corrective` | inspect the joint gate; do not tune on these cohorts |
| any validity failure | `invalid_hidden_order_corruption_corrective` | repair infrastructure only |

## Frozen sources

| Source | SHA-256 |
| --- | --- |
| hidden-order producing runner | `318cf81497960d37f86b1be58af3d819076e7dc9b620fe2ba73ae215ae0adea7` |
| hidden-order valid inconclusive result | `88e44e76c44d654331ea647ccedd8dad505030a65ab7c4ac241d8b98d98bd02e` |

The producing runner transitively pins the speed and acceleration generators,
group action, physical decoder, corruption implementation, common estimator,
and predecessor artifacts. The corrective implementation must call its source
validator and pin these two direct sources plus this preregistration before any
primary generation.

## Expected artifact and accounting

```text
data/experiments/tinyllm_c3_hidden_order_corruption_corrective/
  20260811_preregistered/result.json
```

```text
fresh base examples:                   81,920
fresh corrupted evaluations:          81,920
closed-form observation-only fits:   737,280
old primary examples pooled:               0
law-label reads / selector decisions:      0 / 0
models / checkpoints:                      0 / 0
optimizer steps / changed parameters:      0 / 0
target-using fits:                          0
```
