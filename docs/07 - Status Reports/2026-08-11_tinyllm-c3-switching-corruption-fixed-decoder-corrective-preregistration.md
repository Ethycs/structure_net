# TinyLLM C3 switching-law corruption fixed-decoder corrective preregistration

**Status:** FROZEN AFTER INVALID PRIMARY, BEFORE FRESH CORRECTIVE COHORT GENERATION

**Date:** 2026-08-11

**Conformance:** NAL-STD-EXPERIMENT `PROSPECTIVE FRESH-COHORT NUMERICAL CORRECTIVE`

**Hypothesis:** `tinyllm-c3-switching-corruption-fixed-decoder-corrective-v1`

## Preserved invalid predecessor

The first primary artifact is preserved unchanged at:

```text
data/experiments/tinyllm_c3_switching_corruption_fixed_decoder/
  20260811_preregistered/result.json
```

It is invalid and supplies no scientific classification. Two cells violated
the registered requirement that every estimator be deck invariant within
`2e-12`. The affected arm was the deliberately corrupted
`corrupted_known_switch_no_drop` comparator; its maximum errors were `.2846`
and `1.3097`. All exact group, corruption-equivariance, identifiability,
continuous-future, oracle, data, and shuffle contracts passed.

The cause is a numerical branch discontinuity. The analytic carrier is deck
invariant mathematically but differs at approximately floating-point summation
roundoff after channel permutation. When a corrupted adjacent phase difference
lies at the principal-angle boundary, that perturbation can select opposite
`+pi/-pi` unwraps and create an order-one downstream difference.

The invalid artifact's observed oracle, fixed-decoder, Pareto, and fidelity
counts are outcome-known. They are not reused, pooled, or allowed to set a
threshold in this corrective.

## Single corrective intervention

Before every estimator, canonicalize each complex invariant carrier with the
fixed observation-only map

```text
z <- round(real(z), 12) + i*round(imag(z), 12)
z <- z / max(abs(z), 1e-12).
```

Apply it identically to clean, corrupted, and deck-transformed carriers. The
maximum displacement from the original unit carrier must be `<=1e-12`.

No other implementation, estimator, support, arm, task endpoint, threshold,
classification, or exact identifiability contract changes. In particular:

- switch support remains `{2,3,4}`;
- the full `{2,3,4,5}` support must retain exact distance `2` and its explicit
  target-changing collision;
- the repaired support must retain exact distance `3`;
- the fixed decoder still enumerates the same `24` switch/deletion candidates
  and selects the same retained phase-residual objective;
- the old robust quadratic comparator, oracle, physical decoder, Pareto gate,
  and oracle-fidelity gate remain unchanged.

The correction is valid only if it restores every arm's deck invariance below
the original `2e-12` limit. It may not exempt the failed comparator.

## Fresh corrective population

Use five seeds disjoint from the invalid primary, every predecessor, and all
pilots:

```text
401, 419, 433, 449, 467
```

Generate `4,096` examples per seed and shift from new streams:

| Stream | Composition | Extrapolation |
| --- | ---: | ---: |
| base data and switch | `981107 + seed` | `983107 + seed` |
| donor derangement | `985107 + seed` | `987107 + seed` |
| corrupted frame | `989107 + seed` | `991107 + seed` |
| target derangement | `993107 + seed` | `995107 + seed` |

The generator ranges, switch and frame count floors, corruption law, and all
other controls are identical to the predecessor preregistration. Lifecycle
tests may use `64` examples from seed `997` and streams `997107 + seed` and
`999107 + seed`; they never satisfy a primary gate.

## Retained primary gates

Require at least `4/5` seeds to pass both composition and extrapolation jointly.

| Endpoint | Unchanged requirement |
| --- | --- |
| dynamics materiality | clean known-switch ceiling passes; clean global quadratic fails |
| corruption materiality | clean known-switch ceiling passes; corrupted known-switch no-drop fails |
| oracle recoverability | oracle switch/drop fixed ceiling passes |
| fixed closure | exhaustive fixed switch/drop ceiling passes |
| Pareto repair | fixed/global-corrupted RMSE ratio `<=.50`, accuracy delta `>=-.005`, CE delta `<=-.10` |
| oracle fidelity | fixed RMSE excess `<=.002`, accuracy delta `>=-.01`, CE excess `<=.005` |

The strong fixed ceiling remains scalar RMSE `<=.020`, exact-bin accuracy
`>=.90`, and the complete physical task gate. The task, action, continuous,
chart-margin, shuffle, determinism, count, finite-JSON, and zero-training
validity contracts remain unchanged.

## Locked classifications

Use the predecessor table verbatim with the suffix-free classifications:

| Outcome | Classification | Decision |
| --- | --- | --- |
| all six primary endpoints pass `>=4/5` with exact distance/collision contracts | `fixed_change_point_decoder_closes_identifiable_switching_corruption` | close TinyLLM on the repaired scope; retain late-switch impossibility |
| either global quadratic arm passes `>=4/5` | `switching_corruption_scope_not_material` | do not train |
| oracle passes `<4/5` | `identifiable_switching_not_recoverable_at_required_ceiling` | repair observation precision; do not train |
| oracle passes `>=4/5` but fixed closure passes `<4/5` | `recoverable_switching_exceeds_fixed_change_point_decoder` | license one compact typed continuation comparison only |
| any other valid combination | `inconclusive_switching_corruption_preflight` | preserve the joint failure; do not tune on these cohorts |
| any validity failure | `invalid_switching_corruption_corrective` | infrastructure repair only |

`tinyllm_training_licensed=false` in every row. Only the recoverable-fixed-
failure row may license a compact typed continuation comparison.

## Frozen source lineage

| Source | SHA-256 |
| --- | --- |
| invalid primary runner | `5ebe462c27989e0f09f38bba8ee0e885ea5fb76190bc86c1d7143d79376e2128` |
| invalid primary result | `2dd8f23a2eb7bd5ad7e2c224ee8c08201f648ce905e2fd7d806d7d676badf20c` |
| original preregistration | `d7e7b0cd3774a0b661e331c0c6bf56631734152e05fdc156584952f56d6b6ee2` |

The corrective implementation must call the predecessor source validator and
pin all three artifacts plus this preregistration before any fresh generation.

## Expected artifact and accounting

```text
data/experiments/tinyllm_c3_switching_corruption_fixed_decoder_corrective/
  20260811_preregistered/result.json
```

```text
fresh base examples:                         40,960
fresh corrupted evaluations:                 40,960
observation-only candidate fits:           1,720,320
continuous validation candidate fits:        983,040
invalid-primary examples pooled:                   0
models / checkpoints / optimizer steps:       0 / 0 / 0
changed parameters / target-using fits:        0 / 0
```
