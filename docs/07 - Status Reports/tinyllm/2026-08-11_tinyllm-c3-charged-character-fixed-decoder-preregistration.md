# TinyLLM C3 charged-character fixed-decoder preregistration

**Status:** FROZEN BEFORE FRESH PRIMARY COHORT GENERATION

**Date:** 2026-08-11

**Conformance:** NAL-STD-EXPERIMENT `PROSPECTIVE NO-TRAINING MECHANISTIC FIXED-DECODER REPLICATION`

**Hypothesis:** `tinyllm-c3-charged-character-fixed-decoder-v1`

## Decision question

The identifiable switching-law study established an oracle/fixed gap. The
invariant carrier `q_t = exp(i*3*theta_t)` is exact, but cubing the first
character triples phase noise before discrete switch/corruption selection.
Rare wrong chart choices then dominate outside-range RMSE.

This study asks:

> Does fitting the deck-charged first character, and cubing only the forecast,
> close the quantized change-point decoding gap without learning?

This is the last fixed-decoder falsifier before the licensed compact typed
continuation comparison. No TinyLLM, selector training, checkpoint, optimizer,
or reusable fitted parameter is permitted.

## Frozen charged-character construction

After calibration correction, compute the normalized first Fourier character

```text
c_t = sum_j x_tj exp(-2*pi*i*j/3)
c_t <- c_t / abs(c_t).
```

Under deck element `g`, it obeys the exact declared type

```text
c_t(gx) = exp(-2*pi*i*g/3) c_t(x).
```

For every candidate deletion, choose the first retained character `c_a` as an
anchor and form the invariant relative character

```text
r_t = c_t conjugate(c_a).
```

Canonicalize `r_t` and the invariant anchor `c_a^3` by rounding real and
imaginary components to twelve decimal places and renormalizing. Each
canonicalization displacement must be `<=1e-12`.

Unwrap the relative *first-character* phase and fit the same switching design

```text
X_s(t) = [1, t, max(t-s, 0)]
```

for all `24` switch/deletion candidates. Select the minimum retained phase
residual. If `p_8` is the fitted relative phase, emit

```text
q_hat_8 = canonicalize(c_a^3) exp(i*3*p_8).
```

Thus deck pose remains typed and cancels exactly, while change-point selection
operates before the noise-amplifying cubic map.

## Fresh population

Use five seeds disjoint from every predecessor, invalid run, corrective, and
pilot:

```text
487, 503, 521, 541, 563
```

Generate `4,096` examples per seed and shift with the exact repaired switching
support `{2,3,4}`, generator ranges, calibration, quantizer, one-frame donor
substitution, and physical target from the preceding valid corrective.

| Stream | Composition | Extrapolation |
| --- | ---: | ---: |
| base data and switch | `1001107 + seed` | `1003107 + seed` |
| donor derangement | `1005107 + seed` | `1007107 + seed` |
| corrupted frame | `1009107 + seed` | `1011107 + seed` |
| target derangement | `1013107 + seed` | `1015107 + seed` |

Every switch count must be at least `1,200`; every corrupted-frame count at
least `400`. Lifecycle tests may use `64` examples from seed `997` and streams
`1017107 + seed` and `1019107 + seed`; lifecycle outcomes are not evidence.

## Frozen arms

All candidate fits are per-example, observation-only, and retained nowhere.

1. `oracle_invariant_switch_drop`: cubed invariant carrier, true switch and
   corrupted-frame indices;
2. `fixed_invariant_switch_drop`: the registered `24`-candidate invariant-
   carrier residual selector;
3. `oracle_charged_switch_drop`: charged carrier, true switch and frame;
4. `fixed_charged_switch_drop`: charged carrier, minimum-residual selection over
   the same `24` candidates.

The two invariant arms are paired controls, not imported outcomes. Both decoder
families receive identical fresh observations, candidate identities, physical
decoder, and target shuffles.

## Primary endpoints

Retain the strong fixed ceiling:

```text
scalar RMSE <= .020
exact-bin accuracy >= .90
complete physical task gate passes.
```

The complete task gate remains:

| Endpoint | Composition | Extrapolation |
| --- | ---: | ---: |
| posterior-mean correlation | `>=.90` | `>=.90` |
| exact-bin accuracy | `>=.50` | `>=.35` |
| target cross-entropy | `<=1.80` | `<=2.20` |
| predicted-bin coverage | `>=14` | `>=12` |

Require each endpoint jointly on composition and extrapolation in at least
`4/5` seeds:

| Endpoint | Requirement |
| --- | --- |
| invariant oracle recoverability | oracle invariant ceiling passes |
| charged oracle recoverability | oracle charged ceiling passes |
| charged fixed closure | fixed charged ceiling passes |
| charged oracle fidelity | charged fixed RMSE excess `<=.002`, accuracy delta `>=-.01`, CE excess `<=.005` |

The invariant fixed pass count is a locked paired comparator. It does not enter
the charged absolute gate, but it determines whether the mechanism uniquely
repairs the predecessor gap.

## Mechanistic and validity contracts

- The exact identifiability audit must retain code distance `2` for
  `{2,3,4,5}`, distance `3` for `{2,3,4}`, and the token collision with target
  separation `>=.25`.
- Base, corruption, and shuffle streams must regenerate exactly; saturation is
  zero and derangements have zero fixed points.
- All exact `C3` observation/action/target contracts pass.
- The extracted charged carrier must satisfy
  `c(gx)=exp(-2*pi*i*g/3)c(x)` within `2e-12`.
- Corruption commutes exactly with global deck action.
- Every final prediction arm is deck invariant within `2e-12`.
- Every canonicalization displacement is `<=1e-12`.
- The minimum true-deletion phase-chart margin is `>=.20` for both invariant
  and charged fits.
- On continuous corrupted characters, both oracle and fixed charged forecasts
  equal the exact time-8 invariant carrier within `1e-10` on every example.
- Every shuffled-target arm has absolute scalar correlation `<=.10`, RMSE
  `>=.80`, and fails the complete task gate.
- All values are finite strict JSON.
- Models, checkpoints, optimizer steps, changed parameters, target-using fits,
  and reusable fits are zero.

Any validity failure prevents interpretation.

## Locked classifications

Use the first matching valid row:

| Outcome | Classification | Decision |
| --- | --- | --- |
| both oracles, charged fixed closure, and charged fidelity each `>=4/5`; invariant fixed `<4/5` | `charged_character_fixed_decoder_closes_invariant_quantization_gap` | close learned continuation work; promote charged fixed decoder |
| both oracles, charged fixed closure, charged fidelity, and invariant fixed each `>=4/5` | `both_fixed_decoders_close_fresh_switching_scope` | no unique charged advantage, but no learned job remains |
| charged oracle `>=4/5` and charged fixed `<4/5` | `recoverable_switching_exceeds_charged_fixed_decoder` | retain license for one compact typed chart-mixture comparison |
| either oracle `<4/5` | `fresh_switching_scope_not_oracle_recoverable` | repair observation precision; do not train |
| any other valid combination | `inconclusive_charged_character_fixed_decoder` | inspect joint gate without tuning on these cohorts |
| any validity failure | `invalid_charged_character_fixed_decoder` | infrastructure repair only |

`tinyllm_training_licensed=false` in every row. Only
`recoverable_switching_exceeds_charged_fixed_decoder` may set
`compact_typed_continuation_comparison_licensed=true`.

## Disclosed outcome-known audit

Before freezing this document, the charged algorithm above was executed on the
five sealed corrective seeds `401,419,433,449,467`. It passed the strong
ceiling in `5/5` and numerically matched its charged oracle, including all
former extrapolation failures. This is exploratory mechanism-selection
evidence only. It is not pooled, cannot satisfy a fresh gate, and did not set
any threshold; all thresholds predate the audit.

## Frozen source lineage

| Source | SHA-256 |
| --- | --- |
| switching corrective runner | `452b0fa8f8ff54ddb0afa98e32eef9dc38da9f240eccf6a867386d9b65197939` |
| switching valid result | `20d40a54ce42a904766cab9eb533f6e2baccff79f75a2703e6bd57ff9638675b` |
| switching report | `bb082ec3965d5e8badb3e08d645a0fb958990b8c65667f2da8440cbe6f02295a` |

The implementation must call the source validator and pin all three artifacts
plus this preregistration before fresh generation.

## Expected artifact and accounting

```text
data/experiments/tinyllm_c3_charged_character_fixed_decoder/
  20260811_preregistered/result.json
```

```text
fresh base examples:                         40,960
fresh corrupted evaluations:                 40,960
observation-only candidate fits:           1,966,080
continuous validation candidate fits:      1,966,080
outcome-known audit examples pooled:                0
models / checkpoints / optimizer steps:       0 / 0 / 0
changed parameters / target-using fits:        0 / 0
```
