# TinyLLM C3 time-varying gauge-jump fixed-decoder preregistration

**Status:** FROZEN BEFORE FRESH PRIMARY COHORT GENERATION

**Date:** 2026-08-11

**Conformance:** NAL-STD-EXPERIMENT `PROSPECTIVE NO-TRAINING TIME-VARYING-GAUGE FIXED-DECODER TEST`

**Hypothesis:** `tinyllm-c3-gauge-jump-corruption-fixed-decoder-v1`

## Decision question

The global-gauge switching study showed that a fixed decoder closes the
quantized oracle gap when it retains the deck-charged first character through
discrete chart selection and cubes only the forecast. That result assumes one
coherent deck frame over the full observed sequence.

This study changes the observation law:

> Does one hidden within-sequence `C3` frame jump break the charged fixed
> decoder, or does an exact discrete connection extend fixed closure without
> learning?

This is a new time-varying-gauge scope. Physical switching dynamics, target,
quantizer, calibration, corruption, and task decoder remain fixed. No
checkpoint, TinyLLM, learned selector, optimizer, reusable fitted parameter, or
target-using fit is permitted.

## Identifiability contract

Let the latent physical switching sequence be `theta_t` and let a nonzero deck
element `g` act on every observed frame from hidden time `tau` onward:

```text
c'_t = c_t                              for t < tau
c'_t = exp(-2*pi*i*g/3) c_t             for t >= tau.
```

The cubic invariant is unchanged framewise:

```text
(c'_t)^3 = c_t^3.
```

Consequently the hidden gauge jump does not change the invariant observation
code. After one arbitrary frame corruption, target identifiability inherits
the exact switching result:

```text
switch support {2,3,4,5}: minimum future code distance 2, not correctable
switch support {2,3,4}:   minimum future code distance 3, correctable.
```

Retain support `{2,3,4}`. The implementation must rerun the exact rational
distance audit and explicit late-support token collision. It must also verify
that applying and exactly undoing every sampled suffix action restores the
token tensor and that the analytic cubic carrier is unchanged within `2e-12`.

## Frozen observation change

Every example has:

- one hidden jump time `tau` sampled uniformly from `{1,2,3,4,5,6}`;
- one hidden nonidentity element `g` sampled uniformly from `{1,2}`;
- the existing independent global deck pose;
- one unmarked same-time donor-frame substitution after the jump action.

The suffix convention is inclusive: frames `t >= tau` are rolled by `g`.
Jump time `6` retains two post-jump observations. The target remains the
physical time-8 `cos(3*theta_8)` and never sees either deck label.

## Frozen exact connection decoder

For every candidate `(tau,g)`, undo that suffix deck action on the observed
tokens. Then run the frozen exact-Eisenstein charged decoder over all `24`
switch/deletion candidates:

```text
6 jump times * 2 nonidentity elements * 8 deletions * 3 switches
= 288 candidates per example.
```

Select the candidate with minimum retained first-character phase residual.
Use the candidate-conditioned charged forecast and cube only the output. All
neutral relative products and anchor cubes remain exact integer operations
before floating phase fitting.

This is an exhaustive fixed estimator over the declared one-jump law. Candidate
labels are not endpoints: alternative labels may represent the same physical
future. Hidden-label agreement is descriptive only.

## Fresh population

Use five seeds disjoint from every predecessor, corrective, and lifecycle
cohort:

```text
673, 691, 709, 727, 751
```

Generate `4,096` examples per seed and shift.

| Stream | Composition | Extrapolation |
| --- | ---: | ---: |
| base data and physical switch | `1047107 + seed` | `1049107 + seed` |
| donor derangement | `1051107 + seed` | `1053107 + seed` |
| corrupted frame | `1055107 + seed` | `1057107 + seed` |
| gauge-jump time | `1059107 + seed` | `1061107 + seed` |
| gauge-jump element | `1063107 + seed` | `1065107 + seed` |
| target derangement | `1067107 + seed` | `1069107 + seed` |

Every switch count must be at least `1,200`, every corrupted-frame count at
least `400`, every jump-time count at least `580`, every jump-element count at
least `1,800`, and every jump-time/element cell at least `280`. Lifecycle tests
may use `64` examples from seed `1019` and base-data streams `1071107 + seed`
and `1073107 + seed`; lifecycle outcomes are not evidence.

## Frozen arms

All candidate fits are per-example, observation-only, and retained nowhere.

1. `oracle_invariant_switch_drop`: cubic invariant, true physical switch and
   corrupted-frame indices; no gauge label is needed;
2. `fixed_invariant_switch_drop`: registered `24`-candidate invariant selector;
3. `fixed_charged_no_connection`: exact charged selector over only the `24`
   switch/deletion candidates, without correcting the hidden jump;
4. `oracle_charged_connection`: exact charged bank with true switch, deletion,
   jump time, and jump element;
5. `fixed_charged_connection`: minimum-residual selection over all `288`
   connection/switch/deletion candidates.

The invariant arms are jump-immune paired comparators. The no-connection arm
is the observation-law materiality control.

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
| charged connection oracle recoverability | oracle charged ceiling passes |
| fixed connection closure | fixed charged connection ceiling passes |
| fixed connection oracle fidelity | RMSE excess `<=.002`, accuracy delta `>=-.01`, CE excess `<=.005` |
| gauge-jump materiality/repair | no-connection arm fails and fixed connection passes both shifts |

The invariant fixed pass count is a locked paired comparator. It determines
whether the connection-aware charged method uniquely repairs the inherited
quantization tail.

## Validity and mechanistic contracts

- Base, jump, corruption, and shuffle streams regenerate exactly; saturation
  is zero and both derangements have zero fixed points.
- All exact global `C3` observation, action, and target contracts pass.
- Applying then undoing the sampled suffix jump restores all tokens exactly.
- Suffix jumps commute with the global `C3` action and corruption commutes with
  that global action.
- Exact connection-relative and anchor-cube integer pairs are bit-identical
  under both global nonidentity actions.
- Every charged connection prediction is globally deck invariant exactly;
  every other final arm is invariant within `2e-12`.
- Every canonicalization displacement is `<=1e-12`; invariant and connected
  charged true-chart margins are each `>=.20`.
- On exact continuous characters with the same hidden jump and corruption,
  both oracle and fixed connection forecasts equal the exact time-8 invariant
  carrier within `1e-10` on every example.
- Every shuffled-target arm has absolute correlation `<=.10`, RMSE `>=.80`,
  and fails the complete task gate.
- All values are finite strict JSON; model, checkpoint, optimizer, changed-
  parameter, target-fit, and reusable-fit counts are zero.

Any validity failure prevents interpretation.

## Locked classifications

Use the first matching valid row:

| Outcome | Classification | Decision |
| --- | --- | --- |
| both oracles, fixed connection, connection fidelity, and material repair each `>=4/5`; invariant fixed `<4/5` | `discrete_c3_connection_closes_time_varying_gauge_and_invariant_gap` | promote the fixed connection; close learned work |
| same charged gates pass and invariant fixed `>=4/5` | `discrete_c3_connection_closes_gauge_scope_without_unique_invariant_advantage` | fixed methods close the new scope; no learned work |
| charged oracle `>=4/5` but fixed connection `<4/5` | `recoverable_time_varying_gauge_exceeds_fixed_connection_decoder` | license one compact typed gauge/physical chart-mixture comparison |
| either oracle `<4/5` | `time_varying_gauge_scope_not_oracle_recoverable` | repair observation precision; do not train |
| any other valid combination | `inconclusive_time_varying_gauge_fixed_decoder` | inspect the joint gate without tuning |
| any validity failure | `invalid_time_varying_gauge_fixed_decoder` | infrastructure repair only |

`tinyllm_training_licensed=false` in every row. Only
`recoverable_time_varying_gauge_exceeds_fixed_connection_decoder` may set
`compact_typed_connection_comparison_licensed=true`.

## Disclosed outcome-known audit

Before freezing this document:

- a 64-example lifecycle cohort in each shift exercised all six jump times and
  both elements; fixed connection and oracle predictions passed, global action
  selection differences were zero, and charged action error was exactly zero;
- the prior difficult seed-577 extrapolation cohort was reused only as a
  stress test with a newly sampled gauge jump. No connection produced RMSE
  `.876842`; the invariant fixed arm produced `.026088`; charged oracle and
  fixed connection produced `.004465` and `.004518`. Fixed connection selected
  the exact hidden tuple only `.7871` of the time, confirming that label
  recovery is not the endpoint.

These outcomes selected the fixed algorithm but do not set thresholds, are not
pooled, and cannot satisfy a fresh gate.

## Frozen source lineage

| Source | SHA-256 |
| --- | --- |
| exact charged corrective runner | `5fcd691d3fd910a619bd61aa8dd0432e0050efc44699657ec98f5d7e2e01de97` |
| exact charged corrective result | `d8076bd7bfc42819a17438e188300e8ea0131dcc72d9ceac0bee39ede72fcaaf` |
| exact charged corrective report | `9acd88cf100b67f04a27c5415dcbce755b97c3545d6ba78ce114e97885c5ba10` |

The implementation must validate all three plus this preregistration before
fresh generation.

## Expected artifact and accounting

```text
data/experiments/tinyllm_c3_gauge_jump_corruption_fixed_decoder/
  20260811_preregistered/result.json
```

```text
fresh base examples:                         40,960
fresh corrupted evaluations:                 40,960
primary observation-only candidate fits: 13,762,560
continuous validation candidate fits:    12,779,520
outcome-known audit examples pooled:                0
models / checkpoints / optimizer steps:       0 / 0 / 0
changed parameters / target-using fits:        0 / 0
```
