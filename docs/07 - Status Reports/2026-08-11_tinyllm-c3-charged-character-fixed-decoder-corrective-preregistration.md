# TinyLLM C3 charged-character fixed-decoder corrective preregistration

**Status:** FROZEN BEFORE FRESH CORRECTIVE COHORT GENERATION

**Date:** 2026-08-11

**Conformance:** NAL-STD-EXPERIMENT `NUMERICAL IMPLEMENTATION CORRECTIVE — PROSPECTIVE NO-TRAINING REPLICATION`

**Hypothesis:** `tinyllm-c3-charged-character-fixed-decoder-corrective-v1`

## Why a corrective is required

The first charged-character primary is preserved as invalid. Its runner and
result are immutable:

| Artifact | SHA-256 |
| --- | --- |
| preregistration | `9f1a6b280498f254e51324c8d87d385fa5ab72090cc2487db57d5f84b8925633` |
| runner | `009c23bf287cf233f16d3a40094c011f546c8982c14706ab59b6ddcdc818e5a3` |
| invalid result | `0cf9bd7ca8e22bb5712453aab679dfa6aa76e067fd2e242c07874b42b854264a` |

The result was scientifically uninterpretable because seven of ten cells
failed the declared final-action tolerance. The charged extractor itself was
equivariant to at most `2.025e-15`; every candidate selection was identical
under both nonidentity deck actions; and all endpoint, shuffle, continuous,
identifiability, and data contracts otherwise passed. The only failure was a
`2.568e-12` to `5.113e-12` difference in final charged predictions against a
`2e-12` limit.

The cause is numerical. Twelve-decimal floating rounding can send two values
that differ by machine epsilon to adjacent rounding bins. The switching fit
then extrapolates that `~1e-12` difference and the final cubic phase map triples
it. Relaxing the tolerance would conceal the defect. The corrective instead
computes the same `C3` charged invariants in exact integer coordinates before
entering floating phase arithmetic.

The invalid outcome would have produced both fixed decoder ceilings and
charged fidelity in `5/5`. Those outcomes are disclosed but are not evidence,
are not pooled, and do not satisfy any corrective gate.

## Frozen exact charged construction

For each integer token triple `(k_0,k_1,k_2)`, represent twice the first
Fourier character in Eisenstein coordinates:

```text
a = 2*k_0 - k_1 - k_2
b = k_2 - k_1
z = a + i*sqrt(3)*b.
```

The common quantizer offset cancels and its positive scale does not change
phase, so this is the same normalized first character used in the invalid
runner. Do not fit or estimate a calibration parameter.

For every candidate deletion, let `(a_a,b_a)` be the first retained anchor.
Form the relative character exactly in signed 64-bit integers:

```text
A_t = a_t*a_a + 3*b_t*b_a
B_t = b_t*a_a - a_t*b_a
r_t proportional to A_t + i*sqrt(3)*B_t.
```

Form the invariant anchor cube exactly:

```text
A_a = a_a^3 - 9*a_a*b_a^2
B_a = 3*a_a^2*b_a - 3*b_a^3
c_a^3 proportional to A_a + i*sqrt(3)*B_a.
```

Under a global deck action, both integer pairs are exactly unchanged. Normalize
them only after these exact products, apply the registered twelve-decimal
canonicalization and renormalization, unwrap the relative phase, and fit the
same `24` designs

```text
X_s(t) = [1, t, max(t-s,0)].
```

Select minimum retained phase residual and emit the unchanged physical rule

```text
q_hat_8 = canonicalize(c_a^3) exp(i*3*p_8).
```

No target, reusable fit, learned selector, model, or checkpoint is permitted.

## Fresh corrective population

Use five seeds disjoint from every predecessor, invalid run, corrective, and
lifecycle cohort:

```text
577, 593, 613, 631, 653
```

Generate `4,096` examples per seed and shift with switching support `{2,3,4}`
and the frozen generator, quantizer, calibration, target, and one-frame donor
substitution.

| Stream | Composition | Extrapolation |
| --- | ---: | ---: |
| base data and switch | `1021107 + seed` | `1023107 + seed` |
| donor derangement | `1025107 + seed` | `1027107 + seed` |
| corrupted frame | `1029107 + seed` | `1031107 + seed` |
| target derangement | `1033107 + seed` | `1035107 + seed` |

Every switch count must be at least `1,200`; every corrupted-frame count at
least `400`. Lifecycle tests may use `64` examples from seed `1009` and streams
`1037107 + seed` and `1039107 + seed`; lifecycle outcomes are not evidence.

## Frozen arms and endpoints

Retain the four paired arms without importing outcomes:

1. `oracle_invariant_switch_drop`;
2. `fixed_invariant_switch_drop`;
3. `oracle_charged_switch_drop`, using exact charged arithmetic and the true
   switch/frame indices;
4. `fixed_charged_switch_drop`, using exact charged arithmetic and the same
   minimum-residual `24`-candidate selector.

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

Require each endpoint jointly on both shifts in at least `4/5` seeds:

| Endpoint | Requirement |
| --- | --- |
| invariant oracle recoverability | oracle invariant ceiling passes |
| charged oracle recoverability | oracle charged ceiling passes |
| charged fixed closure | fixed charged ceiling passes |
| charged oracle fidelity | charged fixed RMSE excess `<=.002`, accuracy delta `>=-.01`, CE excess `<=.005` |

The invariant fixed pass count remains a locked paired comparator.

## Corrective validity contracts

- The invalid preregistration, runner, and result hashes must match the table
  above, and the invalid result must remain `status=invalid`.
- The predecessor's exact identifiability audit must pass unchanged.
- Dataset, corruption, and shuffle replay, coverage, saturation, derangement,
  exact `C3` group, and shuffle controls remain unchanged.
- Direct floating extraction retains
  `c(gx)=exp(-2*pi*i*g/3)c(x)` within `2e-12` as a diagnostic.
- For the exact charged implementation, the relative and anchor-cube integer
  pairs must be bit-identical under both nonidentity deck actions.
- All four final predictions must be deck invariant within `2e-12`.
- Every canonicalization displacement is `<=1e-12`; both chart margins remain
  `>=.20`.
- On continuous exact characters, oracle and fixed forecasts equal the exact
  time-8 invariant carrier within `1e-10`.
- Every shuffled-target arm has absolute correlation `<=.10`, RMSE `>=.80`,
  and fails the complete task gate.
- All values are finite strict JSON; all model/training/target-fit accounting
  remains zero.

Any validity failure prevents scientific interpretation.

## Locked classifications

Use the first matching valid row:

| Outcome | Classification | Decision |
| --- | --- | --- |
| both oracles, charged fixed closure, and charged fidelity each `>=4/5`; invariant fixed `<4/5` | `exact_charged_character_corrective_closes_invariant_quantization_gap` | close learned continuation work; promote the exact charged decoder |
| both oracles, charged fixed closure, charged fidelity, and invariant fixed each `>=4/5` | `both_exact_charged_and_invariant_fixed_decoders_close_fresh_switching_scope` | no unique charged advantage and no learned job remains |
| charged oracle `>=4/5` and charged fixed `<4/5` | `recoverable_switching_exceeds_exact_charged_fixed_decoder` | retain license for one compact typed chart-mixture comparison |
| either oracle `<4/5` | `fresh_corrective_switching_scope_not_oracle_recoverable` | repair observation precision; do not train |
| any other valid combination | `inconclusive_exact_charged_character_corrective` | inspect the joint gate without tuning |
| any validity failure | `invalid_exact_charged_character_corrective` | infrastructure repair only |

`tinyllm_training_licensed=false` in every row. Only
`recoverable_switching_exceeds_exact_charged_fixed_decoder` may license the
compact typed continuation comparison.

## Expected artifact and accounting

```text
data/experiments/
  tinyllm_c3_charged_character_fixed_decoder_corrective/
    20260811_preregistered/result.json
```

```text
fresh base examples:                         40,960
fresh corrupted evaluations:                 40,960
observation-only candidate fits:           1,966,080
continuous validation candidate fits:      1,966,080
invalid-primary examples pooled:                    0
models / checkpoints / optimizer steps:       0 / 0 / 0
changed parameters / target-using fits:        0 / 0
```
