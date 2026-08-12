# TinyLLM C3 charged-character fixed-decoder result

**Status:** VALID FRESH CORRECTIVE RESULT — CHARGED FIXED DECODER CLOSES THE INVARIANT-CARRIER GAP

**Date:** 2026-08-11

**Hypothesis:** `tinyllm-c3-charged-character-fixed-decoder-corrective-v1`

**Classification:** `exact_charged_character_corrective_closes_invariant_quantization_gap`

**Preregistrations:** [original charged-character study](../07%20-%20Status%20Reports/2026-08-11_tinyllm-c3-charged-character-fixed-decoder-preregistration.md); [exact-arithmetic corrective](../07%20-%20Status%20Reports/2026-08-11_tinyllm-c3-charged-character-fixed-decoder-corrective-preregistration.md)

**Primary artifact:** `data/experiments/tinyllm_c3_charged_character_fixed_decoder_corrective/20260811_preregistered/result.json`

## Verdict

The replicated switching-law gap closes when discrete chart selection is
performed in the deck-charged first character and the forecast is cubed only
afterward:

```text
invariant oracle recoverability:  5/5
charged oracle recoverability:    5/5
fixed invariant closure:          3/5
fixed charged closure:            5/5
charged oracle fidelity:          5/5
required:                        >=4/5
```

All ten fresh cells are valid. The exact charged implementation has zero final
deck-action error in every charged arm. The registered classification closes
the compact typed continuation branch: no learned selector, TinyLLM model, or
additional optimizer campaign is licensed on this task.

This is a positive result for symmetry typing, not for model capacity:

> Quotienting before a discrete inference step can amplify quantization noise.
> Preserve the equivariant character through chart selection and project to the
> invariant target only at the output.

## The invalid primary remains invalid

The first fresh charged run is preserved and contributes zero evidence. Seven
of ten cells exceeded the `2e-12` final-action bound, with charged prediction
differences from `2.568e-12` to `5.113e-12`. Its direct charged extractor was
equivariant within `2.025e-15`, and candidate identities did not change under
either nonidentity deck action.

The defect arose after twelve-decimal canonicalization. Mathematically
identical values on opposite sides of a floating rounding boundary became
`~1e-12` apart; switching extrapolation and the final cubic phase map amplified
the discrepancy. Although both fixed arms would have passed `5/5`, the action
failure makes that artifact scientifically uninterpretable. No threshold was
relaxed and no outcome was imported into the corrective.

## Exact corrective

For each integer token triple, the corrective represents the charged first
character in Eisenstein coordinates:

```text
a = 2*k_0 - k_1 - k_2
b = k_2 - k_1
z = a + i*sqrt(3)*b.
```

It forms the relative character and anchor cube as exact signed-integer
polynomials:

```text
relative = (a_t*a_a + 3*b_t*b_a)
         + i*sqrt(3)*(b_t*a_a - a_t*b_a)

anchor^3 = (a_a^3 - 9*a_a*b_a^2)
         + i*sqrt(3)*(3*a_a^2*b_a - 3*b_a^3).
```

These neutral products are bit-identical after a global `C3` deck action. Only
then are they normalized and converted to floating phase. This is the same
charged character and the same `24` switch/deletion candidates as the invalid
runner; it changes numerical representation, not the hypothesis or endpoint.

The fresh corrective used seeds `577,593,613,631,653`, disjoint generator,
donor, frame, and shuffle streams, and no examples from the invalid primary.

## Primary results

Means over the five fresh seeds:

| Arm | Composition RMSE / accuracy | Extrapolation RMSE / accuracy | Joint seed ceiling |
| --- | ---: | ---: | ---: |
| oracle invariant switch/drop | `.004121 / .9770` | `.004393 / .9798` | `5/5` |
| fixed invariant switch/drop | `.004152 / .9769` | `.012715 / .9797` | `3/5` |
| oracle charged switch/drop | `.004121 / .9770` | `.004393 / .9798` | `5/5` |
| fixed charged switch/drop | `.004127 / .9770` | `.004410 / .9798` | `5/5` |

The two oracle arms agree because the invariant and charged coordinates contain
the same exact physical information once the discrete chart is supplied. The
fixed arms differ only in which coordinate is used to choose that chart.

Per-seed outside-range results show the tail repair:

| Seed | Fixed invariant RMSE / accuracy | Fixed charged RMSE / accuracy | Invariant ceiling | Charged ceiling |
| ---: | ---: | ---: | ---: | ---: |
| 577 | `.026088 / .9778` | `.004541 / .9778` | fail | pass |
| 593 | `.024314 / .9822` | `.004357 / .9824` | fail | pass |
| 613 | `.004422 / .9805` | `.004406 / .9805` | pass | pass |
| 631 | `.004437 / .9795` | `.004433 / .9795` | pass | pass |
| 653 | `.004315 / .9788` | `.004314 / .9790` | pass | pass |

No result is rescued by averaging: the locked joint seed count itself moves
from `3/5` to `5/5`.

## Mechanistic interpretation

The charged decoder does not recover dramatically more latent switch/frame
labels. Mean selected-oracle agreement changes only modestly:

| Selector | Composition | Extrapolation |
| --- | ---: | ---: |
| invariant carrier | `.9529` | `.9547` |
| charged character | `.9553` | `.9582` |

Instead, it changes the geometry in which small token errors are resolved. The
minimum true-deletion unwrap margin is `2.38575` radians for the charged
character, compared with `.87406` for the cubed invariant carrier. Cubing
triples phase perturbations before the invariant selector chooses among nearby
discrete charts. The charged selector delays that noise-amplifying map until
after the chart-conditioned forecast.

This explains the otherwise unusual endpoint pattern: invariant and charged
exact-bin accuracies remain nearly identical, but rare invariant-carrier chart
errors create enough large outside-range scalar errors to cross the RMSE
ceiling in two seeds. The charged coordinate removes that tail while matching
its oracle in all five seeds.

The result supports a precise design rule:

```text
equivariant carrier
    -> discrete nuisance/chart inference
    -> invariant projection
    -> physical task decoder
```

It does not support exposing an untyped representation to a learned model or
discarding the group charge at the earliest possible layer.

## Contracts and accounting

| Contract | Result |
| --- | ---: |
| fresh requested/completed/invalid cells | `10 / 10 / 0` |
| fresh base / corrupted examples | `40,960 / 40,960` |
| invalid-primary examples pooled | `0` |
| exact relative/cube integer-action errors | `0` |
| maximum charged final-action error | `0.0` |
| maximum action error across every arm | `1.459e-12` |
| maximum direct charged equivariance error | `2.071e-15` |
| maximum continuous forecast error | `5.181e-12` |
| maximum canonicalization displacement | `7.055e-13` |
| minimum invariant / charged chart margin | `.87406 / 2.38575` |
| maximum shuffled absolute correlation | `.03560` |
| minimum shuffled RMSE | `.98382` |
| models / checkpoints / optimizer steps | `0 / 0 / 0` |
| reusable or target-using fits | `0 / 0` |

All exact identifiability, generator replay, switch/frame coverage, saturation,
derangement, corruption commutation, group action, target invariance, task,
shuffle, strict-JSON, and finite-value contracts pass.

## Program decision

Close both contemplated learned branches for this generator:

- `compact_typed_continuation_comparison_licensed=false`;
- `tinyllm_training_licensed=false`.

The preceding oracle/fixed gap was real, but it was a representation-ordering
problem, not evidence that trainable capacity was required. The shortest
decisive path found the missing operation without loading a checkpoint.

A future experiment must change the scientific scope rather than retune this
one—for example, an unknown or noncyclic group, partial group observations, or
a dynamics family whose admissible charged sufficient statistic is not known
analytically. Any such study requires a new identifiability audit and fixed
ceiling before training.

## Reproduction and provenance

```bash
MPLCONFIGDIR=/tmp/mpl-charged-corrective-primary \
pixi run python -m \
  experiments.structure_net.tinyllm_c3_charged_character_fixed_decoder_corrective
```

| Artifact | SHA-256 |
| --- | --- |
| valid corrective result | `d8076bd7bfc42819a17438e188300e8ea0131dcc72d9ceac0bee39ede72fcaaf` |
| corrective runner | `5fcd691d3fd910a619bd61aa8dd0432e0050efc44699657ec98f5d7e2e01de97` |
| corrective preregistration | `1fa97f74a0f88c0016a8b0309f1a392f9b8d45d545cdb0e39bb7754ecafe2020` |
| preserved invalid result | `0cf9bd7ca8e22bb5712453aab679dfa6aa76e067fd2e242c07874b42b854264a` |
| preserved invalid runner | `009c23bf287cf233f16d3a40094c011f546c8982c14706ab59b6ddcdc818e5a3` |
| original preregistration | `9f1a6b280498f254e51324c8d87d385fa5ab72090cc2487db57d5f84b8925633` |

The producing tests revalidate the complete frozen lineage, invalid-primary
diagnosis, fresh population, exact Eisenstein action identities, continuous
forecast, all primary gates, conservative classification, accounting, and
authoritative artifact hashes.
