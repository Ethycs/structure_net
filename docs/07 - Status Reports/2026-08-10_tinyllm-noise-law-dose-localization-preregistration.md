# TinyLLM sensor-noise law dose localization preregistration

**Status:** PREREGISTERED  
**Date:** 2026-08-10  
**Hypothesis:** `tinyllm-noise-law-dose-localization-v1`  
**Schema:** `nal.tinyllm-noise-law-dose-localization.v1`  
**Evidence role:** `preregistered_post_outcome_corrective_frozen_dose_localization`

## Why this is a corrective study

The preceding registered sensor-noise study used one planar RMS scale,
`sigma=0.05`. Its analytic calibrated arm passed the isotropic natural-utility
gate in `0/5` seeds, so the intended asymmetric-law comparison was invalid.
The `sigma=0.05` results are known, remain immutable, and are not fresh evidence
in this study.

This follow-up does not tune the failed primary. It asks the narrower question
selected by that failure:

> Is there a common nonzero dose at which isotropic error preserves natural
> utility in both structured arms, and, if so, do matched-energy asymmetric
> laws preserve the same frozen quotient computation at that locked dose?

No model, front end, observer, task head, action, noise model, or threshold may
be trained or fitted.

## Frozen sources

- preceding campaign:
  `data/experiments/tinyllm_noise_law_observed_twirl/20260810_d10_preregistered/campaign_results.json`
- preceding campaign SHA-256:
  `868ad0ffee546f157e701790c34a83f20bfb3116e78b2f8c5bc34dd7bfe660d7`
- preceding error-array file:
  `data/experiments/tinyllm_noise_law_observed_twirl/20260810_d10_preregistered/noise_law_arrays.npz`
- error-array file SHA-256:
  `d3771eac8e29f7940df7feaedebe74a5a78fb273cda2e70928c9be9e37ff3ba6`
- error-array content SHA-256:
  `93df61bc76ed073ea241c9450e7ec3523e7a98b5ac06e58d7e920a5df07d70aa`
- preceding runner SHA-256:
  `7bed49c064e8a2148268d2a4ab3a42ec70847a15d83c7297cff3d9dccc7970d2`
- DVC root containing the source:
  `19f1fbbe86b6b9235eb211a88bb32aa2.dir`
- lakeFS source commit:
  `f3c895cdf8d5f25e8ae6a87b3f694d0bbacb24cdd14d4736d0c7dfa41399c130`

The ten retained d8/N3 systems, seeds `(7, 17, 29, 41, 53)`, 512-example
composition and extrapolation cohorts, token decoding, calibration packets,
task, minibatching, task metrics, and all utility/intervention thresholds are
unchanged. The two arms remain:

1. `analytic_calibrated`;
2. `learned_calibrated_equivariant`.

## Nested dose ladder

Let `epsilon_law` denote one frozen `sigma=0.05` error array from the preceding
study. Evaluate the exact scaled arrays

`epsilon_law(s) = s * epsilon_law`

at the ordered multipliers

`S = (0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 1.0)`.

The corresponding sigma values are

`(0, 0.00625, 0.0125, 0.01875, 0.025, 0.03125, 0.0375, 0.05)`.

Scaling must be elementwise from the stored arrays. Generating new random
draws is forbidden. The runner must verify both source hashes and exact scaled
array construction. Expected planar squared norm scales as `s^2`; law shape
and reflection defects are otherwise unchanged for every positive multiplier.

## Stage 1: isotropic-only localization

At every multiplier, evaluate only the natural noisy identity under the
isotropic law. Do not evaluate or inspect anisotropic or biased task outcomes
during selection.

For each arm, seed, dose, and shift, retain the existing natural-utility gate:

- exact-bin accuracy loss from the clean identity `<= 0.05`;
- circular-error increase `<= pi/16`;
- target cross-entropy increase `<= 0.10`.

A seed passes a dose only when all three metrics pass on both composition and
extrapolation. An arm passes a dose at `>=4/5` seeds.

The zero-dose integrity control must reproduce the clean posterior within
`2e-6` and pass in `5/5` seeds per arm. Failure invalidates the campaign.

Define a positive multiplier as **prefix-valid** when both arms pass that
multiplier and every smaller positive registered multiplier. The selected dose
is the largest prefix-valid multiplier. This rule is deterministic and uses no
asymmetric-law outcome.

If no positive multiplier is prefix-valid, stop before Stage 2 and classify
the shape comparison as `no_common_nonzero_utility_window`. Do not weaken the
endpoint or insert another dose.

## Stage 2: one locked law-shape comparison

If Stage 1 selects a nonzero dose, evaluate exactly that multiplier under the
three frozen laws:

- `isotropic`;
- `lab_anisotropic`;
- `lab_biased`.

At `pre_block` and `full`, retain the existing correct observed `C2` action,
within-example Reynolds twirl, orthogonal target-changing action, and
orthogonal twirl. The correct action and twirl are compared with the noisy
identity, using the unchanged intervention gate:

- accuracy loss `<= 0.03`;
- circular-error increase `<= pi/16`;
- target cross-entropy increase `<= 0.10`.

A seed/law joint gate requires:

1. natural utility on both shifts;
2. correct action and correct twirl at both cuts on both shifts;
3. source/cut replay, analytic feature invariance, finite outputs, unchanged
   frozen state, and exact source provenance.

Each arm/law must pass in at least `4/5` seeds. Every orthogonal control must
pass in at most `1/5` seeds.

## Registered classifications

Classification is ordered:

1. `invalid_integrity` if source, array, replay, finite-state, or frozen-state
   contracts fail;
2. `invalid_zero_dose_control` if the zero-dose replay gate fails;
3. `no_common_nonzero_utility_window` if Stage 1 selects no positive dose;
4. `nonspecific_target_changing_control` if any orthogonal control exceeds
   `1/5` at the selected dose;
5. `isotropic_closure_fails_at_selected_dose` if the selected-dose isotropic
   joint gate misses `4/5` in either arm despite passing Stage 1 natural utility;
6. `asymmetric_law_breaks_within_isotropic_window` if isotropic passes both
   arms but an asymmetric arm/law misses `4/5`;
7. `additive_noise_closed_at_selected_dose` if all six arm/law populations
   pass.

The primary hypothesis passes only under classification 7. Classifications 3
through 6 are valid negative or narrowing outcomes when their preceding
integrity controls pass; they do not confirm the primary.

## Interpretation boundaries

- Stage 1 localizes tolerance for these exact frozen draws; it does not estimate
  a population noise radius.
- Stage 2 is a paired law-shape comparison at one outcome-informed dose, not an
  independent replication of the original `sigma=0.05` experiment.
- Correct action/twirl preservation relative to a noisy identity is functional
  group closure. Natural utility relative to clean input is measurement
  robustness. Neither may substitute for the other.
- A selected-dose asymmetric failure may show that noise-law shape matters for
  this frozen complete system. It does not establish a universal sensor-noise
  theorem.
- If all laws pass, close this additive-noise branch only at the selected dose;
  do not extrapolate the result back to `sigma=0.05`.

## Execution and retention

The primary run uses all ten systems, both full 512-example cohorts, batch size
256, deterministic algorithms, and CUDA. A reduced shakedown is systems-only
and cannot count as scientific evidence. Completed per-system outputs and the
campaign aggregate must be resumable without changing bytes. Preserve the
source `sigma=0.05` campaign regardless of this outcome.
