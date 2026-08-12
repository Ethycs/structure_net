# TinyLLM shared-bias reference recentering v2 preregistration

**Status:** FROZEN BEFORE PRIMARY MODEL EVALUATION  
**Date:** 2026-08-10  
**Hypothesis:** `tinyllm-bias-reference-recentering-v2`  
**Schema:** `nal.tinyllm-bias-reference-recentering.v2`  
**Evidence role:** `preregistered_pre_model_numerical_corrective_bias_reference_intervention`

## Corrective reason

Version 1 stopped during the full-cohort model-independent preflight, before a
checkpoint was loaded or a primary posterior was evaluated. The declared
offset recentering missed the frozen corrected-planar tolerance on extrapolation
by one float32 unit: `2.3841858e-7` against `2e-7`. The 64-example shakedown had
not included that worst-case value.

The scientific hypothesis, systems, examples, components, population gates,
control ceilings, and every numerical tolerance remain unchanged. Version 2
only separates two numerical contracts that version 1 had conflated:

1. audit the algebraic corrected-planar construction in float64, where its
   maximum extrapolation error is below the existing `2e-7` threshold;
2. retain the existing float32 realized front-end feature (`1e-6`) and posterior
   (`2e-6`) equivalence gates on the actual frozen systems.

This is a pre-model numerical correction, not a post-outcome threshold change.
The invalid v1 preregistration and shakedown remain preserved.

## Frozen source

- Conditions: `analytic_calibrated`, `learned_calibrated_equivariant`.
- Seeds: `7, 17, 29, 41, 53`.
- Shifts: composition and extrapolation.
- Selected dose: `sigma = 0.03125`.
- Source campaign SHA-256:
  `9f7fdf98e83a320d5d49d9191e6a0f0cd6f872f32f406381c5a290f517dbed4b`.
- Source result-manifest SHA-256:
  `17d614cadfeca5e019258578ad9abe8dc269f899f7144e712e8154f7988ce07b`.
- Source component-contract SHA-256:
  `26b8ad368fe8d1af811f2ff62d4874545c6d90b3aa5d376a9a59002092342b2f`.
- Source runner SHA-256:
  `eba5182082d8604fba47d65fc0f64706b00ac9f4fde6dbf45c63fca56ed44bb5`.
- Source DVC root: `e3bfc6a9401916ffc7f942678044fb0a.dir`.
- Source lakeFS commit:
  `a0f6b67d7aad58dc96de58406abf7064728613e73134ba4959e18dd46c0cc92a`.

No model, front end, data point, target, selected component, or predecessor
artifact may change.

## Intervention and controls

For the frozen full-positive sensor

\[
x_+=x+\epsilon_c+\mu,\qquad \mu=0.03125e_x,
\]

the observed zero-signal shared-bias pilot is `p=mu`. It is constant,
phase-independent, target-independent, and contains no label. The repair changes
only the observed calibration offset:

\[
o_{\mathrm{repair}}=o+p.
\]

Evaluate:

- `source_full_plus`: sealed uncorrected positive-bias posterior;
- `recenter_correct`: `x_+` with `o+p`;
- `recenter_wrong_sign`: `x_+` with `o-p`, which leaves centered error plus
  `2 mu`;
- `recenter_target_changing`: correct recentering followed by the declared
  observed orthogonal-axis reflection and signed-speed update.

The sealed centered posterior is the exact repair-equivalence reference.

## Integrity gates

- no new random draws;
- pilot time constancy and value error at most `2e-7`;
- float64 corrected-planar repair versus centered error at most `2e-7`;
- float64 wrong-sign versus centered-plus-`2 mu` error at most `2e-7`;
- target-changing action sensor/calibration involution error at most `2e-6`;
- target-changing analytic-feature RMS at least `0.50`;
- realized float32 repaired feature versus centered feature error at most `1e-6`;
- realized repaired posterior versus sealed centered posterior error at most
  `2e-6`;
- clean posterior and source metrics replay within `2e-6` on the full cohort;
- finite outputs and unchanged model/front-end state.

Any failure invalidates the campaign.

## Natural utility and population gate

Each seed must pass composition and extrapolation simultaneously with:

- exact-bin accuracy loss at most `0.05`;
- circular-error increase at most `pi/16`;
- target cross-entropy increase at most `0.10`.

Confirmation requires:

1. all integrity gates pass;
2. `source_full_plus` passes fewer than four seeds in both arms;
3. `recenter_correct` passes at least four seeds in both arms;
4. `recenter_wrong_sign` passes at most one seed in both arms;
5. `recenter_target_changing` passes at most one seed in both arms.

Classification order remains:

- `observed_bias_reference_repair_specific`;
- `algebraic_repair_without_specificity`;
- `observed_bias_reference_insufficient`;
- `invalid`.

## Boundary

This remains an exact-pilot positive control. It does not estimate a persistent
bias from finite noisy pilot observations, test time-varying bias, establish
sample complexity, or license retraining. A specific pass permits only a later
separately preregistered pilot-acquisition titration.
