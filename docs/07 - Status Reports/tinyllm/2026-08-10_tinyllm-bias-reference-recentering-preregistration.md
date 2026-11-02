# TinyLLM shared-bias reference recentering preregistration

**Status:** FROZEN BEFORE MODEL EVALUATION  
**Date:** 2026-08-10  
**Hypothesis:** `tinyllm-bias-reference-recentering-v1`  
**Schema:** `nal.tinyllm-bias-reference-recentering.v1`  
**Evidence role:** `preregistered_frozen_observed_bias_reference_intervention`

## Motivation

The sealed component experiment classified the selected-dose failure as
`deterministic_mean_sufficient` and `positive_direction_specific`. Centered
error passed `5/5` seeds in both structured arms, positive mean-only missed both
population gates, and reversing the mean recovered `4/5` in both arms.

The cheapest next causal test is not retraining. It is to ask whether a known
zero-signal calibration pilot, observed through the same persistent sensor
bias, can recenter the frozen structured front ends.

## Frozen systems and data

- Conditions: `analytic_calibrated`, `learned_calibrated_equivariant`.
- Seeds: `7, 17, 29, 41, 53`.
- Shifts: composition and extrapolation.
- Models, front ends, examples, task targets, selected dose, and base error draw
  are inherited unchanged from the sealed bias-component campaign.
- Selected dose: `sigma = 0.03125`.
- No model, front end, head, observer, action, denoiser, probe, or noise process
  may be trained or fitted.
- No new random draw may be introduced.

Pinned source artifacts:

- campaign SHA-256:
  `9f7fdf98e83a320d5d49d9191e6a0f0cd6f872f32f406381c5a290f517dbed4b`;
- result-manifest SHA-256:
  `17d614cadfeca5e019258578ad9abe8dc269f899f7144e712e8154f7988ce07b`;
- component-contract SHA-256:
  `26b8ad368fe8d1af811f2ff62d4874545c6d90b3aa5d376a9a59002092342b2f`;
- source runner SHA-256:
  `eba5182082d8604fba47d65fc0f64706b00ac9f4fde6dbf45c63fca56ed44bb5`;
- source DVC root: `e3bfc6a9401916ffc7f942678044fb0a.dir`;
- source lakeFS commit:
  `a0f6b67d7aad58dc96de58406abf7064728613e73134ba4959e18dd46c0cc92a`.

## Intervention

Let the selected biased sensor be

\[
x_{+}=x+\epsilon_c+\mu,
\qquad \mu=\sigma e_x,
\]

where `epsilon_c` is the frozen centered draw. Introduce a phase- and
target-independent zero-signal pilot exposed to the same persistent bias. Its
observed planar value is exactly `p = mu`; it is not inferred from latent phase
or a task label.

Both structured front ends subtract the observed calibration offset `o` before
forming their invariant feature. The registered repair is therefore

\[
o_{\mathrm{repair}}=o+p.
\]

The sensor and every other calibration coordinate remain unchanged. Under the
declared shared-bias model this must make the corrected front-end input
algebraically identical to the previously evaluated centered-only condition.

## Arms

Evaluate four variants:

1. `source_full_plus`: the sealed biased identity, reused without a new forward;
2. `recenter_correct`: `x_+` with calibration offset `o+p`;
3. `recenter_wrong_sign`: `x_+` with calibration offset `o-p`, leaving an
   effective centered error plus `2 mu`;
4. `recenter_target_changing`: apply `recenter_correct`, then the already
   declared observed orthogonal-axis reflection and signed-speed update. This
   action is computed from the observed sensor and calibration only and changes
   the retained cosine target.

`source_centered` posteriors are also reused as the exact repair-equivalence
reference, but they are not a fifth causal arm.

## Integrity contracts

Before accepting any endpoint:

- the pilot is constant across examples, phase-independent by construction,
  and has planar norm exactly `sigma`;
- repaired corrected-planar input matches centered-only corrected input within
  `2e-7`;
- wrong-sign corrected-planar input matches centered plus `2 mu` within `2e-7`;
- repaired front-end features match centered-only features within `1e-6`;
- repaired posterior matches the sealed centered posterior within `2e-6`;
- clean and source posteriors replay within `2e-6`;
- the target-changing action is involutive within `2e-6` and has a nonzero
  analytic target effect;
- all values are finite and all model/front-end parameters remain unchanged.

Failure of any contract invalidates the affected cell and the campaign.

## Natural utility endpoint

Each seed/variant must pass on composition and extrapolation simultaneously:

- exact-bin accuracy loss from clean at most `0.05`;
- circular-error increase at most `pi/16`;
- target cross-entropy increase at most `0.10`.

## Primary gate

The hypothesis is confirmed only if all conditions hold:

1. every integrity contract passes;
2. `source_full_plus` remains below four of five seeds in both arms;
3. `recenter_correct` passes at least four of five seeds in both arms;
4. `recenter_wrong_sign` passes at most one of five seeds in both arms;
5. `recenter_target_changing` passes at most one of five seeds in both arms.

The locked classification order is:

- `observed_bias_reference_repair_specific` if the complete primary gate passes;
- `algebraic_repair_without_specificity` if correct recentering passes both
  population gates but either control ceiling fails;
- `observed_bias_reference_insufficient` if correct recentering misses either
  population gate;
- `invalid` if an integrity contract fails.

## Interpretation

- A specific pass would show that the latest failure is removable at the
  observed sensor-calibration interface without changing TinyLLM.
- Correct repair with failed specificity would show an algebraic cancellation
  but not an identified task-preserving mechanism.
- Failed correct repair would show that the deterministic mean diagnosis does
  not translate into a usable frozen interface correction.

This experiment does not estimate the bias pilot in a realistic deployment,
test drifting or example-dependent biases, establish finite-reference sample
complexity, or license retraining. A successful exact-pilot positive control
would justify a later separately preregistered pilot-noise titration; it would
not by itself establish robustness to imperfect bias acquisition.
