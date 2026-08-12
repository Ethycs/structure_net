# TinyLLM noisy shared-bias pilot acquisition preregistration

**Status:** FROZEN BEFORE MODEL EVALUATION  
**Date:** 2026-08-10  
**Hypothesis:** `tinyllm-noisy-bias-pilot-acquisition-v1`  
**Schema:** `nal.tinyllm-noisy-bias-pilot-acquisition.v1`  
**Evidence role:** `preregistered_frozen_reused_draw_bias_pilot_titration`

## Question

The exact zero-signal shared-bias pilot repairs all ten frozen structured
TinyLLM systems with specific controls. Does the repair remain reliable when
the pilot bias is estimated by averaging a finite number of noisy zero-signal
measurements?

This is a sample-cost experiment, not another model or denoiser optimization.

## Frozen systems and source result

- Arms: `analytic_calibrated`, `learned_calibrated_equivariant`.
- Seeds: `7, 17, 29, 41, 53`.
- Evaluation shifts: composition and extrapolation.
- Bias magnitude and direction: `mu = 0.03125 e_x`.
- Sensor, centered evaluation error, models, front ends, examples, and natural
  utility thresholds are inherited without change.

Pinned exact-pilot source:

- campaign SHA-256:
  `1996ac4c2534b62a25a2f52ceadfd21055a91bdadc81f38ccf01c6855da2b7d0`;
- result-manifest SHA-256:
  `7dbbe3a49f4e3ebac36e891ec63d5336ff3be2e176e26f1a610cbfceecaabb4e`;
- intervention-contract SHA-256:
  `6bed75b6cd9a15be35f21e53463efa28bcc2f775f1490f31414e005398894004`;
- runner SHA-256:
  `fd6ea5108ccd733e360010c83a1a4a411512cbed239e3c9356a6a6bb77a6996a`;
- DVC root: `1de07aeb227a8093fa5973d37d63f9a6.dir`;
- lakeFS commit:
  `23a11ba9918f2adcf4397c619e8b942f7539e1f98bb52962f98be6f520e7c181`.

## Frozen acquisition streams

Reuse, without regeneration, the standard-normal acquisition artifact from the
sixteen-draw stability study:

- path:
  `data/experiments/tinyllm_acquisition_draw_stability/20260810_d16_preregistered/acquisition_draw_errors.npz`;
- SHA-256:
  `57eca80cccf1b916a60d79d5982bdbffe3b515cee7dfbee7645830448779aace`;
- source campaign SHA-256:
  `968f85010129d761268b4816d85ddd2ab578bbc93307e8a936e58fa891e89d93`;
- source result-manifest SHA-256:
  `d13e52a07423e507cef034c78b734219b85abc8468feae6313b52148fa95b163`;
- source runner SHA-256:
  `54c293d94582e4aa826772ac9c9a3791b5ed66c01aa9635fef75f433f7fe4e0d`;
- frozen draw seed root: `81027026`.

Use only `composition__errors[:, :, :2]`, interpreted as sixteen independent
draws, 256 repeated measurements, and two planar channels. The artifact audit
is pinned before model evaluation:

- shape `(16, 256, 2)`;
- channel means approximately `(0.01700, 0.00852)`;
- channel standard deviations approximately `(1.01357, 1.01263)`;
- cross-channel correlation `0.01448`;
- nested-prefix standard-normal mean RMSE at counts `1,4,16,64,256` equal to
  approximately `0.97356, 0.63539, 0.26869, 0.14140, 0.06917`.

The acquisition streams were generated under a distinct sealed SeedSequence
and are not the evaluation sensor-noise arrays. No new randomness is allowed.

## Pilot model

For draw `d` and count `m`, the repeated zero-signal pilot observations are

\[
p_{d,t}=\mu+\sigma_p z_{d,t},
\qquad
\sigma_p=0.03125/\sqrt{2},
\]

with estimate

\[
\widehat\mu_{d,m}
=
\mu+\sigma_p\frac1m\sum_{t=1}^{m}z_{d,t}.
\]

The same global estimate is applied to every example and both evaluation shifts
for a draw. It uses no phase, target, label, activation, or model output.

Registered nested counts:

\[
m\in\{1,4,16,64,256\}.
\]

The repair changes only the calibration offset:

\[
o\leftarrow o+\widehat\mu_{d,m}.
\]

## Controls

- `m0_source_full_plus`: the sealed uncorrected positive-bias source counts,
  pinned at `1/5` analytic and `3/5` learned.
- `exact_pilot`: the sealed exact-pilot source, pinned at `5/5` in both arms.
- `wrong_sign_draw0_m256`: use
  `o <- o - mu_hat[draw=0,m=256]`; it must pass at most one seed per arm.
- The exact-pilot target-changing control is inherited and pinned at `0/5` in
  both arms; it is not rerun.

## Natural utility endpoint

For every system, draw, and count, a cell passes only when composition and
extrapolation simultaneously satisfy:

- exact-bin accuracy loss from clean at most `0.05`;
- circular-error increase at most `pi/16`;
- target cross-entropy increase at most `0.10`.

An arm/draw/count passes when at least four of five seeds pass. A complete draw
passes only when both arms pass.

## Primary gate

The primary hypothesis is that a finite noisy pilot closes the exact-pilot
repair by `m=256`. Confirmation requires:

1. every source, acquisition-array, replay, finite-value, and state-integrity
   contract passes;
2. the uncorrected and exact-pilot source counts replay exactly;
3. at `m=256`, at least `15/16` complete draws pass;
4. `wrong_sign_draw0_m256` passes at most one seed in each arm;
5. the inherited target-changing exact-pilot control remains `0/5` in both
   pinned source arms.

The smallest registered count reaching `15/16` complete draws is the declared
reliable pilot count. No monotonicity is assumed for individual prefixes.

Classification order:

- `finite_noisy_pilot_repair_reliable` if the complete primary gate passes;
- `finite_noisy_pilot_arm_asymmetry` if exactly one arm reaches `15/16` at
  `m=256`;
- `finite_noisy_pilot_insufficient` if neither arm reaches the draw gate;
- `invalid` if any integrity or source-control contract fails.

## Interpretation and stop rule

A pass establishes a finite-sample positive control under an independent,
unbiased, homoscedastic Gaussian pilot law. It does not establish robustness to
bias drift, heavy tails, correlated pilot error, or a bias that varies by
example.

If `m=256` fails, close this practical exact-law recentering branch. Do not
increase counts, fit a bias estimator, retrain TinyLLM, or weaken the utility
endpoint under the same outcome. If it passes, report the smallest reliable
registered count; any broader noise-law study must be separately preregistered.
