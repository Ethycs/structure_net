# TinyLLM biased-noise component causal decomposition preregistration

**Status:** PREREGISTERED  
**Date:** 2026-08-10  
**Hypothesis:** `tinyllm-bias-component-causal-decomposition-v1`  
**Schema:** `nal.tinyllm-bias-component-causal-decomposition.v1`  
**Evidence role:** `preregistered_frozen_bias_component_intervention`

## Question

The selected-dose noise-law study found that isotropic and zero-mean
anisotropic error preserve population utility at `sigma=0.03125`, while the
law with a persistent positive lab-frame mean passes only `1/5` analytic and
`3/5` learned checkpoints. Correct observed `C2` action and twirl remain
task-sufficient in every cell.

The next question is causal and narrower:

> Is the deterministic lab-frame mean alone sufficient to reproduce the
> biased-law utility failure, or does the failure require an interaction
> between the mean and centered stochastic error?

No model, front end, head, action, observer, probe, denoiser, or noise process
may be trained or fitted.

## Frozen source

- selected-dose campaign:
  `data/experiments/tinyllm_noise_law_dose_localization/20260810_d10_preregistered/campaign_results.json`
- campaign SHA-256:
  `9b05823ebdb88bd828f27699da596dc5e7dcf0c4af5e13f1664fa70e5111f9bd`
- campaign result manifest:
  `976545c812e428ea4b020ca46a88643cb741a6ad5c7797389e9a5e6ca81f7562`
- selected arrays content:
  `740c5c30f01c482fa799db1865a11c069ad3b59f474879a59f1906b94f4130f3`
- selected-dose runner SHA-256:
  `39a72dd535f96f13bae644c74096b298b85fb8587d980211dc489ed463aeb725`
- original `sigma=0.05` error-array file SHA-256:
  `d3771eac8e29f7940df7feaedebe74a5a78fb273cda2e70928c9be9e37ff3ba6`
- original error-array content SHA-256:
  `93df61bc76ed073ea241c9450e7ec3523e7a98b5ac06e58d7e920a5df07d70aa`
- source DVC root:
  `c07286d2b9710cd68228cd21f487e425.dir`
- source lakeFS commit:
  `d4fb92ef41e39d0cc672d672e55c9192ea0e9dcf01597b1a549efcf973577061`

Reuse the exact ten retained d8/N3 systems, seeds `(7, 17, 29, 41, 53)`,
composition and extrapolation cohorts, calibration packets, token decoder,
task, and batch size 256. The two arms remain `analytic_calibrated` and
`learned_calibrated_equivariant`.

## Exact component construction

Let `z` be the standard Gaussian draw underlying the frozen isotropic error
and let `sigma*=0.03125`. The sealed biased array is

`epsilon_plus = sigma* (z/sqrt(2) + e_x)`.

Construct, without new randomness:

1. `centered = sigma* z/sqrt(2)`;
2. `mean_plus = sigma* e_x`;
3. `full_plus = centered + mean_plus`;
4. `full_minus = centered - mean_plus`.

`full_plus` must reproduce the stored selected biased array within maximum
absolute error `2e-7`. `mean_plus` must be exactly constant across examples and
sensor steps. `centered` must equal the selected isotropic array divided by
`sqrt(2)`. The empirical planar-energy difference between `full_plus` and
`full_minus` must be at most 2% relative; otherwise the sign comparison is
invalid for these finite draws.

The stored clean and `full_plus` posteriors are reused. Their files and hashes
must match the selected-dose manifest. Recomputed clean posteriors must replay
the stored clean posterior within `2e-6`; stored `full_plus` metrics must replay
from the stored posterior within `2e-6` per scalar metric.

## Primary endpoint

Run new frozen forwards only for `centered`, `mean_plus`, and `full_minus`.
For every variant, retain the existing natural-utility gate relative to clean:

- exact-bin accuracy loss `<=0.05`;
- circular-error increase `<=pi/16`;
- target cross-entropy increase `<=0.10`.

A seed passes a variant only when all three metrics pass on both composition
and extrapolation. An arm passes at `>=4/5` seeds.

The deterministic-mean sufficiency hypothesis passes only if:

1. `centered` passes in at least `4/5` seeds in both arms; and
2. `mean_plus` passes in fewer than `4/5` seeds in both arms.

The source `full_plus` failure (`1/5` analytic, `3/5` learned) is a required
positive failure control but is not counted as a new run.

This gate is intentionally causal rather than correlational. Mean-only has
lower expected energy than the complete biased law; if it independently breaks
both populations while centered noise does not, the persistent mean is
sufficient for the registered utility failure.

## Secondary sign diagnostic

`full_minus` has the same mean magnitude and centered draw with the sign of the
mean reversed. Classify it without changing the primary gate:

- `positive_direction_specific` if `full_minus` passes both arms;
- `bidirectional_mean_magnitude` if `full_minus` fails both arms;
- `arm_specific_directional` otherwise.

Also retain per-shift pass counts, accuracy loss, posterior JS from clean, and
the fraction of examples whose predicted bin differs between `full_plus` and
`full_minus`. These are descriptive and cannot rescue a failed primary.

## Ordered campaign classifications

1. `invalid_integrity` if source, checkpoint, finite-state, or frozen-state
   contracts fail;
2. `invalid_component_reconstruction` if the algebraic or energy contract
   fails;
3. `centered_stochastic_breaks_utility` if `centered` misses either arm-level
   population gate;
4. `deterministic_mean_sufficient` if the primary endpoint passes;
5. `mean_noise_interaction` if `centered` and `mean_plus` both pass both arms
   while the frozen `full_plus` control fails both;
6. `arm_specific_or_underdetermined` for any remaining valid pattern.

Only classification 4 confirms the primary hypothesis. Classifications 3, 5,
and 6 are valid falsifications or narrowing outcomes.

## Mechanistic control and boundaries

The source campaign's correct action/twirl `5/5` counts and orthogonal-control
`0/5` counts are pinned and inherited as a mechanistic control. They are not
rerun because this study asks what damages the noisy identity before the group
intervention, not whether closure already shown at the same selected biased
law can be reproduced again.

The experiment tests one positive lab direction, its sign reversal, one fixed
draw pair, two synthetic shifts, and ten checkpoints. It does not establish a
universal bias theorem, estimate a deployment noise distribution, or license
training a correction. If the mean is sufficient, a later observed centering
intervention may be justified; if not, decompose the mean-noise interaction
before fitting anything.

## Execution and retention

The primary uses deterministic algorithms, CUDA, the full 512-example cohorts,
and all ten systems. A reduced shakedown is systems-only. Per-system results and
diagnostics must be resumable without changing bytes. Preserve the selected-
dose source campaign and record all negative outcomes.

