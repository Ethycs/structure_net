# TinyLLM frozen interval-readout decomposition preregistration

**Status:** FROZEN BEFORE PRIMARY FITS

**Date:** 2026-08-11

**Hypothesis ID:** `tinyllm-frozen-interval-readout-decomposition-v1`

**Evidence role:** `prospective_frozen_backbone_closed_form_interface_fit`

**Depends on:** [calibrated architecture-family replication](../08%20-%20Analysis/2026-08-10_tinyllm-calibrated-architecture-replication.md); [frozen scalar-interface decomposition](../08%20-%20Analysis/2026-08-11_tinyllm-frozen-scalar-interface-decomposition.md); [frozen scalar-domain extension](../08%20-%20Analysis/2026-08-11_tinyllm-frozen-scalar-domain-extension.md); [frozen cyclic answer projection](../08%20-%20Analysis/2026-08-11_tinyllm-frozen-cyclic-answer-projection.md)

## Question and prediction

The architecture-family campaign established a quotient-sufficient structured
representation in every d6/d10 checkpoint, but only ten of twenty systems met
the natural task-adequacy gate. Frozen scalar interventions then showed that a
one-dimensional interface was sufficient in nine of ten failed systems, while
one learned d10 checkpoint had a persistent four-bin hole. A fixed cyclic
answer projection changed the holes but did not complete the chart.

The task is not cyclic at its output. Its sixteen answer centers are ordered on
the interval `[-1, 1]`, and its target posterior is a Gaussian chart over those
centers. The prospective question is therefore:

> Does a physically gauged one-scalar readout with the exact ordered interval
> chart recover architecture-family task adequacy from the frozen final state?

The directional prediction is that the typed interval readout will pass the
original task floors jointly on composition and extrapolation in at least four
of five seeds in every preset/condition stratum. If it does, no jointly trained
input/continuation interface will be launched: the shortest sufficient repair
is a typed readout of an already adequate frozen state.

## Design

The replication unit is one retained checkpoint seed. The primary population
contains the twenty structured d6/d10 checkpoints:

| Axis | Values |
| --- | --- |
| preset | `d6`, `d10` |
| source condition | `analytic_calibrated`, `learned_calibrated_equivariant` |
| seed | `7`, `17`, `29`, `41`, `53` |
| held-out regime | composition, extrapolation |

All TinyLLM, front-end, scalar-embedding, transformer, layer-normalization, and
original language-head parameters remain frozen. Each fitted map is computed
independently from the source checkpoint's original 4,096-example N3 training
cohort. No held-out example, target, floor, or fitted outcome participates in a
fit or hyperparameter choice.

### Arms

| Arm | Fitted object | Frozen continuation | Output chart | Evidential role |
| --- | --- | --- | --- | --- |
| source replay | none | original | original tied answer rows | baseline and replay control |
| input affine gauge | scalar affine map from the natural front-end scalar to physical cosine | original scalar embedding and transformer | original tied answer rows | tests input-coordinate gauge alone |
| untyped final readout | affine map from final normalized residual to sixteen centered target logits | final residual is frozen | free sixteen-class softmax | tests whether untying answer rows is sufficient |
| typed interval readout | affine map from final normalized residual to physical cosine | final residual is frozen | fixed ordered Gaussian interval decoder | primary constructive arm |
| front-end typed bypass | scalar affine map from natural front-end scalar to physical cosine | bypassed | fixed ordered Gaussian interval decoder | positive localization control, not a TinyLLM repair claim |

Each fitted arm also has a target-shuffled control using the same source
features, fit rule, and parameter count.

## Fixed fit contract

All maps use a deterministic closed-form ridge solve; there is no iterative
optimizer, checkpoint selection, early stopping, validation tuning, or update
schedule.

1. The input to each final readout is the source final query residual after the
   frozen final layer normalization, exactly the input to the tied LM head.
2. Training columns are standardized using training means and population
   standard deviations, with a `1e-6` scale floor. Evaluation uses those stored
   training statistics.
3. An unregularized intercept is appended after standardization.
4. The ridge coefficient is fixed at `1e-4` in the mean-squared normal
   equations. The intercept is not regularized.
5. The typed target is exact future cosine from the declared generator. Its
   prediction is clipped to `[-1, 1]` before decoding.
6. The sixteen fixed centers are `linspace(-1, 1, 16)`. With
   `width = 2 / 15`, decoder logits are

   ```text
   logits_i(u) = -0.5 * ((center_i - u) / width)^2.
   ```

   This is the exact chart used by the generator before posterior
   normalization.
7. The untyped target is `log(target_posterior)` centered across the sixteen
   rows for each example.
8. The shuffled control uses one deterministic whole-example permutation per
   checkpoint and applies the same permutation to cosine and posterior targets.

The fit artifacts must store coefficients, training standardization, ranks,
condition numbers, training errors, and all held-out predictions. They are new
readout evidence and are never written into a source checkpoint.

## Data and provenance

The source is the completed prospective campaign at
`data/experiments/tinyllm_calibrated_architecture_replication/20260810_d6_d10_preregistered`.
The runner must reject any source mismatch against:

- campaign SHA-256
  `656d9814a032d1899810e81d398adf935cea3e1116712460e2062da188a0c9e2`;
- result-manifest SHA-256
  `f87740e70062f5fecf5238f00dd00774246e4f3e155dceb87752b099ce4ca80a`;
- artifact-manifest SHA-256
  `5c08c771d04aae513ad9605d9e4818867ab0a8b0303680337dabbf87dce352e0`.

The fit cohort must reproduce each source cell's stored training-data digest.
The two held-out cohorts are the same 1,024-example composition and
extrapolation cohorts used by the architecture replication and frozen scalar
diagnostics. Their generator seeds are `1399` and `2408`, respectively, and
their dataset hashes must match the sealed source values.

The original source task floors are retained per checkpoint and regime. No
floor is recomputed from the new arms.

## Primary endpoint and joint gate

For the typed interval readout, a seed passes only if all of the following hold
on both composition and extrapolation:

1. exact-bin accuracy is at least that checkpoint's original prospective task
   floor;
2. Pearson correlation between the fitted scalar and exact cosine is at least
   `0.90`;
3. every validity, replay, source-identity, decoder-fidelity, and finite-value
   control passes.

The primary hypothesis passes only if the same joint seed gate passes in at
least four of five seeds in each of the four preset/condition strata. Four
different marginal seed sets cannot be combined.

The target-shuffled typed arm may pass at most one of five seeds in each
stratum. A specificity failure invalidates the primary claim even if the true
arm reaches its task floors.

## Comparator gates

The input affine-gauge and untyped final-readout arms use the same task-floor
gate on both shifts, with `4/5` required separately in every stratum. They do
not substitute for a failed primary typed gate.

The exact-cosine fixed decoder must replay the held-out target posterior with
maximum absolute error at most `2e-6`. At least four of five analytic and four
of five learned front-end typed-bypass controls per preset must meet the source
task floors; a failure is explanatory rather than a rescue for the primary
arm.

## Secondary measurements

For every arm and shift, record:

- exact-bin accuracy, non-circular mean absolute bin error, target cross
  entropy, and predicted-bin coverage;
- fitted-scalar Pearson correlation, RMSE, range, and opposite-sheet paired
  absolute difference where applicable;
- number and identity of source-failed checkpoints repaired;
- train residual replay, source state hashes before and after analysis, fit
  matrix rank and condition number, peak CUDA allocation, and wall time.

The source representation and causal-closure gates are inherited facts about
the unchanged residual stream. They will not be re-probed and must not be
claimed as new measurements.

## Locked outcome meanings and stop rule

| Outcome | Interpretation | Next action |
| --- | --- | --- |
| typed interval arm passes all four strata with specificity | the frozen final state already exposes a portable physical scalar; the tied/free-form answer chart was the remaining engineering defect | stop before any joint-interface retraining |
| untyped arm passes but typed arm fails | the frozen state supports task recovery, but not through one stable affine cosine coordinate | prospectively type the continuation, not only its output |
| input affine-gauge arm passes | source front-end gauge mismatch was sufficient despite the prior exact-cosine substitution result | replicate the observable affine calibration; do not add readout capacity |
| only front-end bypass passes | the sensor quotient is adequate but the frozen TinyLLM continuation corrupts the typed interface | train a jointly typed scalar continuation/readout with the backbone comparator retained |
| no fitted arm passes all strata | neither endpoint-only repair is architecture-family sufficient | proceed to the preregistered prospective joint-interface branch |
| shuffled controls reproduce success or validity fails | result is invalid or nonspecific | stop and correct the contract without interpreting model quality |

Partial seed or stratum improvements remain measured secondary evidence and do
not rescue a failed primary gate.

## Shakedown and execution plan

Before primary execution, the implementation must pass:

1. CPU unit tests for the exact interval decoder, ridge fit, target shuffle,
   joint aggregation, source identity, and byte-stable resume;
2. a synthetic closed-form recovery test with no checkpoint;
3. one outcome-known d8 checkpoint lifecycle on CUDA, labeled systems evidence
   only;
4. save/reload checks for coefficients and diagnostics;
5. a representative memory pilot.

Primary artifact root:

```text
data/experiments/tinyllm_frozen_interval_readout_decomposition/
  20260811_d6_d10_preregistered/
```

Planned command:

```bash
MPLCONFIGDIR=/tmp/mplconfig pixi run python -m \
  experiments.structure_net.tinyllm_frozen_interval_readout_decomposition \
  --device cuda:0 \
  --output data/experiments/tinyllm_frozen_interval_readout_decomposition/20260811_d6_d10_preregistered
```

## Method boundaries

- The source checkpoints and their natural-task failures are already known;
  only the fitted interface outcomes are prospective.
- Ridge maps use generator-derived cosine during fitting. They are supervised
  task interfaces, not unsupervised discoveries or deployment-time oracles.
- The typed arm tests affine accessibility of physical cosine at the final
  residual. Failure does not prove that no nonlinear typed readout exists.
- The input affine arm is restricted to a physical cosine calibration and does
  not optimize through the frozen continuation.
- The d6/d10 presets jointly vary depth, width, and head count.
- Success establishes sufficiency on the declared synthetic task and shifts,
  not universal portability to arbitrary model families or real sensor data.
