# TinyLLM C3 temporal continuation/readout decomposition preregistration

**Status:** FROZEN BEFORE PRIMARY FITS OR NEW READOUT OUTCOMES

**Date:** 2026-08-11

**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED / ARTIFACT-ONLY`

**Hypothesis:** `tinyllm-c3-temporal-continuation-readout-v1`

**Evidence role:** `prospective_artifact_only_c3_continuation_readout_decomposition`

**Depends on:** [observable C3 d6 result](../08%20-%20Analysis/2026-08-11_tinyllm-c3-temporal-quotient-d6.md)

## Question and prediction

The analytic `C3` population exposed an exact invariant carrier, passed the
registered semantic and causal-closure gates in `5/5` seeds, but passed the
complete natural task gate in only `2/5`. A fixed analytic temporal computation
on the untouched observations passed the same task floors without loading a
checkpoint. That localizes the failure downstream of the invariant sensor but
does not distinguish the frozen continuation from its tied sixteen-row answer
interface.

This study asks:

> Does the frozen final query expose the future physical cosine through one
> affine scalar, or is the useful invariant carrier lost or nonlinearly
> entangled before the answer interface?

The directional prediction is that a physically typed affine readout from the
frozen normalized final query will pass the complete natural-task gate in at
least four of five seeds. If it passes with specificity, the continuation
already exposes a usable physical coordinate and the inherited answer rows are
the failing component. If only the exact front-end temporal bypass passes, the
next constructive study must type the temporal continuation rather than fit
another answer head.

## Frozen source population

The replication unit is one of the five completed analytic d6 checkpoints:

```text
seed = 7, 17, 29, 41, 53
```

The runner must reject any source mismatch against:

| Source | SHA-256 |
| --- | --- |
| campaign | `e46710de358a3c9c5d30ddd4a19e36182b94a50cabc699014f79d58d3d3de7cc` |
| five-result manifest | `7dfdcf1ff80fe20a975fe6a7d1311dc92e3ff1a396a6da9550c91835a568a0ff` |
| checkpoint/front-end/diagnostic manifest | `a0b90484863346cf2a5e0ef8be65cac3a221cfa50a373f88c9ead07a0cd351a1` |
| C3 campaign runner | `9b2cd0e3ce3752b7eea80d5859c11880a9d3732fb48b58306e34eab4f080d5ec` |
| C3 analysis | `89dacc60d02707678e689c6ce1e8f9c963889af352565a227bb90ed8e367e6a3` |
| C3 system/training implementation | `dbc934190b5f725185ff9a99690409ac63fc4f16bd04686f870e7ef21cba03a6` |
| reusable closed-form interval implementation | `bb6a73c203fcf4e654295bf7567f205826a6f054153ad50bbdd36851297de926` |

Every source result must retain validity, exact action, representation,
four-cut causal closure, and identity replay. The source natural-task outcome
is known and remains `2/5`; no new arm may overwrite or reinterpret it.

## Data contract

For each seed, regenerate its exact sealed 4,096-example paired training cohort
and require its training-data and minibatch digests to match the source result.
All fitted maps use only that training cohort.

Final evaluation reuses the untouched 1,024-history one-sheet cohorts from the
primary campaign:

| Regime | Seed | Role |
| --- | ---: | --- |
| composition | `331003` | complete nuisance composition inside training marginals |
| extrapolation | `331021` | outside-speed and wider nuisance ranges |

Their latent fingerprints and dataset hashes must match the sealed campaign.
No final example, target, result, or threshold participates in fitting or
hyperparameter choice.

## Frozen state and output definitions

For every training and final example, load the exact model and analytic
front-end checkpoints, replay the source posterior, and retain:

```text
s_nat = sum_i softmax(logits)_i * center_i
h_final = ln_f(full_query_residual)
```

where `center_i = linspace(-1, 1, 16)`. The source replay must agree with the
stored task metrics and direct model path within `2e-6`.

The fixed ordered interval decoder is:

```text
width = 2 / 15
logit_i(u) = -0.5 * ((center_i - clip(u, -1, 1)) / width)^2
```

It must reproduce the generator target posterior from exact future cosine to
maximum absolute error `<=2e-6`.

## Arms

| Arm | Fitted object | Frozen object | Role |
| --- | --- | --- | --- |
| source replay | none | complete source computation | inherited baseline |
| output scalar recalibration | affine ridge map `s_nat -> cos(3(theta_0+8v))` | complete continuation | tests whether sign/scale/offset of the natural posterior mean is sufficient |
| untyped final readout | affine ridge map `h_final -> 16` centered log-targets | complete final residual | tests whether replacing the tied answer rows is sufficient |
| typed final readout | affine ridge map `h_final -> cos(3(theta_0+8v))` plus fixed interval decoder | complete final residual | primary continuation-versus-readout intervention |
| exact temporal bypass | no fitted parameter; analytic `q_7 conjugate(q_6) q_7` plus fixed interval decoder | bypasses TinyLLM | positive localization control only |

Each fitted arm has a same-width target-shuffled control. One deterministic
whole-example permutation per seed is applied to both scalar and posterior
targets during fitting. The permutation is fixed by base seed `20260811` and
must not be changed after any outcome.

## Closed-form fit contract

All maps use the already tested deterministic float64 ridge implementation.

1. Standardize each input column with training means and population standard
   deviations, floored at `1e-6`.
2. Append an unregularized intercept.
3. Solve the mean-squared normal equations once with ridge `1e-4` on all
   non-intercept coefficients.
4. The typed and recalibration targets are exact future cosine.
5. The untyped target is `log(target_posterior)` centered over its sixteen
   rows.
6. Store coefficients, standardization, design rank, condition number,
   training RMSE, and complete held-out predictions.
7. Use zero iterative optimizer steps, write no source checkpoint, and make no
   hyperparameter, feature, cohort, or threshold selection.

## Primary endpoints

Every true or shuffled arm is evaluated against the original absolute task
gates:

| Endpoint | Composition | Extrapolation |
| --- | ---: | ---: |
| posterior-mean correlation | `>=.90` | `>=.90` |
| exact-bin accuracy | `>=.50` | `>=.35` |
| target cross-entropy | `<=1.80` | `<=2.20` |
| predicted-bin coverage | `>=14` | `>=12` |

The typed final readout must additionally have direct fitted-scalar correlation
with exact future cosine `>=.90` on both shifts. One seed passes only if every
endpoint passes simultaneously in both regimes.

The primary hypothesis passes if the typed final readout passes at least four
of five seeds and its shuffled control passes at most one. The recalibration
and untyped arms use the same `4/5` true and `<=1/5` shuffled population gates
as comparators; neither rescues a failed typed claim.

The exact temporal bypass must pass every absolute endpoint. Because its value
is model-independent on shared cohorts, it is one analytic population control,
not five independent model replications.

## Validity and lifecycle gates

Every cell must satisfy:

- source campaign, result, checkpoint, front-end, and diagnostic hashes match;
- regenerated training and final cohort hashes match;
- source checkpoint and front-end state digests remain unchanged;
- direct source posterior and stored task metrics replay within `2e-6`;
- exact interval posterior fidelity is within `2e-6`;
- every array and scalar is finite;
- coefficient/diagnostic save and reload is exact;
- zero backbone optimizer steps and zero changed model parameters.

Before primary execution, require CPU tests for ridge recovery, exact decoder
fidelity, target shuffling, gate aggregation, source rejection, and artifact
resume. Run one reduced seed-7 CUDA lifecycle outside the primary root and do
not pool its metrics.

## Locked outcome meanings

| Outcome | Interpretation | Decision |
| --- | --- | --- |
| output recalibration passes | the inherited output is mainly a scalar chart-calibration failure | freeze that interface prospectively before any learned-`C3` study |
| typed and untyped final readouts pass | the continuation exposes a linearly usable physical task state; inherited tied answer rows are inadequate | adopt the typed fixed decoder; do not retrain TinyLLM |
| untyped passes, typed fails | task state is linearly usable but not as one affine physical cosine | type the continuation, not merely its final rows |
| only exact temporal bypass passes | the analytic sensor is sufficient but the frozen continuation does not expose the target through the registered affine interfaces | preregister a fixed temporal operator/metric interface; do not fit another endpoint map |
| shuffled controls pass or validity fails | result is nonspecific or invalid | stop without a model-quality interpretation |

Failure of the affine arms means only that the target is not accessible through
these registered linear interfaces. It does not prove that target information
is absent or that no nonlinear decoder could recover it.

## Artifact root and command

Primary root:

```text
data/experiments/tinyllm_c3_temporal_continuation_readout/
  20260811_d6_preregistered/
```

Planned command:

```bash
MPLCONFIGDIR=/tmp/mpl-c3-readout pixi run python -m \
  experiments.structure_net.tinyllm_c3_temporal_continuation_readout \
  --device cuda:0 \
  --output \
  data/experiments/tinyllm_c3_temporal_continuation_readout/20260811_d6_preregistered
```

## Method boundaries

- The source checkpoint failures are outcome-known; the new fitted readout
  outcomes are prospective.
- These are supervised diagnostic interfaces using generator targets on the
  sealed training cohort, not unsupervised discoveries.
- The exact temporal bypass is an analytic positive control, not a TinyLLM
  repair.
- Probe decodability is not used as a primary endpoint.
- No conclusion applies to the stopped raw or learned-`C3` arms, d10, another
  temporal task, or real sensor data.
