# TinyLLM observed-deck twirl preregistration

**Status:** PREREGISTERED — NO LEARNED-CHECKPOINT ACTION OUTCOME INSPECTED  
**Date:** 2026-08-10  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, outcome-directed,
existing-checkpoint, no-fit observed-action intervention  
**Hypothesis:** `tinyllm-observed-deck-twirl-causal-closure-v1`  
**Schema:** `nal.tinyllm-observed-deck-twirl-causal-closure.v1`

## Decision question

The calibrated front-end causal-closure experiment established that an oracle
barycenter of two exact target-equivalent rows is sufficient before block-0
attention in all retained analytic and learned-equivariant systems. Those two
rows carry independently sampled nuisance values. Their average is therefore
not a deterministic orbit projection that can be constructed from one
observation.

This prospective experiment asks the remaining deployment question:

> Can a target-preserving `C2` partner be constructed from one observed
> calibrated structured input, without latent phase, target labels, branch,
> fiber ID, or a second nuisance draw, such that its within-example Reynolds
> twirl is causally sufficient for the unchanged TinyLLM continuation?

The test must precede richer-group training or architecture-population
replication. Failure would show that the preceding oracle causal result does
not yet yield an observable projection. Success would close the current `C2`
oracle-membership gap without fitting a parameter.

## Locked source

Use only the completed calibrated causal-closure campaign and its immutable
calibrated source systems:

```text
causal-closure campaign
data/experiments/tinyllm_calibrated_frontend_causal_closure/
    20260810_d15_preregistered/campaign_results.json

campaign SHA-256
1457fdcce224157c5d8b19c11317b3d3c47cc7d8575097c78e76ba8798431b14

causal-closure implementation SHA-256
5060b45674430351dabb6cd67af5e41a215f883d09b9702edd3d36b3d1d51260

15-result manifest SHA-256
baed34a16dca206536b2e9cd221fd9f7556f4c063f85ee857352522e770844f4

calibrated-system source campaign SHA-256
80623a7283fed3d902a1cbf9fb58afe2a2fef82ab86406212f907de40a96c501
```

Hard-validate the campaign, result, posterior-diagnostic, checkpoint, model-
state, front-end-state, task-configuration, preregistration, and cohort hashes
before inspecting an action outcome. Use seeds `7`, `17`, `29`, `41`, and
`53` in the `analytic_calibrated` and
`learned_calibrated_equivariant` arms. The checkpoint seed is the replication
unit. The raw-calibrated arm is outside scope because it consumes the full
tokenized three-channel history rather than the structured planar interface.

## Locked cohorts

Regenerate the same two held-out cohorts:

| Split | Seed | Examples | Cohort SHA-256 |
| --- | ---: | ---: | --- |
| composition | `1399` | `1024` | `b025623e23f534ca3670f49f1bafc1c3979d1ca834aec2223f9c3166be8f52a6` |
| extrapolation | `2408` | `1024` | `f2f1e8c2c06b38fbecf856f0679e9861d9fab3e1c2a41358e55043f4d309a214` |

No cohort row is selected, removed, paired, or tuned after outcome inspection.
Fiber IDs and branch labels are forbidden inputs to the primary intervention.

## Observed `C2` action

The structured observable is the decoded planar history and calibration
packet. Let

```text
q = observed orientation unit vector
a = observed positive amplitude
o = observed planar offset
d = observed planar drift
s = observed signed speed
t = normalized history in [-1, 0]
v(t) = (x_planar(t) - o - d t) / a
```

Define the reflection across the observed orientation axis

```text
R_q v = 2 q <q,v> - v.
```

The correct observable action is

```text
x'_planar(t) = a R_q v(t) + o + d t
s' = -s,
```

with orientation, amplitude, offset, and drift unchanged. The third harmonic
channel is carried through unchanged because neither structured front end
consumes it. The transform is applied after token decoding and before the
structured front end; it is not re-quantized.

After calibration, this sends current phase and direction to
`(phi, direction) -> (-phi, -direction)`. Future phase therefore goes to its
negative, preserving absolute cosine exactly. Construction may read only the
decoded planar history, calibration packet, and fixed sensor-history grid.

## Pre-outcome action contract

The action was audited without loading a learned checkpoint. On the two locked
cohorts it must satisfy:

| Contract | Ceiling / floor | Observed pre-outcome audit |
| --- | ---: | ---: |
| sensor involution error | `<=2e-6` | `8.35e-7` composition; `1.08e-6` extrapolation |
| calibration involution error | `<=1e-7` | `0.0` both |
| target-cosine error | `<=1e-7` | `0.0` both |
| corrected-norm error | `<=1e-6` | `4.77e-7` both |
| analytic-feature error | `<=1e-6` | `1.79e-7` both |
| transformed planar absolute maximum | `<=2.0` | `1.385`; `1.960` |
| relative RMS from independent-nuisance oracle mate | `>=0.5` | `1.093`; `1.292` |

The last floor prevents relabeling the old oracle pair as the new observable
action. These are mathematical and input-lifecycle contracts, not learned-
checkpoint outcomes.

## Frozen interventions

Capture the complete token-by-channel residual sequence for three inputs:

1. `identity`: the ordinary observed structured input;
2. `observed_deck`: the correct reflection and signed-speed flip above;
3. `orthogonal_axis`: the matched semantic control, replacing `q` by its
   orthogonal axis in the reflection while still flipping signed speed.

The orthogonal-axis transform is equally norm-preserving and involutive, but
maps calibrated future cosine to its negative. It is therefore a task-changing
isometric control constructed without target labels.

At `pre_block`, `block0_post_attention`, `block0_post_mlp`, and `full`, form:

```text
correct twirl = 0.5 * (R_identity + R_observed_deck)
control twirl = 0.5 * (R_identity + R_orthogonal_axis).
```

Patch each repeated state into the frozen continuation from that cut. Also
continue the unaveraged transformed states to measure task invariance under
the correct action and task change under the control. The primary causal
intervention is the correct `pre_block` twirl; later cuts and transformed-
alone states are localization diagnostics.

For the first attention and MLP sublayers, compare propagation of the correct
twirl with the actual next-cut twirl. Record the Reynolds/Jensen residual norm,
posterior Jensen--Shannon divergence, and the four causal regimes, using the
same task gate as the source closure experiment.

## Locked endpoint

Relative to the ordinary frozen posterior, a state is task-sufficient only if
all three conditions hold:

```text
exact-bin accuracy loss <= 0.03
mean circular-error increase <= pi/16 radians
target cross-entropy increase <= 0.10 nats
```

A seed passes only when composition and extrapolation both pass. The primary
hypothesis requires the correct `pre_block` twirl in at least four of five
seeds in **both** structured arms.

Specificity additionally requires the orthogonal-axis `pre_block` twirl to
pass at most one of five seeds per arm. The correct transformed-alone state is
secondary: it distinguishes a task-invariant action response from a pipeline
whose Reynolds average is sufficient despite action variance.

## Locked classification

Apply this table in order after all source, replay, state, numerical, action,
and specificity contracts:

| Outcome | Classification | Primary pass |
| --- | --- | --- |
| correct pre-block twirl `>=4/5` both arms; transformed alone `>=4/5` both | `observable_twirl_closed_action_invariant` | yes |
| correct pre-block twirl `>=4/5` both arms; either transformed-alone arm below `4/5` | `observable_twirl_closed_action_variant` | yes |
| analytic twirl passes; learned twirl fails | `analytic_only_observable_twirl` | no |
| learned twirl passes; analytic twirl fails | `learned_only_observable_twirl` | no |
| neither structured twirl passes | `observable_twirl_not_causally_sufficient` | no |
| any contract fails | `invalid` | no |

Later-cut success cannot rescue a failed pre-block primary claim. No threshold,
cut, control, cohort, or classification may change after the first learned-
checkpoint action response is visible.

## Lifecycle and stopping rule

Run one analytic seed-7 lifecycle shakedown. The analytic action response is
fixed by the declared canonicalizer and serves as a systems positive control;
the run is not pooled. It must validate source loading, all three action
captures, continuation replay, serialization, and exact resume.

Then run all ten primary frozen cells. If the observable twirl passes, close
the current `C2` oracle-membership gap and do not train a group encoder. If it
fails only in the learned arm, report that failure before considering an
explicit reflection-equivariant front end. If both fail, stop using the oracle
closure result as deployment evidence. No model, front end, task head, probe,
observer, or action parameter may be trained or fit in this experiment.

## Scope boundary

This is an internal software transform of the decoded structured planar
observation. It assumes the calibration axis and signed speed are observed and
the planar noise law is reflection-compatible. It does not produce a partner
for the raw three-channel token model, infer an unknown group, eliminate the
cost of calibration, or establish behavior for richer groups, natural
language, real sensors, or an architecture population.
