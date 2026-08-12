# TinyLLM observable C3 temporal-quotient training design

**Status:** DESIGN FROZEN FOR IMPLEMENTATION; NO TRAINING AUTHORIZED YET

**Date:** 2026-08-11

**Conformance:** NAL-STD-EXPERIMENT `DESIGN / CONDITIONAL PROSPECTIVE CAMPAIGN`

**Hypothesis:** `tinyllm-c3-temporal-quotient-training-v1`

**Pre-model license:** `c3_temporal_quotient_preflight_passed`

## Question

Given an exactly observable `C3` action and a nontrivial invariant temporal
carrier, does architectural symmetry make the learned carrier task-useful and
causally quotient-sufficient under composition and extrapolation?

This design follows the passed [no-training preflight](../08%20-%20Analysis/2026-08-11_tinyllm-c3-temporal-quotient-preflight.md). It does not authorize an immediate 30-cell launch. Implementation, lifecycle tests, exact parameter accounting, analytic-arm shakedown, and a dated preregistration must complete first.

## Matched arms

Use one generator, target, example order, minibatch schedule, optimizer, model
preset, and task decoder per seed.

### Raw sequence

Normalize the decoded three-channel observations using only the target-free
amplitude/offset/drift packet. Project each ordered channel triple directly into
the residual width. Channel order remains visible, so this arm may carry deck
identity.

### Fixed analytic invariant

Compute the registered two-coordinate cubic carrier `q(t)` at each of eight
time steps. A fixed `2 -> d_model` injection maps the sequence into TinyLLM.
The carrier computation has no trainable parameters; its injection and
TinyLLM remain trainable.

### Learned C3 invariant

Apply one shared scalar feature map to every normalized channel, take fixed
first-character Fourier components over the channel axis, combine feature
channels with a learned complex-linear map, normalize, and cube. This produces
an exact invariant two-coordinate sequence for every parameter state. Then use
the same shaped `2 -> d_model` injection and TinyLLM continuation as the
analytic arm.

The learned arm receives task loss only. Do not regress the analytic carrier,
add contrastive/adversarial losses, or penalize branch information. The
scientific question is whether the restricted function class learns a useful
quotient, not whether direct representation supervision can copy the positive
control.

## Interface and capacity controls

- Analytic and learned arms must expose identical sequence shapes and the same
  injection/TinyLLM/decoder parameterization.
- The shared learned channel map is the only extra learned front-end capacity.
- Report exact trainable counts by component.
- Add a parameter-matched unrestricted three-channel front-end diagnostic only
  if it can retain the exact raw input information; it cannot replace the raw
  arm or enter the primary gate.
- The learned encoder's `C3` invariance must hold below `1e-5` before and after
  training for arbitrary weights and two registered parameter states.

## Conditional execution ladder

### Stage 0: implementation and systems lifecycle

Before primary training:

1. pin the passed generator/preflight and all imported source hashes;
2. test token group laws, target invariance, carrier invariance, raw sheet
   visibility, paired examples, and target-changing controls;
3. verify same-seed data and minibatch hashes across all three arms;
4. verify identical TinyLLM initialization across arms;
5. run CPU two-step lifecycle cells outside the evidence root;
6. run one d6 analytic CUDA shakedown and prove exact resume;
7. serialize task floors and the final gate table in a dated preregistration.

No shakedown outcome may enter scientific evidence.

### Stage 1: d6 decisive population

Train all three arms for seeds `7,17,29,41,53` using d6. This is 15 primary
cells. Stop after d6 unless both structured arms pass their declared population
gates and all controls remain valid.

If the analytic arm fails, classify the campaign as a task/optimizer positive-
control failure and do not interpret the learned arm. If analytic passes but
learned fails, freeze the checkpoints and run the shortest no-training
sensor-versus-continuation decomposition; do not launch d10 or tune the loss.

### Stage 2: d10 architecture extension

Only a valid passing d6 structured population licenses the matched 15-cell d10
extension. D10 remains prospective because its outcomes are unseen when its
conditional preregistration is frozen. The full architecture-family claim
requires both stages; d6 cannot outvote d10.

## Training protocol

Start from the established task-only TinyLLM protocol unless lifecycle testing
finds a systems incompatibility:

```text
examples                4096
optimizer steps          600
batch size                64
optimizer              AdamW
learning rate          3e-4
weight decay            .01
global gradient clip    1.0
seeds                    7,17,29,41,53
```

Pair minibatches by exact future target fiber across deck elements. Every arm
within a preset/seed receives byte-identical latent phases, speeds,
calibrations, targets, deck draws, token arrays, and minibatch indices.

The task decoder is the fixed sixteen-bin interval likelihood over
`cos(3 theta_future)`. Train cross-entropy only. Record continuous posterior
mean, exact-bin accuracy, cross-entropy, circular/triple-angle error, predicted
bin coverage, and calibration slope.

## Representation measurements

At the front-end sequence, block-0 post-attention, block-0 post-MLP, and full
depth, measure on composition and extrapolation:

- exact `C3` action error;
- first and second character energy;
- invariant-carrier temporal prediction;
- deck decodability conditioned on the target and carrier history;
- task-posterior geometry and full-bin coverage.

The primary learned representation gate must jointly require temporal base
retention and chance conditional deck leakage in at least four of five seeds on
both shifts. Numeric thresholds must be frozen after the analytic CUDA
shakedown but before any learned primary cell.

## Frozen causal endpoint

Capture the full residual for all three exact deck sheets. At each cut compare:

1. natural continuation;
2. exact orbit-barycenter continuation;
3. a fixed target-changing derangement of whole orbit barycenters;
4. identity replay.

Require the orbit barycenter to preserve natural accuracy, cross-entropy, and
triple-angle error within preregistered tolerances on both shifts. Identity
replay must close within `2e-6`; target-changing controls may pass at most one
seed per arm/preset.

For raw models, also compute the exact Reynolds/Jensen defect at every attention
and MLP sublayer and classify the four causal regimes. For structured arms, the
front-end carrier is already invariant, so causal closure before attention is
the expected positive control.

## Mechanistic diagnostic

At the first raw synthesis front, decompose deck Fourier characters and measure
the task effect of the exact Reynolds defect. Estimate quadratic and cubic
terms only as secondary approximations.

Do not preregister "cubic dominance." In a real `C3` carrier, conjugate
characters `r=1` and `r=2` permit the neutral quadratic interaction
`1+2=0 mod 3`. The exact defect and frozen causal continuation remain primary.

## Population decisions

The seed is the replication unit. A structured seed passes only if natural
task adequacy, representation, exact symmetry, causal closure, replay, and
specificity pass jointly on both shifts.

| Outcome | Classification | Next action |
| --- | --- | --- |
| analytic and learned d6 pass `>=4/5`; controls valid | `c3_d6_structured_quotient_supported` | freeze conditional d10 preregistration |
| analytic passes; learned fails | `c3_architectural_invariance_not_learned_useful` | no-training sensor/continuation decomposition; stop |
| analytic fails | `c3_positive_control_task_failure` | repair task/training systems only; no learned interpretation |
| structured representation passes but causal/task gate fails | `c3_representation_without_causal_utility` | frozen interface localization; no loss tuning |
| d6 and conditional d10 both pass | `c3_temporal_quotient_architecture_stable` | report architecture-family support |
| d6 passes and d10 fails | `c3_temporal_quotient_architecture_conditional` | report boundary; no rescue sweep |
| any source, action, data, replay, finiteness, or control contract fails | `invalid` | preserve artifacts; repair systems only |

No post-outcome threshold, extra step, seed, loss, width, or endpoint map may
rescue a failed stage.

## Current stopping point

The no-training generator preflight passes, but this training design is not yet
a preregistration. The next authorized work is runner implementation and
lifecycle validation. No GPU training should begin until Stage 0 produces a
frozen dated preregistration with exact numeric task and causal tolerances.
