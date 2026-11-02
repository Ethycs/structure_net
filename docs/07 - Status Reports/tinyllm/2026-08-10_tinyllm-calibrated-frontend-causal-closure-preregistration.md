# TinyLLM calibrated front-end causal-closure preregistration

**Status:** PREREGISTERED — NO CAUSAL PATCH OUTCOME GENERATED OR INSPECTED  
**Date:** 2026-08-10  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, outcome-directed,
existing-checkpoint, no-fit activation intervention  
**Hypothesis:** `tinyllm-calibrated-frontend-causal-closure-v1`  
**Schema:** `nal.tinyllm-calibrated-frontend-causal-closure.v1`

## Decision question

The calibrated d8/N3 campaign established that analytic and learned
symmetry-respecting front ends preserve absolute cosine and remove conditional
branch decodability at their output and at full transformer depth in all five
seeds. Those were probe and task endpoints. They did not establish that the
front-end quotient itself is a causally sufficient state for the frozen
continuation.

This outcome-directed diagnostic asks:

> If the two exact task-fiber sheets are averaged at the structured front-end
> output, can that barycenter replace the natural activation while preserving
> the unchanged task on composition and extrapolation?

It also asks whether the first attention and MLP sublayers are already closed
on that quotient, rather than needing branch-bearing cover variation to
synthesize a new invariant.

## Locked source

Use only the completed calibrated-identifiability campaign:

```text
campaign
data/experiments/tinyllm_calibrated_frontend_causal/
    20260806_d8_preregistered/campaign_results.json

campaign SHA-256
80623a7283fed3d902a1cbf9fb58afe2a2fef82ab86406212f907de40a96c501

source implementation SHA-256
73e152b9b6c7ab51ad08e6a80786c32a71566f9d4fb410ff2cf3df53761faf77

15-result source manifest SHA-256
34bf25feb896abc9b9e06386b474fc6795c94566a23cd97a06795435fba64d68
```

Hard-validate all campaign, per-result, model-checkpoint, front-end-checkpoint,
model-state, system-state, task-configuration, and held-out dataset hashes
before inspecting a causal patch outcome.

The source probe and task outcomes are already known. This study is
prospective only with respect to the new activation interventions and may
confirm causal sufficiency within the retained checkpoint cohort; it cannot
independently confirm the prevalence of the preceding representation result.

## Systems and replication units

Primary structured conditions:

- `analytic_calibrated`;
- `learned_calibrated_equivariant`.

Use seeds `7`, `17`, `29`, `41`, and `53`. The checkpoint seed is the
replication unit. All TinyLLM, front-end, embedding, layer-normalization, and
answer-row parameters remain frozen.

The matched `raw_calibrated` checkpoints are a descriptive comparator. Their
baseline task accuracy is too low for failure or success of a relative-loss
patch gate to serve as primary evidence. They cannot promote or invalidate the
structured-arm hypothesis.

## Data

Regenerate the exact locked held-out cohorts:

| Split | Seed | Regime | Examples | Exact fibers | Cohort SHA-256 |
| --- | ---: | --- | ---: | ---: | --- |
| composition | `1399` | composition | `1024` | `512` | `b025623e23f534ca3670f49f1bafc1c3979d1ca834aec2223f9c3166be8f52a6` |
| extrapolation | `2408` | extrapolation | `1024` | `512` | `f2f1e8c2c06b38fbecf856f0679e9861d9fab3e1c2a41358e55043f4d309a214` |

Every fiber contains the two target-equivalent `C2` branch sheets. Preserve
the source row order, input IDs, calibration packet, target posterior, target
bin, cosine, branch, and fiber ID exactly. No new examples or nuisance values
are fit or selected.

The cohort digest covers those tensors plus the calibration packet and is
locked before any causal-patch outcome is generated.

## Frozen activation cuts

Capture the complete token-by-channel residual sequence at:

1. `pre_block`: embedded front-end sequence plus position embedding, before
   block-0 attention;
2. `block0_post_attention`;
3. `block0_post_mlp`;
4. `full`: after the last transformer block.

At every cut, average the two exact fiber activations and repeat the barycenter
back to both sheets. Continue that state through the unchanged downstream
computation and answer rows.

For block-0 attention and MLP, separately retain:

```text
propagated barycenter: F(mean_j h_j)
actual next barycenter: mean_j F(h_j)
Reynolds/Jensen defect: mean_j F(h_j) - F(mean_j h_j)
```

Patch both invariant states into the continuation beginning at the next cut.
The predicted regime is `passes / passes`: quotient dynamics is already
causally closed before each sublayer. Record posterior Jensen--Shannon
divergence and residual defect norm descriptively; exact posterior equality is
not required for task sufficiency.

## Task-sufficiency gate

Compute the unchanged baseline posterior once per system and regime. A patched
state passes a regime only when all three conditions hold:

```text
exact-bin accuracy loss from baseline <= 0.03
mean circular-error increase <= pi/16 radians
target cross-entropy increase <= 0.10 nats
```

A seed passes a cut only when both composition and extrapolation pass. An arm
passes a cut at `4/5` seeds. The primary structured-arm gate requires the
orbit-average patch to pass at **all four cuts** in at least four of five seeds
per arm.

The specific front-end causal claim additionally requires `pre_block` to pass
at four of five in both structured arms. Later-cut success cannot rescue a
failed pre-block claim; it would instead locate synthesis inside the
transformer.

## Controls and contracts

1. Continuing the unmodified captured state from every cut must reproduce the
   ordinary posterior with maximum absolute error at most `2e-6`.
2. The orbit-average state must be identical across the two patched sheets to
   `1e-7`.
3. On every cut, use one locked cyclic permutation of whole fiber barycenters
   as a matched negative control. At `pre_block`, at most one of five
   checkpoints per structured arm may pass the task gate.
4. The exact source model and system state hashes must remain unchanged.
5. Every task metric, posterior, residual norm, and divergence must be finite.

The shuffle preserves the barycenter marginal and repeats one state across
both target sheets; it changes only which semantic fiber receives that state.
It therefore tests task-specific orbit membership rather than branch removal
alone.

## Locked classification

After validity and control gates, apply this table in order:

| Outcome | Classification |
| --- | --- |
| both structured arms pass all cuts at `>=4/5` and pre-block passes both | `frontend_causal_quotient_closed` |
| analytic passes all cuts, learned pre-block fails | `analytic_only_frontend_closure` |
| learned pre-block fails but reaches `>=4/5` at a later cut | `learned_frontend_requires_transformer_synthesis` |
| either structured arm never reaches `>=4/5` | `structured_frontend_not_causally_sufficient` |
| any other valid arm/cut pattern | `mixed_frontend_causal_closure` |
| source, replay, state, numerical, or shuffle contract fails | `invalid` |

For the first two block-0 sublayers, also report the four-regime Reynolds
classification per seed and shift:

```text
propagated fails / actual fails: cover still required
propagated fails / actual passes: invariant synthesis
propagated passes / actual passes: quotient already closed
propagated passes / actual fails: quotient corruption
```

This mechanistic label is secondary to the front-end causal gate and cannot be
used to rewrite it after outcome inspection.

## Lifecycle and stopping rule

Run a single `raw_calibrated`, seed-7 lifecycle shakedown. Because raw is
descriptive, this does not expose a primary structured-arm cell. The
shakedown must validate source loading, activation replay, paired averaging,
continuation, shuffle construction, serialization, and exact resume. It is
never pooled.

Then execute all ten primary structured cells and the five raw descriptive
cells. No seed, cut, threshold, control, cohort, or task metric may change
after the first structured patch outcome is visible.

If the front-end gate passes, close the causal sufficiency question for the
retained calibrated cohort. Do not add a probe, observer, representation loss,
or model training. If it fails, localize the first passing frozen cut; do not
train before that localization is reported.

## Scope boundary

Exact orbit averaging is an oracle intervention using known synthetic fiber
membership. It tests whether an invariant state is sufficient for the frozen
computation, not whether a deployable system can discover fiber membership.
The result is conditioned on the answer decoder, five retained d8/N3
checkpoints, the calibrated synthetic generator, and two held-out shifts. It
does not establish arbitrary nuisance groups, natural-language tasks, or
architecture-population prevalence.
