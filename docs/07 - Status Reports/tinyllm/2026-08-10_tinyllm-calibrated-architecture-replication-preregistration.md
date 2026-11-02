# TinyLLM calibrated architecture-family replication preregistration

**Status:** PREREGISTERED — NO FRESH D6/D10 QUALITY OUTCOME GENERATED OR
INSPECTED  
**Date:** 2026-08-10  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, prospective matched
training plus automatic frozen causal intervention  
**Hypothesis:** `tinyllm-calibrated-architecture-replication-v1`  
**Schema:** `nal.tinyllm-calibrated-architecture-replication.v1`

## Decision question

Does calibrated front-end causal quotient closure replicate across the
matched d6/d8/d10 TinyLLM preset family rather than only across initialization
seeds of d8?

The retained d8 population is an outcome-known anchor. Fresh primary evidence
comes only from d6 and d10. Because the presets jointly vary layer count, head
count, and embedding width, the experiment tests family replication and may
not be interpreted as a factorized depth, width, or head-count effect.

## Pre-outcome state

The no-training preflight passed before this record was frozen. It performed
zero optimizer steps and inspected zero new model outcomes. It verified:

- the retained d8 source and causal campaign hashes;
- the observation/target identifiability contract;
- analytic canonicalizer fidelity on both held-out shifts;
- the learned encoder's architectural acquisition-group contract;
- exact d6/d8/d10 parameter counts;
- same-seed training-data and minibatch-schedule identity across presets;
- the 30-cell fresh grid and the scheduling/storage budget.

The detailed rationale and cost calculation are in the
[design record](2026-08-10_tinyllm-calibrated-architecture-replication-design.md).
This preregistration is authoritative if the two records conflict.

## Locked retained anchors

| Artifact | SHA-256 | Role |
| --- | --- | --- |
| d8 calibrated campaign | `80623a7283fed3d902a1cbf9fb58afe2a2fef82ab86406212f907de40a96c501` | outcome-known family anchor |
| calibrated implementation | `73e152b9b6c7ab51ad08e6a80786c32a71566f9d4fb410ff2cf3df53761faf77` | reused generic training/front-end implementation |
| d8 15-result manifest | `34bf25feb896abc9b9e06386b474fc6795c94566a23cd97a06795435fba64d68` | source integrity |
| d8 causal-closure campaign | `1457fdcce224157c5d8b19c11317b3d3c47cc7d8575097c78e76ba8798431b14` | outcome-known causal anchor |

The new runner must validate these artifacts before scheduling any cell. The
d8 checkpoints are not retrained and do not count toward fresh replication.

## Fresh grid and replication unit

Fresh presets:

| Preset | Layers | Heads | Width | Model parameters |
| --- | ---: | ---: | ---: | ---: |
| d6 | 6 | 6 | 384 | 29,956,608 |
| d10 | 10 | 10 | 640 | 81,418,240 |

Conditions:

1. `raw_calibrated`;
2. `analytic_calibrated`;
3. `learned_calibrated_equivariant`.

Seeds are `7`, `17`, `29`, `41`, and `53`. This yields exactly
`2 x 3 x 5 = 30` fresh cells. The checkpoint seed is the replication unit.

The analytic and learned-equivariant arms are primary. The raw arm is a
matched specificity comparator. Because its retained d8 natural task accuracy
is low, raw causal patch success alone is descriptive and cannot validate or
invalidate a structured arm.

## Locked data and optimization

Use the calibrated N3 generator and source task without modification:

```text
training examples       4096
optimizer steps          600
paired batch size          64
optimizer              AdamW
learning rate          3e-4
weight decay            0.01
gradient clip            1.0
vector channels           16
```

Use source seeds and offsets for training examples and paired minibatches. For
each seed, all presets and conditions must have identical training-data and
minibatch-schedule hashes. Model initialization is identical across conditions
within a preset/seed; architecture necessarily differs across presets.

Train only ordinary task cross-entropy. Do not add a representation penalty,
adversary, contrastive loss, auxiliary semantic head, or checkpoint selection.
Run exactly 600 updates for every primary cell.

Persist hashes for the training tensors, minibatch schedule, initial/final
TinyLLM state, initial/final complete-system state, model checkpoint, front-end
checkpoint, runner sources, preflight, preregistration, and scientific
fingerprint.

## Held-out representation endpoint

After training, freeze the complete system. Fit fresh nonlinear conditional
branch probes and cosine-only nulls on the locked held-out protocol:

```text
probe train examples        2048
probe validation examples    512
probe test examples         1024 per regime
probe steps                  240
```

Measure at `frontend` and `full` on composition and extrapolation. A cell
passes only when all three conditions hold simultaneously:

```text
cosine Pearson correlation                         >= 0.90
conditional branch balanced accuracy               <= 0.55
conditional log-loss gain over cosine-only null    <= 0.02
```

The conditional log-loss endpoint is part of the Boolean gate. It was stored
but not included in the historical d8 helper. All retained d8 structured cells
pass it retrospectively, so the correction does not change the anchor result.
The held-out probe, not any learned front-end quantity, decides branch leakage.

## Natural-task adequacy guard

A fresh structured cell must retain exact-bin accuracy within three percentage
points of its outcome-known, same-condition, same-seed d8 anchor separately on
composition and extrapolation:

```text
accuracy(fresh preset, condition, seed, regime)
  >= accuracy(d8, condition, seed, regime) - 0.03.
```

The preflight serializes all 20 exact floors before new training. Both regimes
must pass. Raw cells have no adequacy floor and remain comparators.

## Automatic frozen causal intervention

Every completed checkpoint must immediately enter the same causal pipeline;
there is no checkpoint selection step. Use the exact composition and
extrapolation cohorts from the retained closure campaign:

| Regime | Seed | Examples | Exact two-sheet fibers | SHA-256 |
| --- | ---: | ---: | ---: | --- |
| composition | 1399 | 1024 | 512 | `b025623e23f534ca3670f49f1bafc1c3979d1ca834aec2223f9c3166be8f52a6` |
| extrapolation | 2408 | 1024 | 512 | `f2f1e8c2c06b38fbecf856f0679e9861d9fab3e1c2a41358e55043f4d309a214` |

Capture the complete token-by-channel residual at:

1. `pre_block`;
2. `block0_post_attention`;
3. `block0_post_mlp`;
4. `full`.

At every cut, average the two exact target-fiber sheets, repeat the barycenter
across both rows, and continue it through the unchanged remainder. A cut/shift
passes only when all three relative task conditions hold:

```text
exact-bin accuracy loss       <= 0.03
circular-error increase       <= pi/16 radians
target cross-entropy increase <= 0.10 nats.
```

A seed passes the causal endpoint only when every cut passes on both shifts.
Also retain the propagated-versus-actual Reynolds/Jensen defect and the
four-regime block-0 attention/MLP classification as secondary mechanism data.
Those labels cannot rescue a failed primary endpoint.

## Validity and specificity controls

1. The calibrated identifiability contract must pass before training.
2. The analytic canonicalizer must retain correlation `>=0.99` and RMSE
   `<=0.05` on composition / `<=0.08` on extrapolation without phase or target
   access.
3. The learned encoder's declared positive-similarity acquisition-group
   contract must have maximum absolute canonical-feature error `<=1e-5` both
   in the preflight and for each trained encoder.
4. Continuing an unmodified captured state must replay the natural posterior
   with maximum absolute error `<=2e-6`.
5. Patched barycenters must be identical across both repeated sheets to
   `<=1e-7`.
6. TinyLLM and complete-system state hashes must remain unchanged throughout
   held-out probes and causal analysis.
7. All metrics and stored arrays must be finite.
8. Apply one locked cyclic permutation of whole semantic-fiber barycenters.
   At `pre_block`, at most one of five seeds may pass in each fresh preset and
   primary arm.

Any failure of controls 1--7 makes the campaign `invalid`. Control 8 is a
population specificity requirement and also makes the campaign `invalid` if
its ceiling is exceeded. A semantic-shuffle result is evaluated at population
level and is not included in an individual seed's joint gate.

## Seed, arm, preset, and family gates

A structured seed passes jointly only if all of these pass on both shifts:

- representation at front-end and full depth;
- natural-task adequacy;
- orbit-barycenter sufficiency at all four causal cuts;
- identifiability, front-end contract, replay, state identity, state
  immutability, and numerical validity.

An arm passes a preset when at least four of five seeds pass jointly and its
semantic-shuffle count is at most one. The family hypothesis passes only when
both primary arms pass **both d6 and d10**. The known d8 anchor is required for
validity but cannot outvote a failed fresh preset.

## Locked classification

Apply this table in order:

| Outcome | Classification | Primary pass |
| --- | --- | --- |
| both structured arms pass d6 and d10; raw joint endpoint fails both presets | `structured_family_replication_with_specificity` | yes |
| both structured arms pass d6 and d10; raw joint endpoint passes a preset | `structured_family_replication_without_raw_specificity` | yes |
| analytic passes both presets; learned fails either | `analytic_closure_stable_learned_family_dependent` | no |
| both structured arms pass one fresh preset but not the other | `preset_dependent_structured_closure` | no |
| analytic fails either fresh preset after validity controls pass | `structured_closure_not_architecture_stable` | no |
| any source, identifiability, group, replay, state, fiber, numerical, or semantic-shuffle contract fails | `invalid` | no |
| any remaining valid pattern | `mixed_architecture_family_result` | no |

No post-outcome threshold, checkpoint, seed, probe, or classification amendment
is allowed.

## Lifecycle before primary execution

1. Run unit tests for the preflight, config lock, grid, representation
   log-loss gate, task-floor attachment, population classification, semantic
   shuffle invalidation, artifact hashes, and resume recognition.
2. Run a two-step d10/raw seed-7 lifecycle on CPU or CUDA in a separate output
   root. It is never pooled.
3. Run a representative d10 CUDA shakedown with a 4 GB reservation per cell.
4. Run an intended two-slot d10 concurrency shakedown. Reduce concurrency if
   measured free memory requires it; do not change the primary cell.
5. Freeze the implementation and this preregistration hash before launching a
   600-step cell.
6. Execute all 30 cells with exact resume. A retry repeats the identical cell;
   it does not change its seed or data.

The preflight projects `58.76` aggregate GPU-minutes including probes and
causal analysis, approximately `29.38` ideal minutes at two slots, and
`6.22 GiB` of new checkpoints. These are planning estimates. Measured d10
shakedown memory decides concurrency.

## Stopping rule

After the locked aggregate is available, stop and report it. Do not add an
intermediate preset, more training steps, an extra calibration feature, a new
probe, or a relaxed task floor to rescue a failure.

If the family gate passes, the next claim may concern a richer identifiable
group or a new task family. If it fails, localize the failure to natural task
adequacy, representation, causal use, learned optimization, or control
validity; do not return to same-scope residual penalties.

## Artifact and backup contract

Primary output root:

```text
data/experiments/tinyllm_calibrated_architecture_replication/
    20260810_d6_d10_preregistered/
```

Each cell stores owned model/front-end checkpoints, a typed result, and causal
diagnostic arrays. The campaign stores the preflight, source provenance,
implementation-source hashes, scheduler record, aggregate gate, and exact
resume fingerprint.

After verification, add the new data under the repository's existing `data.dvc`
root, push its objects to the configured DVC remote, commit the pointer to the
configured lakeFS branch, and verify immutable object hashes at the lakeFS
commit before promoting the analysis or meta-hypothesis record.

