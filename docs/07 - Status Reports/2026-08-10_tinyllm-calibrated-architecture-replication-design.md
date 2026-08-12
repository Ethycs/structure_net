# TinyLLM calibrated architecture-family replication design

**Status:** COMPLETED — SEE MEASURED REPORT; THIS FILE REMAINS THE PRE-OUTCOME DESIGN  
**Date:** 2026-08-10  
**Conformance:** NAL-STD-EXPERIMENT `DESIGN`, prospective matched training plus
frozen causal intervention  
**Hypothesis:** `tinyllm-calibrated-architecture-replication-v1`  
**Preflight schema:**
`nal.tinyllm-calibrated-architecture-replication-preflight.v1`

## Decision question

The calibrated d8/N3 result is replicated across initialization seeds, but its
architecture prevalence is unknown. This study asks:

> Does the calibrated analytic or learned-equivariant front end expose a
> representation- and causally-sufficient quotient across the matched d6,
> d8, and d10 TinyLLM preset family?

The d8 result is a known anchor. Only d6 and d10 are fresh architecture cells.
The presets jointly change depth, width, and head count, so this is an
**architecture-family replication**, not a factorized claim about depth or
width individually.

## Why this is the next admissible scope

The [frontier audit](2026-08-10_tinyllm-interpretability-frontier.md) closes
same-scope penalties, probes, writers, topology scans, causal-front rescans,
and acquisition-count studies. Architecture-family replication changes the
smallest remaining assumption while retaining the successful observation,
target, nuisance group, front ends, training task, and causal intervention.

No optimizer step was executed during this preflight. No d6 or d10 primary
outcome exists or was inspected.

## Locked retained anchors

| Artifact | SHA-256 | Evidence role |
| --- | --- | --- |
| d8 calibrated source campaign | `80623a7283fed3d902a1cbf9fb58afe2a2fef82ab86406212f907de40a96c501` | known anchor |
| calibrated source implementation | `73e152b9b6c7ab51ad08e6a80786c32a71566f9d4fb410ff2cf3df53761faf77` | pinned reusable implementation |
| 15-result source manifest | `34bf25feb896abc9b9e06386b474fc6795c94566a23cd97a06795435fba64d68` | source-integrity contract |
| d8 causal-closure campaign | `1457fdcce224157c5d8b19c11317b3d3c47cc7d8575097c78e76ba8798431b14` | known causal anchor |

The preflight revalidated every hash, all 15 source results, source completion,
and the valid passing causal-closure verdict. The pinned source runners must
not be edited. A dedicated architecture-replication runner will import their
generic model/front-end and causal-analysis primitives while owning a new
schema, fingerprint, and evidence role.

## Matched architecture grid

| Preset | Layers | Heads | Width | Model parameters | Role |
| --- | ---: | ---: | ---: | ---: | --- |
| d6 | 6 | 6 | 384 | 29,956,608 | fresh primary/comparator cells |
| d8 | 8 | 8 | 512 | 50,965,504 | retained outcome-known anchor |
| d10 | 10 | 10 | 640 | 81,418,240 | fresh primary/comparator cells |

Use all three conditions at each fresh preset:

- `raw_calibrated`;
- `analytic_calibrated`;
- `learned_calibrated_equivariant`.

Use seeds `7`, `17`, `29`, `41`, and `53`. The fresh campaign therefore has
`2 presets x 3 conditions x 5 seeds = 30` cells. Reuse all 15 d8 anchor cells;
do not retrain d8.

The raw arm is required. It is the only matched comparator that can distinguish
structured architectural closure from a capacity-only explanation. Its low d8
task accuracy makes it non-primary: its result changes specificity and
interpretation but cannot rescue a failed structured arm.

## Training contract

Retain the source protocol exactly:

```text
training examples       4096
optimizer steps          600
batch size                64, paired by exact task fiber
optimizer              AdamW
learning rate          3e-4
weight decay            0.01
gradient clip            1.0
probe train/validation/test  2048/512/1024
probe steps              240
```

For each seed, every condition and every preset receives the same generated
examples and minibatch indices. The preflight regenerated all 15 preset/seed
protocols and verified identical training-data and minibatch-schedule hashes
across d6, d8, and d10 for every seed.

The initial TinyLLM state must match across conditions within a preset and
seed. The architecture changes across presets by design. Persist the training
data, minibatch, initial/final model, initial/final complete-system, checkpoint,
front-end, implementation, preregistration, and scientific-fingerprint hashes.

## Pre-training validity contracts

Before scheduling a primary cell:

1. rerun the calibrated observation identifiability contract;
2. verify the analytic canonicalizer recovers the declared absolute cosine
   coordinate from observations without phase or target access at correlation
   `>=0.99` and RMSE `<=0.05` on composition / `<=0.08` on extrapolation;
3. verify the learned encoder's declared acquisition-group equivariance to
   numerical tolerance before and after training;
4. verify all retained anchor and source-implementation hashes;
5. verify same-seed data and minibatch hashes across presets and conditions;
6. verify the exact composition and extrapolation fiber cohorts and semantic
   shuffle constructor before any primary intervention.

Failure of any contract makes the campaign `invalid`; it cannot be converted
into a scientific negative.

## Representation endpoint

Measure at front-end output and full depth on composition and extrapolation:

```text
cosine Pearson correlation                         >= 0.90
conditional branch balanced accuracy               <= 0.55
conditional log-loss gain over cosine-only null    <= 0.02
```

All three conditions must hold simultaneously in every cut/shift cell. The
conditional log-loss ceiling is explicit here even though the historical d8
aggregator omitted it from its Boolean helper. The stored d8 primary-arm
values all pass retrospectively (`-0.000696` through `0.000837`), so adding the
declared endpoint changes neither the anchor result nor its pedigree.

Fit evaluation probes only on the locked training/validation cohorts after
the task model is frozen. The learned front-end training representation is not
reused as the held-out branch estimator.

## Natural-task adequacy guard

A useless model can pass a relative causal patch gate. Therefore each fresh
structured cell must also keep exact-bin accuracy within three percentage
points of the outcome-known, same-condition, same-seed d8 anchor separately on
composition and extrapolation:

```text
accuracy(preset, condition, seed, shift)
    >= accuracy(d8, condition, seed, shift) - 0.03.
```

The exact seedwise floors are serialized by the preflight before any new
training. This is a conservative family-stability claim, not a tuning target.
Cross-entropy and circular error remain reported but do not replace a failed
accuracy guard.

## Frozen causal endpoint

Automatically run the established exact task-orbit intervention on every
completed checkpoint. Capture the full token-by-channel residual at:

1. `pre_block`;
2. `block0_post_attention`;
3. `block0_post_mlp`;
4. `full`.

At every cut, repeat the exact two-sheet fiber barycenter across both sheets
and continue through the frozen model. A patch passes only when, relative to
the unchanged natural checkpoint:

```text
exact-bin accuracy loss       <= 0.03
circular-error increase       <= pi/16 radians
target cross-entropy increase <= 0.10 nats.
```

Composition and extrapolation must both pass. Retain the propagated versus
actual barycenter/Reynolds-defect classification for block-0 attention and
MLP as a secondary mechanism. It cannot rescue a failed front-end claim.

## Controls

- Continuing an unmodified captured state must reproduce the natural posterior
  within `2e-6` maximum absolute error.
- Every orbit-barycenter state must be identical across its two repeated sheets
  within `1e-7`.
- A locked cyclic permutation of whole semantic-fiber barycenters is the
  task-changing control. At most one of five structured checkpoints per preset
  and arm may pass at `pre_block`.
- Model and complete-system hashes must remain unchanged during held-out probes
  and causal analysis.
- All metrics and stored arrays must be finite.
- Raw cells are analyzed identically but remain specificity comparators because
  their natural task adequacy is not a positive control.

## Replication unit and primary gate

The checkpoint seed is the replication unit. A structured seed passes only if
all of the following pass jointly on both shifts:

- representation endpoints at front-end and full depth;
- the same-seed d8-relative natural-task adequacy guard;
- orbit-barycenter task sufficiency at all four causal cuts;
- replay, state-identity, numerical, and semantic-shuffle controls.

Each primary arm passes a preset at `4/5` seeds. The architecture-family
hypothesis passes only if **both analytic and learned-equivariant arms pass in
both fresh presets**. The known d8 anchor is required for validity but is not
counted as fresh confirmation and cannot outvote a failed d6 or d10 preset.

## Locked interpretation table

| Fresh outcome | Classification |
| --- | --- |
| both structured arms pass d6 and d10; raw fails its joint representation/causal endpoint in both | `structured_family_replication_with_specificity` |
| both structured arms pass d6 and d10; raw also passes somewhere | `structured_family_replication_without_raw_specificity` |
| analytic passes both presets; learned fails either | `analytic_closure_stable_learned_family_dependent` |
| both arms pass one fresh preset but not the other | `preset_dependent_structured_closure` |
| analytic task/causal result fails after its direct canonicalizer contract passes | `structured_closure_not_architecture_stable` |
| source, identifiability, equivariance, replay, hash, fiber, numerical, or control contract fails | `invalid` |

No result licenses a separate depth, width, or head-count claim.

## Lifecycle and stopping rule

1. The no-training preflight must pass and its JSON must be sealed.
2. Unit tests must validate grid cardinality, parameter accounting, source
   anchors, same-seed protocol hashes, gates, classification, and exact resume.
3. Run lifecycle-only two-step d10 cells outside the primary output root.
4. Run a representative d10 CUDA shakedown with a 4 GB reservation per cell,
   then an intended two-slot concurrency shakedown. Shakedown outcomes are
   never pooled.
5. Freeze the runner and dated preregistration hashes before the first
   600-step d6/d10 cell.
6. Run all 30 fresh cells and automatically execute the causal intervention on
   every completed checkpoint. Do not select checkpoints after training.
7. Stop after the locked classification. Do not add steps, seeds, probes,
   thresholds, or an intermediate preset to rescue a failure.

## Resource budget

The retained d8 training-plus-probe campaign consumed `1,242.41` aggregate
GPU-seconds for 15 cells; the d8 causal campaign consumed another `242.07`
GPU-seconds. Scaling each by `layers * width^2` gives:

| Work | Projected aggregate GPU time |
| --- | ---: |
| fresh d6+d10 training and probes | 49.18 min |
| fresh d6+d10 causal analysis | 9.58 min |
| total | 58.76 min |
| ideal two-slot wall time | 29.38 min |

The stored d8 peaks were approximately `1.02` GB for structured cells and
`1.52` GB for raw cells. Reserve 4 GB per d10 cell until the representative
shakedown measures the actual peak. Projected new checkpoint storage is
`6.22 GiB`, excluding small causal arrays and JSON.

These are scheduling estimates, not evidence. The CUDA shakedown determines
whether two d10 cells may run concurrently.

## Preflight command

```bash
MPLCONFIGDIR=/tmp/mplconfig pixi run python -m \
  experiments.structure_net.tinyllm_calibrated_architecture_replication_preflight \
  --output /tmp/tinyllm-calibrated-architecture-replication-preflight.json
```

The 2026-08-10 preflight passed with 30 fresh cells, 15 retained anchor cells,
zero optimization steps, zero new outcomes, matching protocol hashes for all
five seeds, and projected total cost of 58.76 GPU-minutes. The output remained
in `/tmp`; it is a design validation, not a promoted experiment artifact.

## Completion note

The dedicated runner, lifecycle shakedowns, dated preregistration, and all 30
fresh cells are complete. The design itself remains unchanged as the
pre-outcome record. The authoritative measured result is the
[calibrated architecture-family replication report](../08%20-%20Analysis/2026-08-10_tinyllm-calibrated-architecture-replication.md): structured representation and causal closure replicate, while the full
architecture-family hypothesis fails its same-seed task-adequacy conjunct.
