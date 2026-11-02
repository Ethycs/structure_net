# TinyLLM Joint Full-Interface Physical Typing Preregistration

**Status:** FROZEN BEFORE ANY FULL-INTERFACE FIT

**Date:** 2026-08-11

**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED CONDITIONAL EXTENSION`

**Hypothesis:** `tinyllm-joint-full-interface-physical-typing-v1`

## Question

Can a learned calibrated sensor carry one declared physical cosine convention
end to end when the TinyLLM residual continuation is allowed to adapt?

This is the conditional Stage B experiment licensed before Stage A outcomes
were inspected. It is reached only after two cheaper explanations failed:

- frozen-backbone equal-weight joint typing passed learned d6/d10 `0/5`;
- prospective parameter-block clipping also passed learned d6/d10 `0/5`.

Analytic frozen-backbone positive controls passed d6 `5/5` and d10 `4/5`.
Thus the declared chart is usable by these architecture families, but the
learned sensor and frozen continuation did not select it jointly.

## Population and exact initialization

Use the same ten learned calibrated equivariant architecture-replication
sources:

```text
preset: d6, d10
seed:   7, 17, 29, 41, 53
```

Each arm begins from the original source model, learned encoder, and scalar
embedding plus the exact zero final scalar extractor used by Stage A. It does
not continue from a failed Stage A or block-clipped final state.

Before training, require exact replay of:

- architecture source result and checkpoint hashes;
- source model and structured-interface state digests;
- Stage A initial interface digest;
- 4,096-example training tensor and complete 600-step pair schedule;
- Stage A target permutation;
- held-out cohort hashes and inherited task floors.

Outcome-known comparators remain external and are not pooled:

| comparator | d6 physical | d10 physical | shuffled |
| --- | ---: | ---: | ---: |
| Stage A global clip | `0/5` | `0/5` | all `0/5` |
| frozen-backbone block clip | `0/5` | `0/5` | all `0/5` |
| analytic frozen-backbone positive control | `5/5` | `4/5` | all `0/5` |

Pinned parent campaigns:

- Stage A SHA-256
  `65ab4b4e887212c4754cf918908cd5e3f04727af4d7876de1d3aa749bc50ac51`;
- block-clipping SHA-256
  `2f7c7cdd5494322ff89e20fb55407c6d4d8de66dde852ca9a8ec67fbc22a2349`.

## Matched arms

Train two independent source reloads per cell:

1. `physical_true`: physical `u = cos(phi)` supervises the sensor, final
   scalar, and fixed interval task decoder;
2. `pair_shuffled`: the exact Stage A pair-preserving target permutation is
   used for all three objectives.

The control preserves target marginal and opposite-sheet pairing while
destroying examplewise semantic correspondence.

## Trainable continuation

Train:

- the learned equivariant sensor encoder;
- scalar embedding;
- token embedding, including its tied LM-head weight;
- position embedding;
- every attention, MLP, and layer-normalization parameter in every transformer
  block;
- final layer normalization;
- final affine scalar extractor.

No separate untied answer-head parameter enters the physical-scalar forward
path. Feedback connections are absent in all source models and may not be
created.

The full model and interface must be checkpointed per arm. Validity requires
an exact checkpoint reload, a changed continuation state, and no change in
model topology or parameter names.

## Objective and optimizer

Retain the exact Stage A objective:

```text
L = CE(P_interval(q(r)), P_interval(y))
    + MSE(sensor_scalar, y)
    + MSE(q(r), y).
```

Retain the exact Stage A optimizer protocol:

| quantity | value |
| --- | ---: |
| examples | `4,096` |
| updates | `600` |
| pair-balanced minibatch | `64` |
| optimizer | AdamW |
| learning rate | `3e-4` for every trainable parameter |
| weight decay | `.01` |
| one global gradient-norm ceiling | `1.0` |
| objective weights | all `1.0` |

There is no block clipping, warm start, differential learning rate, layer
freezing schedule, validation selection, early stopping, or loss reweighting.
Record losses, pre-clip global gradient norm, and model/interface state digests.

## Held-out endpoint

Reuse the exact Stage A composition and extrapolation cohorts and nonlinear
branch probes conditioned on physical cosine. Evaluate at:

- learned front-end scalar;
- full-depth final scalar.

At each cut and shift require simultaneously:

```text
Pearson correlation(predicted cosine, physical cosine) >= .90
conditional branch balanced accuracy                    <= .55
conditional branch log-loss gain                        <= .02
fixed interval-decoder exact-bin accuracy                >= inherited source floor
```

A seed passes only if all four conditions pass at both cuts on both shifts.

## Population gate and locked outcomes

Success requires at least `4/5` physical-true passes separately for d6 and
d10, at most `1/5` shuffled passes separately for each preset, and all ten
cells valid.

| Outcome | Classification | Meaning and next action |
| --- | --- | --- |
| both populations pass; controls specific | `full_interface_physical_typing_architecture_stable` | the frozen continuation was the remaining obstruction; retain full model/interface checkpoints and test transfer on a new identifiable task |
| one population passes | `architecture_conditional_full_interface_repair` | the construction is not portable; stop this interface branch and study the passing architecture only under a new preregistration |
| neither population passes | `flexible_full_interface_physical_typing_insufficient` | close flexible joint physical supervision; require sign/scale/chart typing by construction in any successor architecture |
| shuffled specificity fails | `specificity_control_failed` | invalidate semantic causality and audit memorization/leakage |
| source, topology, intervention, or artifact contract fails | `invalid` | preserve root; repair systems only |

No partial full-depth success, high absolute correlation, lower training loss,
or post-hoc affine calibration can rescue the joint endpoint. No further loss,
optimizer, warm-start, or seed sweep is licensed after a valid negative result.

## Artifact root

Primary campaign:

`data/experiments/tinyllm_joint_full_interface/20260811_d6_d10_preregistered`

Systems shakedowns use separate roots and cannot enter the aggregate.

