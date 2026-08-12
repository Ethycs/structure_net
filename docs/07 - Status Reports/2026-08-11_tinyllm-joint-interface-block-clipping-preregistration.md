# TinyLLM Joint-Interface Parameter-Block Clipping Preregistration

**Status:** FROZEN BEFORE ANY BLOCK-CLIPPED FIT

**Date:** 2026-08-11

**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED POST-DIAGNOSTIC INTERVENTION`

**Hypothesis:** `tinyllm-joint-interface-block-clipping-v1`

## Question

Does removing cross-block gradient-norm coupling allow the learned calibrated
sensor to retain one declared physical cosine convention through the same
frozen TinyLLM continuation?

The outcome-known Stage A campaign trained one AdamW optimizer over the learned
encoder, scalar embedding, and zero-initialized final scalar extractor, then
clipped their combined gradient norm to `1.0`. Learned d6 and d10 both passed
the joint physical endpoint in `0/5` seeds. Its registered no-training
attribution found no population-wide pure-starvation or persistent-conflict
mechanism. It also measured a descriptive initial encoder suppression ratio of
`.00186–.00339` in every learned cell because the zero final head received a
gradient hundreds of times larger than the direct sensor gradient.

This experiment is prospective evidence for that broader clipping mechanism.
The descriptive attribution cannot itself establish causality.

## Sealed sources and population

Use the ten learned calibrated equivariant source cells:

```text
preset:    d6, d10
condition: learned_calibrated_equivariant
seed:      7, 17, 29, 41, 53
```

Every cell starts from the same architecture-replication source checkpoint and
the same zero final extractor as Stage A. Before training, require exact source
result/checkpoint hashes, source training-tensor and minibatch hashes, target
permutation hashes, and the corresponding Stage A initial interface-state
digest.

Outcome-known comparators are fixed external evidence and are not pooled as
new arms:

- Stage A campaign SHA-256
  `65ab4b4e887212c4754cf918908cd5e3f04727af4d7876de1d3aa749bc50ac51`;
- Stage A result manifest
  `3299a6cd2edf8816b8bb65ef1ddfb7dfc18f0f6edd41ff09730af632580fc9f3`;
- gradient-attribution v2 campaign SHA-256
  `a3540216800a0cccf0d3725cf349f8a5c91bf01b8680d44c814afd8f4fa6ba25`;
- Stage A analytic positive-control counts: d6 `5/5`, d10 `4/5`;
- Stage A learned global-clip counts: d6 `0/5`, d10 `0/5`;
- all Stage A pair-shuffled counts: `0/5`.

## Matched arms

Train two independent reloads per source cell:

1. `physical_true`: all three objectives use physical `u = cos(phi)`;
2. `pair_shuffled`: targets are permuted between fiber pairs using the exact
   sealed Stage A permutation, while opposite sheets retain the same target.

This preserves target marginal and quotient pairing while destroying the
examplewise input-to-target relation.

## Sole intervention

Keep the Stage A trainable set unchanged:

- learned equivariant encoder;
- scalar embedding;
- affine final scalar extractor initialized to zero.

Keep every transformer, token embedding, position embedding, layer norm, and
LM-head parameter frozen.

Use one AdamW optimizer with the Stage A learning rate and weight decay. After
backpropagating the unchanged sum of losses, partition trainable gradients into
the three declared parameter blocks. For each block `k`, independently apply

```text
c_k = min(1, 1 / ||g_k||)
g_k <- c_k g_k
```

There is no subsequent global norm clip. The total post-clip norm may therefore
reach `sqrt(3)`. A block with no gradient has coefficient `1.0`.

This is the only optimizer change. Do not add a warm start, loss reweighting,
learning-rate schedule, extra steps, head initialization change, transformer
update, or per-seed adjustment.

## Locked training protocol

| quantity | value |
| --- | ---: |
| examples | `4,096` exact source examples |
| updates | `600` |
| pair-balanced minibatch | `64` |
| optimizer | AdamW |
| learning rate | `3e-4` |
| weight decay | `.01` |
| per-block clip ceiling | `1.0` |
| objective weights | task CE `1`; sensor MSE `1`; final MSE `1` |

Reuse the complete sealed pair schedule. There is no validation selection,
early stopping, hyperparameter search, or visible-shift tuning.

Record at step 1 and every 25 steps:

- all three losses and their sum;
- pre-clip norm and applied coefficient for each parameter block;
- equivalent one-global-clip coefficient computed from the same pre-clip
  gradients;
- total pre-clip and post-block-clip norms.

These optimization measurements are secondary and cannot rescue a failed
task/representation endpoint.

## Held-out endpoints

Reuse the exact Stage A composition and extrapolation cohorts and conditional
branch-probe protocol. Evaluate the physical scalar at:

- learned front-end output;
- full depth after final layer normalization and the trained scalar extractor.

At each cut and shift, require simultaneously:

```text
Pearson correlation(predicted cosine, physical cosine) >= .90
conditional branch balanced accuracy                    <= .55
conditional branch log-loss gain                        <= .02
fixed interval-decoder exact-bin accuracy                >= inherited source floor
```

A seed passes an arm only if all endpoints pass at both cuts under both shifts.
Counts may not be pooled across cuts, shifts, endpoints, or seeds.

## Population gate and classifications

The intervention repairs the architecture-family interface only if:

1. `physical_true` passes at least `4/5` seeds separately in d6 and d10;
2. `pair_shuffled` passes at most `1/5` seeds separately in d6 and d10;
3. every cell replays the exact source and Stage A initial state, keeps the
   backbone unchanged, reloads its checkpoint and diagnostics, and remains
   finite.

| Outcome | Classification | Next action |
| --- | --- | --- |
| both learned populations pass, controls specific | `parameter_block_clipping_repairs_physical_interface` | cross-block coupling is causally sufficient under the declared protocol; stop before full-interface fine-tuning |
| one learned population passes | `architecture_conditional_block_clipping_repair` | reject portable optimizer repair; close clipping sweeps and execute the licensed full-interface stage |
| neither learned population passes | `parameter_block_clipping_insufficient` | close optimizer repair and execute the licensed full-interface stage |
| shuffled specificity fails | `specificity_control_failed` | invalidate causal attribution and audit leakage/capacity |
| source, initial-state, or artifact validity fails | `invalid` | repair systems only; preserve the failed root |

No post-outcome threshold, seed substitution, global/block mixture, loss-weight
sweep, warm-start duration, or endpoint-only recalibration is licensed.

## Artifact root

Primary campaign:

`data/experiments/tinyllm_joint_interface_block_clipping/20260811_d6_d10_preregistered`

Any systems shakedown uses a separate root and cannot enter the aggregate.

