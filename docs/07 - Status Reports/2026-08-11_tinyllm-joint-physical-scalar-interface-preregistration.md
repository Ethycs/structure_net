# TinyLLM Joint Physical Scalar Interface Preregistration

**Status:** FROZEN BEFORE ANY JOINT-INTERFACE FIT

**Date:** 2026-08-11

**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`

**Hypothesis:** `tinyllm-joint-physical-scalar-interface-v1`

## Question

Can the architecture-family failure of the frozen typed readout be repaired by
calibrating one declared physical cosine coordinate jointly at the observed
sensor, scalar injection, frozen TinyLLM continuation, and final scalar
extraction, without changing any transformer, token-embedding, layer-norm, or
LM-head parameter?

The preceding frozen diagnostic is the registered comparator. Its typed
interval readout passed `d6/analytic 5/5`, `d6/learned 4/5`, `d10/analytic
5/5`, and `d10/learned 1/5`; it repaired five of ten source task failures, but
the architecture-family gate failed. No new endpoint-only fit is licensed.

## Population and source evidence

The primary population is the complete Cartesian product

```text
preset:    d6, d10
front end: analytic_calibrated, learned_calibrated_equivariant
seed:      7, 17, 29, 41, 53
```

The 20 source systems are loaded from
`data/experiments/tinyllm_calibrated_architecture_replication/20260810_d6_d10_preregistered`.
The source campaign, result manifest, model/front-end artifacts, source
training-tensor hash, and minibatch-schedule hash MUST match their sealed
records before a cell can run. The frozen-readout comparator is loaded from
`data/experiments/tinyllm_frozen_interval_readout_decomposition/20260811_d6_d10_preregistered`.

The source model is an observed starting state, not prospective evidence. All
joint-interface outcomes remain unread when this document is frozen.

## Matched arms

Each source cell produces two matched interface fits from independent reloads
of the same source weights:

1. `physical_true`: supervision is the generator's physical
   `u = cos(phi)` coordinate.
2. `pair_shuffled`: the cosine targets are deterministically permuted between
   fiber pairs, with both opposite sheets in a pair retaining the same
   permuted target. This preserves the quotient form and target marginal while
   destroying the input-to-target relation.

The analytic condition is the positive control. Its sensor canonicalizer is
fixed and has no trainable parameters. The learned condition is the primary
architectural result. The true and shuffled fits receive identical source
states, examples, pair minibatches, optimizer hyperparameters, update count,
and initialization policy.

No raw-input arm is repeated because the present question is calibration of
the already identified structured scalar interface, not rediscovery of the
front-end comparison.

## Intervention

Every source transformer parameter is frozen, including token and position
embeddings, all attention and MLP blocks, final layer norm, and LM head.

The only trainable modules are:

- the existing learned equivariant encoder, for the learned condition only;
- the existing scalar embedding from one physical scalar into the frozen
  residual stream;
- a new affine scalar extractor from the normalized final query residual.

The final scalar extractor is initialized to zero in both matched arms. The
learned encoder and scalar embedding start from the corresponding source
checkpoint. The analytic canonicalizer remains exact and fixed.

For target scalar `y`, sensor scalar `s`, normalized final residual `r`, and
final scalar `q(r)`, minimize

```text
L = CE(P_interval(q(r)), P_interval(y))
    + MSE(s, y)
    + MSE(q(r), y).
```

`P_interval` is fixed: it places Gaussian logits on the 16 ordered centers
from `-1` through `1`, with width `2/15`. It has no trainable parameters and
does not wrap endpoints. The scalar is not clipped inside the training
decoder; physical MSE supplies the range constraint.

For the analytic condition the sensor MSE is measured but constant. For the
shuffled condition, all three terms use the pair-shuffled target. No loss
weight is tuned: all three coefficients are `1.0`.

## Locked optimization protocol

The intervention reuses the source task protocol:

| quantity | value |
| --- | ---: |
| training examples | 4,096 |
| updates | 600 |
| pair-balanced minibatch size | 64 |
| optimizer | AdamW |
| learning rate | `3e-4` |
| weight decay | `0.01` |
| gradient-norm ceiling | `1.0` |

The original per-seed examples and pair minibatch schedule are reused exactly.
The target shuffle is deterministic from the study shuffle seed, preset,
condition, and model seed. There is no validation selection, early stopping,
learning-rate schedule, or hyperparameter search.

## Held-out cohorts and cuts

The primary held-out cohorts are the exact 1,024-example composition and
extrapolation cohorts already sealed in the source and frozen-readout studies.
No primary example participates in interface training.

Evaluate the typed scalar at:

- `frontend`: the analytic or learned sensor scalar before scalar embedding;
- `full`: the affine scalar extracted after the frozen transformer's final
  layer norm.

At both cuts, fit a held-out nonlinear branch probe conditioned explicitly on
exact cosine, using the existing disjoint train/validation protocol. The
training-time physical heads are not used as branch estimators.

## Per-cut endpoint

For every cut and primary regime, success requires simultaneously:

```text
Pearson correlation(predicted cosine, physical cosine) >= 0.90
conditional branch balanced accuracy                         <= 0.55
conditional branch log-loss gain over cosine-only null       <= 0.02
fixed interval-decoder exact-bin accuracy                     >= inherited source floor
```

The inherited source floor is the source checkpoint's registered accuracy
minus at most three percentage points, clipped at zero. A seed passes an arm
only if all four conditions pass at both cuts on both composition and
extrapolation. Counts may not be pooled across endpoints, cuts, shifts, or
seeds.

Cosine RMSE, affine slope/intercept, range, opposite-sheet scalar difference,
cross-entropy, absolute bin error, and bin coverage are secondary calibration
diagnostics. They cannot rescue a failed primary gate.

## Population decision rule

The frozen-backbone joint physical interface is architecture-stable only if:

1. `physical_true` passes in at least four of five seeds separately for both
   `d6/learned` and `d10/learned`;
2. the analytic positive control passes in at least four of five seeds
   separately for d6 and d10;
3. `pair_shuffled` passes in at most one of five seeds in every stratum;
4. all 20 source cells are valid, every source checkpoint replays, every
   frozen model digest is unchanged, and saved diagnostics reload exactly.

The source typed-readout counts remain a fixed external comparator and are not
re-aggregated with the new trained arms.

## Outcome meanings and stop rules

| Outcome | Interpretation | Next action |
| --- | --- | --- |
| analytic and learned gates pass | a jointly typed interface is sufficient with a frozen backbone | stop; do not fine-tune the transformer |
| analytic passes, learned fails | structural sensor family or frozen continuation prevents stable learned calibration | preregister the conditional full-interface stage; do not tune Stage A |
| front end passes, full fails | sensor is physically calibrated but the frozen continuation cannot transport the coordinate | conditional full-interface stage is licensed |
| branch gate fails while cosine/task pass | scalar calibration preserves a branch-correlated error channel | inspect sensor residuals; do not call it a quotient |
| analytic positive control fails | implementation, optimization, or gate calibration is invalid | audit; do not interpret the learned arm |
| shuffled specificity fails | capacity or leakage makes the intervention non-specific | invalidate the causal claim |

The conditional full-interface stage is not part of this Stage A evidence. If
licensed, it receives a new preregistration and artifact root before any such
fit is inspected. No additional frozen endpoint readout, ridge sweep, loss
weight sweep, or seed substitution is allowed.

## Validity and artifacts

Each cell MUST record source hashes, exact data and minibatch hashes, target
permutation hash, frozen-model digests before and after both arms, trainable
parameter names/counts, optimization history, probe seeds, per-example scalar
arrays, strict finite checks, and checkpoint hashes. A cell is independently
resumable only when its scientific fingerprint and all required artifacts
match.

Primary artifact root:

`data/experiments/tinyllm_joint_physical_scalar_interface/20260811_d6_d10_preregistered`

The CUDA shakedown uses a separate systems-only root and cannot enter the
scientific aggregate.
