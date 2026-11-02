# TinyLLM Joint-Interface Gradient Attribution Preregistration

**Status:** FROZEN BEFORE PER-OBJECTIVE GRADIENT INSPECTION

**Date:** 2026-08-11

**Conformance:** NAL-STD-EXPERIMENT `REGISTERED POST-OUTCOME DIAGNOSTIC`

**Hypothesis:** `tinyllm-joint-interface-gradient-attribution-v1`

## Known outcome and question

The prospective frozen-backbone joint-interface campaign is outcome-known. Its
analytic controls passed d6 `5/5` and d10 `4/5`, every pair-shuffled control
passed `0/5`, and both learned populations passed `0/5`. Learned full-depth
scalars were often accurate while learned sensor scalars remained compressed
or sign-reversed. Final logged sensor MSE was much larger than final-state MSE,
and pre-clip total gradient norms were much larger than the global ceiling.

This diagnostic asks:

> Did the equal-weight objective plus one global gradient-norm clip suppress
> the direct physical-sensor update across parameter blocks, and did
> downstream objectives oppose sensor calibration at the trained state?

It does not retest Stage A and cannot rescue its failed gate.

## Population and states

Use all 20 sealed d6/d10 analytic and learned source cells from
`tinyllm-joint-physical-scalar-interface-v1`. The ten learned cells are the
primary population; the ten analytic cells are a fixed-sensor control.

For each cell, reconstruct two exact physical-true interface states:

1. `initial`: sealed source encoder/scalar embedding plus the registered zero
   final extractor;
2. `final`: the saved `physical_true_interface.pt` state.

Evaluate each state on the first and last minibatches of the sealed 600-step
source schedule. Use evaluation mode to remove stochastic dropout and make the
local objective geometry exact and repeatable. No optimizer step, fitted map,
probe, or parameter modification is allowed.

## Exact objective decomposition

Recompute the three registered losses separately:

```text
L_sensor = MSE(sensor scalar, physical cosine)
L_final  = MSE(final scalar, physical cosine)
L_task   = CE(fixed interval decoder(final scalar), target interval posterior)
```

Differentiate each loss with respect to these parameter blocks:

- learned equivariant encoder, when present;
- scalar embedding;
- final scalar extractor.

Store the full flattened gradient arrays. For every objective/block pair,
report gradient norm. For every objective pair in a block, report cosine. Also
report:

```text
g_total = g_sensor + g_final + g_task
global_clip = min(1, 1 / ||g_total across all blocks||)
block_clip_k = min(1, 1 / ||g_total,k||)
cross_block_suppression_k = global_clip / block_clip_k
sensor_descent_ratio = <g_sensor, g_total> / ||g_sensor||^2
```

`cross_block_suppression < 1` means unrelated gradient magnitude in other
parameter blocks reduces this block's step beyond what a separate block clip
would do. `sensor_descent_ratio <= 0` means the combined local SGD direction
does not reduce sensor loss to first order.

Validate gradient additivity by differentiating the summed loss directly and
comparing it with the sum of stored objective gradients.

## Registered gates

### Initial cross-block starvation

A learned seed passes if, on both locked minibatches at the initial state:

- the encoder's separate block clip coefficient is `1.0`;
- the global clip coefficient is at most `0.10`;
- encoder cross-block suppression is at most `0.10`;
- the encoder sensor-gradient norm is nonzero.

The population gate requires at least four of five seeds separately for d6 and
d10.

### Persistent learned-state conflict

A learned seed passes if, on both locked minibatches at the final state:

- sensor-gradient norm is nonzero;
- `sensor_descent_ratio <= 0` in the encoder.

The population gate again requires at least four of five seeds separately for
d6 and d10. This gate is independent of the initial-starvation gate.

### Validity

All 20 cells must replay source/interface hashes, preserve source and interface
state digests, use the exact first/last schedule rows, produce finite arrays,
reload diagnostics exactly, and satisfy maximum gradient-additivity error
`<= 1e-5`.

## Locked classifications

| Initial gate | Final conflict gate | Classification |
| --- | --- | --- |
| pass | pass | `global_clip_starvation_and_persistent_conflict` |
| pass | fail | `initial_cross_block_clip_starvation_only` |
| fail | pass | `persistent_objective_conflict_without_initial_starvation` |
| fail | fail | `no_registered_gradient_failure_mechanism` |

If validity fails, classify `invalid`.

The first classification supports a separately normalized or structurally
fixed sensor update before any backbone unfreezing. The second says the first
updates were suppressed but does not establish a persistent explanation. The
third favors objective separation over a global-clip account. The fourth sends
the program to the already licensed full-interface fine-tune without changing
Stage A.

No post-outcome threshold, new minibatch, loss rescaling, optimizer comparison,
or retraining is permitted in this diagnostic.

## Artifact root

`data/experiments/tinyllm_joint_interface_gradient_attribution/20260811_d6_d10_registered`
