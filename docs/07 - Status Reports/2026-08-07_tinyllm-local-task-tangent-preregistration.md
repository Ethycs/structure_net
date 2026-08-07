# TinyLLM local task-tangent preregistration

**Status:** PREREGISTERED POST-OUTCOME DIAGNOSTIC — TANGENT OUTCOMES NOT INSPECTED  
**Date:** 2026-08-07  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, `UNDERPOWERED`, post-outcome diagnostic  
**Hypothesis:** `tinyllm-c2-local-task-tangent-v1`  
**Schema:** `nal.tinyllm-c2-local-task-tangent.v1`

## Known result and question

The fixed-gauge sequence has already established that:

- exact quotient coordinates do not rescue a global linear writer;
- Fourier phase through order 40 does not produce a complete causal write;
- the observed eight-field calibration packet does not rescue order 4; and
- a source-fitted rank-three chart of the propagated Reynolds barycenter also
  fails every checkpoint despite capturing `0.939--0.961` of source variance.

The actual example-specific rank-three coordinates still pass every target
cell. The unresolved question is therefore local and causal:

```text
Is the order-4 coordinate residual mostly harmless carrier error
except for its projection onto the frozen continuation's local task tangent?
```

No model, writer, probe, decoder, basis, or threshold is fit in this study.

## Locked sources and replication units

Reuse the three selected d6 degree-two checkpoints `7`, `29`, and `53`, their
source-selected rank-three block-0 attention defect bases, the same two
held-out cohorts under composition and extrapolation, and the exact order-4
writers from:

```text
data/experiments/tinyllm_fixed_gauge_writer_capacity/
    20260806_d6_preregistered_diagnostic/campaign_results.json
SHA-256 c5592ee53b96e2d064a992070be6c3e699b24d88dbb658283e8dc2f00c267078
```

The three frozen checkpoints are the replication units. The 12 held-out cells
and 768 exact orbits are repeated measurements, not independent models. This
is an underpowered mechanistic diagnostic and cannot establish population
prevalence.

## Residual and local derivative

For each example, let `c_4` be the stored order-4 predicted coordinate and
`c_*` the actual rank-three defect coordinate. Let `s` be the alignment-fit
standard deviation of each coordinate and define

```text
e = (c_* - c_4) / s.
```

At the order-4 predicted state, estimate the derivative of the signed circular
output angle in output-bin units with respect to standardized carrier
coordinates. Use centered finite differences at `0.025` standard deviations;
`0.05` is the convergence control:

```text
g_j ~= wrap(theta(c_4 + h s_j e_j) - theta(c_4 - h s_j e_j)) / (2 h bin_width).
```

The first-order task tangent and kernel residuals are

```text
e_T = ((e dot g) / (g dot g + epsilon)) g,
e_K = e - e_T.
```

Convert both back to raw carrier coordinates before patching. Record gradient
norms, tangent/kernel residual norms, reconstruction error, and tangent-kernel
orthogonality. The tangent definition is decoder-conditioned and local; it is
not an intrinsic representation decomposition.

## Frozen causal states

At each existing held-out cell, run the unchanged frozen continuation from:

| State | Coordinate write | Meaning |
| --- | --- | --- |
| `zero` | no rank-three defect | required negative control |
| `exact` | full actual Reynolds defect | required positive control |
| `direct_rank3` | `c_*` | carrier-sufficiency control |
| `order4` | `c_4` | locked failed predecessor |
| `tangent_only` | `c_4 + e_T s` | causal task-tangent correction |
| `kernel_only` | `c_4 + e_K s` | complement without task tangent |
| `tangent_flipped` | `c_4 - e_T s` | signed-direction control |
| `tangent_shuffled` | `c_4` plus a within-cell permutation of `e_T s` | correspondence control |
| `tangent_random` | `c_4` plus a deterministic random direction with the same per-example norm as `e_T` | direction control |

The permutation and random stream are fixed from checkpoint and evaluation
seed before evaluation. No held-out choice selects them.

## Local-linearization gates

Pool the four cells within each checkpoint. The local model is adequate only
when all five conditions hold:

1. fine/coarse gradient cosine is at least `0.98`;
2. fine/coarse relative L2 difference is at most `0.15`;
3. zero-referenced R2 of predicted versus actual direct-minus-order4 signed
   angle change is at least `0.50`;
4. prediction residual MAE is at most `0.50` of observed MAE; and
5. sign agreement is at least `0.75` where observed magnitude is at least
   `0.01` output bins.

These thresholds exactly reuse the validated carrier-Jacobian audit protocol.

## Primary causal endpoint

All states retain the predecessor's continuous endpoint:

- circular alignment loss from exact at most `0.005`;
- mean circular-moment shift at most `0.125` output bins;
- p95 shift at most `0.50` bins;
- winding degree within `0.10` of degree two; and
- resolved sampling.

A checkpoint passes the **local task-tangent gate** only if:

1. predecessor order-4 metrics replay within `1e-6`;
2. zero fails while exact and direct-rank-three pass in all four cells;
3. coordinate scaling, residual reconstruction, and local-linearization gates
   pass;
4. `tangent_only` passes all four cells;
5. `kernel_only`, `tangent_flipped`, `tangent_shuffled`, and
   `tangent_random` each fail at least one cell; and
6. tangent-only aggregate mean shift is at least `0.125` bins better than each
   of those four controls.

The campaign supports the hypothesis only if all three checkpoints pass the
joint gate. Secondary averages cannot rescue a failed checkpoint.

## Fixed classifications

Apply the first matching row per checkpoint:

| Outcome | Classification |
| --- | --- |
| provenance, replay, numerical, or target controls fail | `invalid` |
| local model and complete tangent causal gate pass | `local_task_tangent_sufficient` |
| tangent passes but kernel also passes or specificity fails | `nonunique_or_curved_correction` |
| local model passes and tangent improves aggregate mean shift by at least `0.125` bins but misses a cell | `tangent_helpful_not_sufficient` |
| local model passes but tangent does not meet the helpful threshold | `task_tangent_insufficient` |
| local model fails | `nonlinear_at_writer_residual_scale` |

The campaign conclusion reports whether all three checkpoints share a class;
otherwise it is `checkpoint_stratified_local_geometry`.

## Interpretation and next action

| Outcome | Interpretation | Next shortest action |
| --- | --- | --- |
| tangent sufficient in all three | the writer error is small in Euclidean norm but concentrated in a local decoder-sensitive direction | architecturally fix or learn the typed task metric, then test fresh cohorts |
| tangent helpful but incomplete | first-order metric is real but p95, degree, or support tails remain | add a fixed radius/second-order diagnostic, not another global writer |
| task tangent insufficient | coordinate error is not explained by the scalar local task direction | audit the two-dimensional raw-moment Jacobian or matched-context defect transport |
| local model inadequate | residual leaves the linear regime | run a residual-radius titration and HVP correction |
| stratified | no single local correction mechanism spans these checkpoints | retain checkpoint-local causal charts and stop portable sidecar claims |

## Artifacts and execution

- runner:
  `experiments/structure_net/tinyllm_local_task_tangent.py`
- tests:
  `tests/structure_net/test_tinyllm_local_task_tangent.py`
- primary root:
  `data/experiments/tinyllm_local_task_tangent/20260807_d6_preregistered_diagnostic`
- report:
  `docs/08 - Analysis/2026-08-07_tinyllm-local-task-tangent.md`
- meta hypothesis:
  `tinyllm-c2-local-task-tangent-v1`

The runner must record the producer digest, predecessor campaign and result
hashes, checkpoint identities, scientific fingerprints, strict JSON, and
immutable resume. A CUDA shakedown is systems-only evidence and cannot enter
the scientific aggregate.

## Method boundaries

The tangent is conditioned on the frozen answer-token decoder and circular
angle, so it is not an intrinsic information-geometric object. The residual,
basis, writer, and held-out cells are post-outcome selections from earlier
studies. Finite differences and off-manifold patches establish local frozen
causal behavior, not natural use. Three selected checkpoints remain
underpowered, and a passing result would not by itself provide a deployable
encoder or portable cross-seed gauge.
