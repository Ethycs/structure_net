# TinyLLM group-anchored task-metric transport preregistration

**Status:** PREREGISTERED — PRIMARY OUTCOMES NOT INSPECTED  
**Date:** 2026-08-06  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, `UNDERPOWERED`  
**Hypothesis:** `tinyllm-c2-group-task-metric-carrier-transport-v1`  
**Schema:** `nal.tinyllm-group-task-metric-transport.v1`

## Question

The preceding frozen experiment found strong and specific cross-checkpoint
coordinate correspondence in every directed pair, but zero causal transports.
Does that discrepancy arise because Euclidean coordinate fitting weights the
wrong directions inside the multiplicity space of the invariant `C2` carrier?

This is the last low-cost frozen diagnostic before architecturally fixing a
shared equivariant sidecar. It does not retrain TinyLLM or add a predictive
observer.

## Group anchor

Every source and target state remains the exact block-0 post-attention Reynolds
defect on a two-sheet deck orbit. The charged sheet component transforms in the
nontrivial `C2` character and its neutral synthesis lies in

```text
c1 tensor c1 -> c0.
```

The rank-three defect basis is therefore a multiplicity space inside the
trivial representation, not an arbitrary activation subspace. A generic basis
change inside that space is group-compatible but need not preserve the metric
used by the frozen continuation. This experiment tests that remaining gauge.

## Fixed predecessor and data

- d6 checkpoints and frontends, seeds 7, 29, and 53;
- block-0 post-attention synthesis front;
- source-fitted rank-three defect basis in every checkpoint;
- cross-seed predecessor campaign SHA-256
  `44707fd4bcd810e63614671aa491095fae735ee52359d464ab25abb10a2bc228`;
- the same alignment-fit and two held-out cohorts, each with 64 exact orbits
  under composition and extrapolation;
- the same six directed checkpoint pairs and frozen scalar rotations;
- no model, frontend, probe, decoder, or calibration training.

All predecessor hashes and the original paired, shuffled, and affine-ridge maps
must be validated before analysis. Re-evaluated predecessor states must match
their stored continuous metrics within `1e-6`.

## Label-free pullback Fisher metric

For target rank-three coordinates `y_i`, patch the corresponding direct target
state and let `l_i(y)` be the 16 answer-token logits of the frozen continuation.
Estimate the coordinate Jacobian by centered finite differences:

```text
J_i[:, d] = (l_i(y_i + eps_d e_d) - l_i(y_i - eps_d e_d)) / (2 eps_d),
eps_d = 1e-3 * std(y[:, d]).
```

The minimum absolute step is `1e-4`. Repeat with half the step as a numerical
control. Let `p_i = softmax(l_i(y_i))`; the pullback Fisher metric is

```text
G_i = J_i^T (diag(p_i) - p_i p_i^T) J_i.
```

No phase, target bin, branch label, or nuisance label enters `G_i`. Normalize
each nondegenerate metric by its trace and add an isotropic floor
`0.01 I / 3`. The median relative difference between full- and half-step
metrics must be at most `0.05`; otherwise the pair fails the metric contract.

## Task-metric map

For augmented source coordinate `z_i = [x_i, 1]`, fit one affine map `W` by

```text
min_W sum_i (z_i W - y_i)^T G_i (z_i W - y_i)
      + 1e-6 ||W_linear||_F^2.
```

Fit only on the two alignment-fit regimes. The paired group/task-metric map is
evaluated without adaptation on the four held-out cells. A deterministic
regime-preserving shuffled-pair control permutes target coordinates and their
associated metrics together before fitting.

The original whitened-orthogonal, shuffled, and unconstrained affine-ridge maps
are evaluated in the same continuation batch as exact, zero, and direct-rank-3
states.

## Primary endpoints

For each directed pair, require all of the following:

1. **metric contract:** median full/half-step relative error at most `0.05`,
   finite symmetric positive-semidefinite metrics, and nonzero unregularized
   Fisher trace;
2. **continuous target controls:** zero fails while exact and direct rank three
   pass in all four held-out cells, with decomposition error at most `1e-6`;
3. **coordinate transport:** task-metric held-out variance explained is at least
   `0.80` in all four cells;
4. **causal transport:** task-metric alignment loss is at most `0.005`, mean
   circular-moment shift at most `0.125` bins, p95 shift at most `0.50` bins,
   winding within `0.10` of degree two, and sampling is resolved in all four
   cells;
5. **baseline dominance:** its four-cell mean moment shift is at most `75%` of
   the better aggregate value from the original paired and affine-ridge maps;
6. **shuffled specificity:** the task-metric shuffled map fails coordinate or
   continuous causal transport in at least one cell, and the paired aggregate
   mean shift is at least `0.125` bins lower than the shuffled aggregate.

Confirmation requires every gate in all six directed pairs. The frozen
scalar-calibrated exact-bin endpoint is retained as a secondary measurement
and cannot rescue or invalidate the continuous primary endpoint. This boundary
is declared because the predecessor already established that seed 7's scalar
rotation is unstable on two fresh composition cells even for exact states.

## Interpretation

| Outcome | Interpretation |
| --- | --- |
| all six pairs pass | the checkpoints share a symmetry-typed causal carrier after the correct label-free task metric fixes its multiplicity-space gauge |
| geometry improves but causal gates fail | task sensitivity is nonlinear or checkpoint-local beyond a first-order Fisher metric |
| only directions into one target pass | the carrier atlas is checkpoint-stratified rather than globally gauge-equivalent |
| shuffled control passes | marginal metric/coordinate distributions, not paired representation, explain the result |
| metric contract fails | finite-difference task geometry is numerically unresolved; do not interpret transport outcomes |

If the campaign does not confirm, stop post-hoc carrier alignment. The next
constructive step is an explicitly equivariant sidecar with declared irrep
channels, neutral tensor-product fusion, and a fixed readout metric trained as
part of the architecture.

## Planned artifacts

- runner:
  `experiments/structure_net/tinyllm_group_task_metric_transport.py`
- tests:
  `tests/structure_net/test_tinyllm_group_task_metric_transport.py`
- primary root:
  `data/experiments/tinyllm_group_task_metric_transport/20260806_d6_preregistered`
- report:
  `docs/08 - Analysis/2026-08-06_tinyllm-group-task-metric-transport.md`
- meta hypothesis:
  `tinyllm-c2-group-task-metric-carrier-transport-v1`

Any change to cohort seeds, ranks, metric definition, finite-difference steps,
map objective, gates, or thresholds after a quality outcome is visible requires
a new root and explicit post-outcome evidence role. Underpowered CUDA
shakedowns are systems-only evidence and cannot be pooled.
