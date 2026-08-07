# TinyLLM local continuation tangent preregistration

**Status:** PREREGISTERED POST-OUTCOME DIAGNOSTIC — TANGENT/KERNEL OUTCOMES NOT INSPECTED  
**Date:** 2026-08-07  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, `UNDERPOWERED`, post-outcome diagnostic  
**Hypothesis:** `tinyllm-c2-local-continuation-tangent-v1`  
**Schema:** `nal.tinyllm-c2-local-continuation-tangent.v1`

## Evidence boundary and question

The frozen writer-capacity campaign is already known to classify seeds `7`,
`29`, and `53` as `small_writer_insufficient`. Exact quotient phase plus the
top three propagated-barycenter coordinates predicts the rank-three defect
with high ordinary coordinate fidelity, but no tested writer closes all four
held-out causal cells.

This study asks the shortest remaining causal question:

> Is the residual error of a fixed small writer concentrated in the local
> task-sensitive direction of the frozen continuation, or does its nominal
> first-order kernel remain causally relevant?

The tangent/kernel interventions and their outcomes have not been inspected.
This is a reused-cell, three-checkpoint mechanistic diagnostic. It cannot
independently replicate the writer failure or establish population
prevalence.

## Locked source and arm

The sole predecessor is:

```text
data/experiments/tinyllm_frozen_writer_capacity/
    20260807_d6_preregistered_diagnostic/campaign_results.json
SHA-256 7ab6079d56bc8dc78cc5e1f4011c581b8f507bcd8ec20573bbf8b11227df8d1b
implementation d53edaedd49ae553af9f8393d92254664239e5100246ac0fd3a06cb420ca80ed
```

Reuse exactly:

- the d6 degree-two checkpoints for seeds `7`, `29`, and `53`;
- the source-fitted rank-three block-0 post-attention carrier bases;
- the two alignment-fit regimes and four held-out cells;
- 64 exact `C2` orbits per cell and the existing evaluation seeds;
- the calibrated answer-token readout and continuous endpoint; and
- the stored `context_m04` mapping.

`context_m04` is fixed because it is the highest-order member of the declared
low-order, state-conditioned ladder. It is not selected separately by seed or
held-out outcome. The prior context PCA is deterministically reconstructed
from its original alignment-fit cells and must reproduce its stored model
digest. The stored writer weights are loaded, not refit. No TinyLLM, probe,
decoder, writer, coordinate map, or predictive observer is trained or fitted.

## Standardized local chart

Let `z_pred` be the stored `context_m04` prediction and `z_star` the exact
rank-three defect coordinate for one held-out orbit. Let `s` be the
coordinate-wise population standard deviation of the exact alignment-fit
coordinates, computed across the two original fit regimes. Define

```text
e = (z_star - z_pred) / s.
```

All projections below use this fixed standardized chart. A coordinate scale
below `1e-8`, a context replay mismatch, or predecessor endpoint replay above
`1e-6` invalidates that checkpoint.

## Frozen local derivative

At each predicted state, estimate the signed circular-moment angle derivative
in output-bin units with respect to the three standardized carrier
coordinates. Centered finite differences use the already validated carrier
scales:

```text
fine step   = 0.025 standard deviations
coarse step = 0.050 standard deviations
g_i ~= wrap(theta(z_pred + h s_i e_i)
             - theta(z_pred - h s_i e_i)) / (2 h bin_width).
```

The local task-tangent and first-order kernel components are

```text
e_tan = g (g dot e) / (g dot g)
e_ker = e - e_tan.
```

They must reconstruct `e` to relative error at most `1e-8`, satisfy
`|g dot e_ker| / (|g| |e|) <= 1e-8`, and have nondegenerate gradient norm.
The fine derivative defines the intervention; the coarse derivative is only a
convergence control.

## Causal interventions

Every held-out cell runs the same frozen continuation from block-0
post-attention:

| State | Coordinate write | Role |
| --- | --- | --- |
| `zero` | no Reynolds defect | negative target control |
| `exact` | full exact Reynolds defect | full-state reference |
| `direct_rank3` | `z_star` | positive carrier control |
| `predicted` | `z_pred` | fixed failed writer arm |
| `tangent` | `z_pred + s e_tan` | local task-gradient correction |
| `kernel` | `z_pred + s e_ker` | nominal first-order-null correction |
| `full_residual` | `z_pred + s(e_tan + e_ker)` | reconstruction control |
| `random_tangent` | random direction with norm `|e_tan|` | norm control |
| `random_kernel` | independent random direction with norm `|e_ker|` | norm control |

Random directions are deterministically generated per seed, cohort, regime,
and orbit in the standardized chart. They use no target, phase, or output
information beyond the norm they match.

## Primary endpoints

The unchanged continuous endpoint passes a state only when all of the
following hold relative to the full exact state:

- circular alignment loss at most `0.005`;
- mean moment shift at most `0.125` output bins;
- p95 moment shift at most `0.50` bins;
- winding degree within `0.10` of degree two; and
- resolved sampling.

The direct `zero`, `exact`, `direct_rank3`, and `predicted` records must replay
the predecessor within `1e-6`. `full_residual` must reproduce `direct_rank3`
within that tolerance.

Pool all 256 held-out orbits per checkpoint for the derivative audit. The
local linearization is adequate only when:

1. coarse/fine derivative cosine is at least `0.98`;
2. coarse/fine relative L2 difference is at most `0.15`;
3. zero-referenced signed-error `R2` is at least `0.50`;
4. prediction residual MAE is at most `0.50` of observed error MAE; and
5. sign agreement is at least `0.75` where observed error is at least `0.01`
   bins.

The **local task-tangent sufficiency** gate passes a checkpoint only if:

- every numerical, replay, and target-control contract passes;
- the pooled local linearization is adequate;
- `tangent` passes the continuous endpoint in all four held-out cells;
- `random_tangent` fails at least one held-out cell; and
- `tangent` improves aggregate mean moment shift over `random_tangent` by at
  least `0.125` bins.

Because the campaign contains only three previously selected checkpoints, the
campaign records common mechanistic support only if all `3/3` pass. It remains
formally underpowered regardless of outcome.

## Secondary locked classifications

When tangent sufficiency fails but all validity controls pass, define the
kernel-change fraction in each cell as

```text
mean |wrap(theta_kernel - theta_predicted)|
-------------------------------------------------
mean |wrap(theta_direct_rank3 - theta_predicted)|.
```

Pool the numerator and denominator across cells. A fraction of at least `0.10`
is `material_kernel_effect`. Classify each checkpoint by the first applicable
rule:

1. `invalid` if any replay, numerical, target, or full-residual control fails;
2. `local_task_tangent_sufficient` if the primary checkpoint gate passes;
3. `local_linearization_inadequate` if the derivative audit fails;
4. `nominal_kernel_causally_active` if the kernel-change fraction is at least
   `0.10`;
5. `tangent_kernel_interaction_or_endpoint_curvature` otherwise.

Kernel materiality is mechanistic localization and cannot rescue the failed
primary tangent gate.

## Outcome interpretation

| Outcome | Interpretation | Next shortest action |
| --- | --- | --- |
| tangent sufficient in 3/3 | the writer used the wrong local task metric | implement a frozen task-metric correction before any learned sidecar |
| linear model adequate, kernel active | a nominal first-order-null direction affects the nonlinear continuation | estimate the directional Hessian only along the measured kernel |
| linear model adequate, neither component alone closes | tangent/kernel interaction or endpoint curvature matters | test the single mixed Hessian term; do not fit a larger writer |
| local model inadequate | the writer state leaves the validated local regime | run a radius titration around `z_pred` |
| invalid controls | the predecessor or numerical chart was not replayed | stop without scientific interpretation |

## Controls and split contract

| Field | Locked value |
| --- | --- |
| intervention | tangent, kernel, their sum, and deterministic norm-matched random directions |
| fixed controls | checkpoints, architecture, rank-three basis, datasets, orbit count, readout, continuation, thresholds |
| stochastic controls | no training randomness; random controls use declared deterministic seed streams |
| evaluation families | heldout-A/B composition and extrapolation |
| exclusions/retries | no scientific exclusion or threshold retry; infrastructure failure may resume only fingerprint-matched incomplete work |

The alignment-fit cells provide only the already used context chart and
coordinate scales. Every primary causal endpoint is evaluated on the reused
held-out cells. No measurement from a held-out cell changes the writer,
gradient definition, step size, projection, or threshold.

## Artifacts and execution plan

- runner:
  `experiments/structure_net/tinyllm_local_continuation_tangent.py`
- tests:
  `tests/structure_net/test_tinyllm_local_continuation_tangent.py`
- primary root:
  `data/experiments/tinyllm_local_continuation_tangent/20260807_d6_preregistered_diagnostic`
- report:
  `docs/08 - Analysis/2026-08-07_tinyllm-local-continuation-tangent.md`
- meta hypothesis:
  `tinyllm-c2-local-continuation-tangent-v1`

The implementation must pass focused CPU contracts, a systems-only CUDA
lifecycle, immutable aggregate resume, strict JSON, implementation hashing,
and scientific fingerprints before the primary run is interpreted.

## Method boundaries

The circular-moment angle and frozen answer-token decoder define this task
metric; it is not an intrinsic representation metric. The rank-three basis,
context chart, writer arm, and evaluation cells were selected or inspected in
earlier post-outcome work. Finite differences and residual writes are
off-manifold interventions. A first-order kernel is local and
decoder-conditioned, not globally task-null. Three selected checkpoints make
this an underpowered mechanistic decomposition, not a population claim.
