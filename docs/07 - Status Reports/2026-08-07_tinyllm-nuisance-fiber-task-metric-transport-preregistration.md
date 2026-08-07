# TinyLLM nuisance-fiber task-metric transport preregistration

**Status:** PREREGISTERED — PRIMARY OUTCOMES NOT INSPECTED  
**Date:** 2026-08-07  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, `UNDERPOWERED`  
**Hypothesis:** `tinyllm-c2-nuisance-fiber-task-metric-transport-v1`  
**Schema:** `nal.tinyllm-c2-nuisance-fiber-task-metric-transport.v1`

## Question

The frozen local-continuation audits found a rank-two task tangent that closes
the continuous endpoint in all 12 checkpoint/cohort/shift cells, but no stable
global Euclidean metric direction. Is that local tangent nevertheless a
well-defined quotient object: does it transport between observations with the
same semantic phase and different nuisance realizations inside one frozen
checkpoint?

This is a no-training, no-fit causal symmetry audit. It is the shortest test
before building an explicitly typed equivariant sidecar.

## Representation contract

The measured rank-three Reynolds-defect carrier is in the neutral `C2`
multiplicity space. The deck action on these coordinates is therefore the
trivial representation, not another learned rotation:

```text
rho(g) = I3,
J(gx) = J(x),
P(gx) = P(x),
```

where `J` is the two-output continuation-moment Jacobian and

```text
P = J+ J
```

is its rank-two row-space projector. Projectors and kernel lines are compared
instead of signed eigenvectors, avoiding sign gauge.

The analytic calibration symmetry used as a contract is

```text
Gcal = (R>0 x SO(2)) semidirect (R2_offset x R2_drift).
```

It acts jointly on the observed planar sensor and its calibration packet by
rotation, positive scaling, added offset, and added linear drift. The analytic
phase carrier is exactly invariant, so `rho(g)=I3`. This contract is distinct
from the primary nuisance-fiber test.

The full N3 generator also changes speed, harmonic content, and observation
noise. Those transformations are treated honestly as an observation groupoid,
not claimed to form one globally invertible finite-dimensional group. Two
observations are paired only when their quotient phase is exactly shared.

## Frozen sources and fresh cohorts

- d6 checkpoints 7, 29, and 53;
- the frozen source-fitted rank-three carrier basis in each checkpoint;
- the frozen order-four phase-only writer from campaign
  `7ab6079d56bc8dc78cc5e1f4011c581b8f507bcd8ec20573bbf8b11227df8d1b`;
- the corrected local full-moment Jacobian method from campaign
  `8b2f25dc61d1ffb93ba9dc66f1a85f36c07bda5a43ed3ed34a9627ca1485c12a`;
- 64 exact `C2` orbits per cell;
- two fresh cohorts under composition and extrapolation.

Fresh seeds are fixed before execution:

| Cohort | Regime | quotient phase | nuisance source | nuisance target |
| --- | --- | ---: | ---: | ---: |
| fresh 1 | composition | 910101 | 910111 | 910121 |
| fresh 1 | extrapolation | 910102 | 910112 | 910122 |
| fresh 2 | composition | 920101 | 920111 | 920121 |
| fresh 2 | extrapolation | 920102 | 920112 | 920122 |

The quotient phases are generated once per cell and supplied unchanged to the
source and target nuisance generators. None of these seeds occurs in the
writer or tangent predecessors.

## Intervention

For each matched source/target orbit, evaluate the Jacobians at the identical
order-four predicted coordinate `yhat`, using each observation's own propagated
block-0 state:

```text
Js = d moment(continue(xs, y)) / dy at yhat,
Jt = d moment(continue(xt, y)) / dy at yhat.
```

Let the target residual be `e = y_exact,t - yhat`. Compare:

```text
local tangent       = Pt e,
transported tangent = Ps e,
local kernel        = (I - Pt) e.
```

Patch every state into the same frozen target continuation. Thus a transported
pass cannot be explained by evaluating source and target with different
downstream models.

Eight deterministic per-orbit random rank-two projectors and one
phase-shuffled source-projector control are norm- and rank-matched. No phase,
target bin, nuisance label, or outcome is used to choose a projector.

## Locked gates

Every checkpoint must first satisfy:

1. **exact calibration-action contract:** for the fixed nonidentity action
   `(rotation=0.71, scale=1.6, offset=(0.31,-0.17),
   drift=(-0.05,0.08))`, the analytic carrier, propagated state, predicted
   coordinate, Jacobian, and projector agree with the untransformed target to
   maximum absolute error `1e-4`;
2. **numerical and target controls:** every Jacobian is finite and rank two,
   zero/predicted fails, full and local tangent pass, local kernel is inert,
   and tangent/kernel decomposition error is at most `1e-6` in all four fresh
   cells;
3. **geometric transport:** per matched cell, median absolute cosine between
   source and target kernel lines is at least `0.90` and its 10th percentile is
   at least `0.75`;
4. **causal transport:** the transported tangent passes the frozen continuous
   endpoint in all four cells, and its four-cell aggregate mean moment shift is
   no more than `0.05` bins above the local tangent;
5. **specificity:** the phase-shuffled source projector fails at least one
   cell and is at least `0.125` bins worse in aggregate mean shift; at most one
   of eight random projectors may pass all four cells, and their median
   aggregate mean shift must be at least `0.125` bins worse than transported.

The continuous endpoint is unchanged: alignment loss at most `0.005`, mean
moment shift at most `0.125` bins, p95 at most `0.50` bins, winding within
`0.10` of degree two, and resolved sampling. Frozen scalar exact-bin accuracy
is secondary and cannot rescue or veto the primary endpoint.

The hypothesis is confirmed only if all five gates pass in all three selected
checkpoints. This three-checkpoint cohort is explicitly underpowered and does
not estimate population prevalence.

## Interpretation

| Outcome | Interpretation |
| --- | --- |
| exact action and nuisance transport pass | the local task tangent descends to the quotient and supplies the minimal typed sidecar law |
| exact action passes, nuisance transport fails | analytic invariance is correct, but the downstream task metric remains nuisance-contextual; retain an orbit-local atlas |
| geometry fails but causal transport passes | projector equality is too strong; only an intervention-equivalence class is stable |
| geometry passes but causal transport fails | first-order row space is insufficient; metric scale, curvature, or propagated context matters |
| exact action contract fails | implementation or numerical contract failure; do not interpret the primary result |
| random or shuffled controls pass | low codimension alone explains the apparent transport |

No failed outcome authorizes fitting a post-hoc group representation. The next
constructive branch after failure is an architectural sidecar with declared
character channels and neutral fusion; after success it is the same sidecar
with the empirically validated trivial action on the neutral carrier.

## Planned artifacts

- runner:
  `experiments/structure_net/tinyllm_nuisance_fiber_task_metric_transport.py`
- tests:
  `tests/structure_net/test_tinyllm_nuisance_fiber_task_metric_transport.py`
- primary root:
  `data/experiments/tinyllm_nuisance_fiber_task_metric_transport/20260807_d6_preregistered_diagnostic`
- report:
  `docs/08 - Analysis/2026-08-07_tinyllm-nuisance-fiber-task-metric-transport.md`
- meta hypothesis:
  `tinyllm-c2-nuisance-fiber-task-metric-transport-v1`

Any change to seeds, pair construction, projector definition, controls,
endpoints, or thresholds after a quality outcome is visible requires a new
root and an explicit post-outcome evidence role. CUDA shakedowns are
systems-only evidence and cannot be pooled.
