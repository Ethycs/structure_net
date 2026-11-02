# TinyLLM scalar action-defect decomposition preregistration

**Status:** COMPLETED — RESIDUAL-COORDINATE DEFECT CONFIRMED IN 3/3; SEE MEASURED REPORT  
**Date:** 2026-08-07  
**Profile:** NAL-STD-EXPERIMENT `PREREGISTERED`, `UNDERPOWERED`, no-fit mechanistic diagnostic  
**Hypothesis:** `tinyllm-c2-scalar-action-defect-decomposition-v1`  
**Schema:** `nal.tinyllm-c2-scalar-action-defect-decomposition.v1`

**Measured report:**
[`2026-08-07_tinyllm-scalar-action-defect-decomposition.md`](../08%20-%20Analysis/2026-08-07_tinyllm-scalar-action-defect-decomposition.md)

## Decision question

The completed nuisance-scalar transformation-law experiment established that
the exact correction multiplying the portable phase-conditioned task covector
is action-dependent in all three checkpoints. Before training any
action-conditioned sidecar, determine which stored local quantity carries that
dependence:

1. the rank-three writer-coordinate residual changes under the action;
2. the local task covector changes with the propagated activation context; or
3. both changes are required jointly.

The primary hypothesis is the shortest continuation of the earlier portable
covector result:

> The action defect is residual-coordinate dominated: changing the coordinate
> residual at a symmetrized local task covector is sufficient, while changing
> the covector alone is not.

If this prediction fails, the experiment still selects the required interface
type before prospective training.

## Locked evidence

Use only the completed transformation-law campaign:

```text
data/experiments/tinyllm_nuisance_scalar_transformation_law/
    20260807_d6_existing_group/campaign_results.json
SHA-256 e1e21cf08b736547d8e77de0f15b5ac34b0a8a92ccba8afa19fa8dfb8f22b633
implementation c7ee01f82a257e779b8fc8a656321d0daf9e3a05df418aa97b74598b04de008d
```

For seeds `7`, `29`, and `53`, reuse its exact signed-scalar and fine-gradient
arrays together with the locked rank-three residual arrays from the source
group campaign. Reuse composition and extrapolation, all four actions
`amplitude`, `orientation`, `offset`, and `composed`, and all `64` aligned
orbits per cell.

No continuation is rerun. No model, writer, encoder, observer, or regression is
fit. The source arrays and their producing records are hash-gated before any
metric is computed. No component metric from these arrays was inspected before
this document.

## Symmetric first-order decomposition

For reference `x` and transformed input `gx`, let

```text
z_x = coordinate_residual(x) / coordinate_scale
J_x = fine task-angle gradient at the order-four prediction
y_x = exact observed signed output correction
```

and define the wrapped observed action defect

```text
Delta_y = wrap_bins(y_gx - y_x).
```

The first-order scalar prediction is `hat_y_x = J_x z_x`. Decompose its action
change symmetrically:

```text
D_residual = 0.5 (J_x + J_gx) (z_gx - z_x)
D_covector = 0.5 (J_gx - J_x) (z_x + z_gx)
D_joint    = D_residual + D_covector
           = J_gx z_gx - J_x z_x.
```

This is the symmetric two-factor decomposition; it does not privilege an
ordering in which residual or covector changes first. `D_joint` is algebraically
exact for the two local linear models, but its agreement with nonlinear
`Delta_y` is empirical.

## Metrics and controls

For `D_joint`, `D_residual`, and `D_covector`, report prediction of `Delta_y`:

- zero-referenced R2;
- relative L2 and MAE;
- sign agreement above `0.01` bins;
- correlation and target RMS; and
- component RMS and residual/covector cosine.

The action defect must have RMS at least `0.02` bins to be nondegenerate.

Use two deterministic negative controls:

1. shift each prediction by one of the 16 semantic-phase blocks while retaining
   nuisance-replicate index; and
2. flip the joint prediction's sign.

A valid joint explanation must exceed both controls by at least `0.10` R2, and
each control must fail the primary prediction thresholds. The controls test
example-level alignment, not the algebraic decomposition identity.

## Gates and taxonomy

A candidate predicts an action cell only when all three hold:

```text
zero-referenced R2 >= 0.90
relative L2 <= sqrt(0.10)
sign agreement >= 0.90.
```

Apply the first matching cell label:

| Condition | Label |
| --- | --- |
| joint fails | `nonlinear_or_unresolved` |
| residual passes, covector fails | `residual_coordinate_defect` |
| covector passes, residual fails | `covector_transport_defect` |
| both pass | `dual_sufficient` |
| joint passes, neither component passes | `coupled_action_defect` |

A checkpoint receives a stable type only if all eight cells have the same
non-null label and joint specificity passes everywhere. Otherwise it is
`checkpoint_internal_mixture`. The primary hypothesis requires
`residual_coordinate_defect` in all eight cells of all `3/3` checkpoints.

The joint first-order mechanism is separately supported only if all eight cells
pass in all `3/3` checkpoints. It cannot promote the residual-coordinate
hypothesis if component type varies.

## Outcome-directed decisions

| Outcome | Consequence |
| --- | --- |
| residual-coordinate defect `3/3` | train an action-conditioned scalar amplitude channel with the portable covector fixed |
| covector-transport defect | a scalar amplitude is insufficient; expose or predict the local task covector |
| coupled defect | use a typed `(residual coordinates, local covector/context)` interface, with scalar-only as a negative control |
| checkpoint-internal or checkpoint-stratified | do not claim one shared sidecar type; retain an orbit-local atlas or replace the frozen writer |
| nonlinear/unresolved | stop local sidecar decomposition and use the successful calibrated invariant front end |

This experiment characterizes the frozen writer's error interface. It does not
test whether any selected prospective architecture can learn that interface.

## Fixed artifacts

- runner:
  `experiments/structure_net/tinyllm_scalar_action_defect_decomposition.py`
- tests:
  `tests/structure_net/test_tinyllm_scalar_action_defect_decomposition.py`
- result root:
  `data/experiments/tinyllm_scalar_action_defect_decomposition/20260807_d6_stored_arrays`
- report:
  `docs/08 - Analysis/2026-08-07_tinyllm-scalar-action-defect-decomposition.md`
- meta hypothesis:
  `tinyllm-c2-scalar-action-defect-decomposition-v1`

The runner must preserve strict JSON, source and producing-code hashes,
per-result and NPZ hashes, deterministic exact resume, and the zero-training,
zero-fitting evidence role.

## Post-run source amendment (not preregistered)

The locked source was superseded by its carrier-basis gauge-replay correction.
The first decomposition root remains immutable, and the authoritative campaign
was rerun from source campaign
`1e1795c931335c739a38550b85def7c4b442be39afcea76af8cd8491b1ff6589`
under
`data/experiments/tinyllm_scalar_action_defect_decomposition/20260807_d6_stored_arrays_gauge_replay`.
The corrected source's basis-gauge gate passes `3/3`. The decomposition still
classifies all three checkpoints as `residual_coordinate_defect`; all primary
gates and reported metrics are unchanged at the precision used in the report.
This is a provenance/coordinate-contract repair, not a change to the hypothesis,
metrics, thresholds, controls, cohort, or interpretation rule.
