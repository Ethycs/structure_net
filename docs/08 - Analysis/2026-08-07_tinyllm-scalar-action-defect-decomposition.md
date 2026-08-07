# TinyLLM scalar action-defect decomposition

**Status:** CONFIRMED IN TESTED SCOPE — RESIDUAL-COORDINATE DEFECT, `3/3`  
**Date:** 2026-08-07  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, `UNDERPOWERED`, stored-array mechanistic diagnostic  
**Hypothesis:** `tinyllm-c2-scalar-action-defect-decomposition-v1`  
**Schema:** `nal.tinyllm-c2-scalar-action-defect-decomposition.v1`  
**Preregistration:** [scalar action-defect decomposition preregistration](../07%20-%20Status%20Reports/2026-08-07_tinyllm-scalar-action-defect-decomposition-preregistration.md)

## Verdict

The nuisance-dependent signed correction is a **residual-coordinate defect**,
not a task-covector transport defect, in all three tested checkpoints. Seeds
`7`, `29`, and `53` each pass the complete preregistered gate across all four
actions and both shifts.

The joint first-order decomposition and the residual-coordinate component alone
pass all `24/24` action cells. The covector-change component passes `0/24`.
Semantic-phase-shift and sign-flip controls fail every cell with the required
margin.

Across all cells:

- joint first-order R2 is `0.99817--0.99968`;
- residual-coordinate-only R2 is `0.99842--0.99969`;
- residual-coordinate sign agreement is `100%`;
- covector-only R2 is `-0.0238--0.0088`; and
- the best negative-control R2 is `-0.841`.

This resolves the output-type question left by the preceding negative result.
The correction amplitude cannot be invariant, but the already portable
phase-conditioned task covector does not need to become action-dependent. The
prospective interface may remain a scalar amplitude **provided that amplitude
retains action/calibration context**.

## Decomposition

For standardized rank-three residual `z`, local task covector `J`, reference
input `x`, and transformed input `gx`, the symmetric two-factor decomposition
is

```text
D_residual = 0.5 (J_x + J_gx) (z_gx - z_x)
D_covector = 0.5 (J_gx - J_x) (z_x + z_gx)
D_joint    = D_residual + D_covector
           = J_gx z_gx - J_x z_x.
```

The decomposition is algebraically exact for the two stored local linear
models. Its agreement with the nonlinear observed scalar action defect was the
empirical test. Maximum numerical identity error was `1.67e-16`.

The authoritative campaign consumes the corrected nuisance-scalar gauge-replay
root. Its source basis-gauge contract passes `3/3`, including maximum all-cell
coordinate replay error `3.14e-6`. The initial decomposition root consumed the
superseded pre-gauge source and is retained for audit only; rerunning from the
corrected source changes no gate, classification, or reported metric at the
precision shown here.

The observed action-defect RMS ranges from `0.194` to `0.303` bins, comfortably
above the locked `0.02` nondegeneracy floor. The result is therefore not caused
by dividing performance metrics by a nearly zero target.

## Primary measurements

Each row pools three checkpoints. Values are mean
`(joint R2, residual-only R2, covector-only R2)`.

| Shift | Action | Mean component result |
| --- | --- | --- |
| composition | amplitude | `(0.99949, 0.99948, 0.00211)` |
| composition | orientation | `(0.99948, 0.99955, 0.00002)` |
| composition | offset | `(0.99942, 0.99949, -0.00128)` |
| composition | composed | `(0.99940, 0.99953, -0.00088)` |
| extrapolation | amplitude | `(0.99924, 0.99951, -0.01167)` |
| extrapolation | orientation | `(0.99903, 0.99929, -0.00255)` |
| extrapolation | offset | `(0.99927, 0.99934, 0.00478)` |
| extrapolation | composed | `(0.99861, 0.99878, 0.00128)` |

Residual-only relative L2 is `0.0176--0.0397`, far below the `0.316` ceiling.
Covector-only relative L2 is `0.996--1.012`, indistinguishable from predicting
zero at this scale. The residual and covector components also have no stable
alignment: their cosine varies from approximately `-0.384` to `0.147`.

## Mechanistic conclusion

The action dependence found by the transformation-law experiment comes almost
entirely from the rank-three coordinate residual changing under the observed
similarity action. Changes in the local task covector contribute negligibly to
the signed output correction.

The refined component split is

```text
portable phase-conditioned task covector
  x action-conditioned residual amplitude
  -> exact local task correction.
```

This is consistent with both earlier results:

1. A frozen source covector transports accurately when supplied the exact
   example-local scalar.
2. An invariant scalar is structurally incompatible because the writer-coordinate
   residual itself changes under target-preserving actions.

The negative scalar-sensor result is therefore not evidence that a scalar
interface is intrinsically too small. It is evidence that source-only invariant
or weakly observed scalar summaries omitted the action-conditioned residual.

## Architectural decision

Do not train an invariant amplitude head. The subsequent no-fit
[scalar groupoid-defect decomposition](2026-08-07_tinyllm-scalar-groupoid-defect.md)
showed that the coordinate residual cannot be reduced to the frozen writer's
observable action defect: a material direct-state term is required in all 24
cells, while writer-only prediction passes none.

This supersedes the prospective scalar-head comparison proposed from this
first-order result. The frozen sidecar branch is closed unless a separately
justified observable direct-state estimator is introduced. Prefer the already
validated calibrated invariant front end, and test calibration degradation or
a learned pilot estimator against its analytic positive control if further
constructive work is required.

This result does not motivate link cobordism: the decisive structure is an
ordinary first-order transformation law with no required singular locus.

## Campaign integrity

| Item | Value |
| --- | --- |
| checkpoints requested/completed | `3/3` (`7`, `29`, `53`) |
| source action cells | `24` |
| derived component fields | `72` |
| trained models / fitted modules | `0 / 0` |
| compute | CPU stored arrays only; no CUDA allocation |
| analysis time | `0.133` seconds |
| implementation SHA-256 | `dae6ccef726a8b0909067399b2a6228896c1256a68d763a17a318f08772616b0` |
| campaign SHA-256 | `bcd826ebe195c55a9b42aaaae68f7bcfc6923be3114c2cff7119b25675007f82` |
| source campaign SHA-256 | `1e1795c931335c739a38550b85def7c4b442be39afcea76af8cd8491b1ff6589` |
| DVC root | `ea8a92f9da1dd9bfd88038d86717e306.dir` (`1,988` files; `39,818,920,754` bytes) |
| lakeFS snapshot | `c4641a09d51015a803058d3434e802a0562a39c1beb39bf1ecc2052a1eb773f3` |

This is a preregistered decomposition of already measured checkpoint evidence,
not a fresh checkpoint replication. The three checkpoints are replication
units. Exact resume verified the completed aggregate and preserved its bytes.

## Artifacts and reproduction

- campaign:
  `data/experiments/tinyllm_scalar_action_defect_decomposition/20260807_d6_stored_arrays_gauge_replay/campaign_results.json`
- per-checkpoint records and arrays:
  `data/experiments/tinyllm_scalar_action_defect_decomposition/20260807_d6_stored_arrays_gauge_replay/runs/seed_*/`
- superseded pre-gauge-source root (audit only):
  `data/experiments/tinyllm_scalar_action_defect_decomposition/20260807_d6_stored_arrays/`
- runner:
  `experiments/structure_net/tinyllm_scalar_action_defect_decomposition.py`
- tests:
  `tests/structure_net/test_tinyllm_scalar_action_defect_decomposition.py`

```bash
pixi run python -m \
  experiments.structure_net.tinyllm_scalar_action_defect_decomposition \
  --output \
  data/experiments/tinyllm_scalar_action_defect_decomposition/20260807_d6_stored_arrays_gauge_replay
```

The exact DVC directory object is present at
`lakefs://artifacts/c4641a09d51015a803058d3434e802a0562a39c1beb39bf1ecc2052a1eb773f3/structure-net/files/md5/ea/8a92f9da1dd9bfd88038d86717e306.dir`.
The lakeFS branch was clean after commit, and the meta-hypothesis readback
verified all three checkpoint records.
