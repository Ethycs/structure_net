# TinyLLM analytic observable-residual preregistration

**Status:** INVALIDATED BY SOURCE-ONLY LIFECYCLE — PRIMARY EXECUTION FORBIDDEN  
**Date:** 2026-08-07  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, `UNDERPOWERED`, no-training mechanistic diagnostic  
**Hypothesis:** `tinyllm-c2-analytic-observable-residual-v1`  
**Schema:** `nal.tinyllm-c2-analytic-observable-residual.v1`

> Final lifecycle amendment: the required one-checkpoint source-only run failed
> the locked observed-carrier validity contract before a primary campaign was
> launched. All provenance, input, basis-gauge, signed-scalar, covector,
> local-linearization, target-control, and oracle gates passed, but maximum
> paired-sheet carrier difference was `0.588`, against the registered `0.01`
> ceiling. The campaign is invalid and supplies no evidence about analytic
> residual quality. No three-checkpoint campaign may execute under this
> protocol.

## Decision question

The preceding studies established that:

1. a source-fitted phase-conditioned task covector transports to fresh
   checkpoints when supplied the exact signed scalar;
2. the scalar is not invariant under target-preserving similarity actions; and
3. its action dependence is carried by the writer-coordinate residual rather
   than by transport of the task covector.

Before training an action-conditioned scalar head, test the cheapest declared
positive control:

> Does the signed difference between an analytic observation-derived semantic
> angle and the frozen order-four writer's own output angle causally close the
> frozen continuation when written along the portable task covector?

This is deliberately stricter than asking whether the analytic estimate has
good task accuracy. The endpoint is equivalence to the exact frozen
continuation. If the exact frozen computation implements a checkpoint-local
semantic offset, an observation-perfect task estimate may still fail this
mechanistic interface.

## Locked sources

Use the corrected nuisance-scalar gauge-replay campaign:

```text
data/experiments/tinyllm_nuisance_scalar_transformation_law/
    20260807_d6_existing_group_gauge_replay/campaign_results.json
SHA-256 1e1795c931335c739a38550b85def7c4b442be39afcea76af8cd8491b1ff6589
implementation e92c3894b3e8820c0242edeba29e1f893f321e7f1dc20d75583c898ce7b1526f
```

and the frozen source-covector campaign:

```text
data/experiments/tinyllm_source_task_covector_portability/
    20260807_d6_preregistered_fresh_cohort/campaign_results.json
SHA-256 fe153abd0aa1749a862095e577e6e9cf8a0f7054bade2ef3c6833bf32d7f4ac5
implementation 6716b909d0c245059a1ed1310f20f4d9e56deb8c49a7d3a031972542fccb3046
```

The three d6 C2 checkpoints `7`, `29`, and `53` are replication units. Reuse
the exact group cohort seeds `430007/430008`, all four similarity actions, the
reference arm, both composition and extrapolation, and all 64 paired orbits.
No model, writer, covector, encoder, observer, or regression is fit.

## Observable construction

For each input, decode the sensor history and use only its observed calibration
packet—orientation, signed speed, amplitude, offsets, and drifts—to recover the
future planar phase. Apply the exact neutral C2 fusion

```text
(x, y) -> (x^2 - y^2, 2xy, x^2 + y^2)
```

and let `theta_obs` be the angle of its first two channels. This must replay the
known semantic angle only for an audit; latent phase and target labels may not
enter the candidate correction.

Let `theta_writer` be the first circular-moment angle obtained by continuing
the frozen order-four coordinate through the unmodified checkpoint. Define

```text
s_obs = wrap_bins(theta_obs - theta_writer).
```

Evaluate the locked source-fitted covector `g_obs` from the same observed
neutral carrier. The minimum-norm standardized coordinate correction is

```text
delta_z = g_obs * s_obs / (||g_obs||^2 + epsilon).
```

Patch `order4 + coordinate_scale * delta_z` into the same frozen continuation.
The construction may use the checkpoint's frozen writer and decoder because
those are the mechanism under test. It may not use the direct-rank-three
coordinate, exact continuation angle, target posterior, target bin, latent
phase, or a fresh derivative to form `s_obs` or `g_obs`.

## States and controls

Evaluate at every action cell:

| State | Scalar / direction | Role |
| --- | --- | --- |
| `zero`, `exact`, `direct_rank3`, `order4` | predecessor definitions | validity and baseline |
| `frozen_covector_oracle_error` | exact signed error / frozen covector | portable-direction positive control |
| `analytic_observable` | `s_obs` / frozen covector | primary positive control |
| `analytic_phase_shift` | semantic-phase-block shifted `s_obs` | semantic correspondence control |
| `analytic_nuisance_shift` | nuisance-replicate shifted `s_obs` at fixed phase | example correspondence control |
| `analytic_flipped` | `-s_obs` | sign control |

Control permutations are deterministic and fixed before outcome inspection.

## Validity contracts

A checkpoint is valid only when all of the following pass:

1. all source campaign, result, array, checkpoint, writer, covector, and
   implementation hashes replay;
2. regenerated group inputs and pair summaries equal the locked source record;
3. the rank-three carrier basis aligns to the stored group coordinate gauge
   with maximum all-cell replay error at most `1e-5`;
4. the observation-derived neutral carrier has circular alignment at least
   `0.99`, mean shift at most `0.125` bins, p95 shift at most `0.50` bins, and
   maximum paired-sheet difference at most `0.01` in every cell;
5. the frozen covector predicts local fine gradients across all ten cells with
   zero-referenced R2 at least `0.90`, relative L2 at most `0.15`, and mean row
   cosine at least `0.99`;
6. all fine/coarse local-linearization contracts pass the established
   thresholds;
7. `zero` fails and `exact` plus `direct_rank3` pass in all cells; and
8. `frozen_covector_oracle_error` passes in all cells.

A failed validity contract is not evidence against the analytic residual.

## Primary endpoint

Retain the existing continuous frozen-continuation endpoint:

- alignment loss from exact at most `0.005`;
- mean circular-moment shift at most `0.125` bins;
- p95 shift at most `0.50` bins;
- winding degree within `0.10` of degree two; and
- resolved sampling.

The analytic observable-residual gate passes a checkpoint only if:

1. every validity contract passes;
2. `analytic_observable` passes all ten cells;
3. each control fails at least one cell; and
4. its aggregate mean shift is at least `0.125` bins lower than every control.

Support requires `3/3` checkpoints. Cell counts are repeated measurements, not
independent replications.

Scalar R2 against the exact frozen correction is descriptive and reported, but
the causal state endpoint is primary. This prevents a local linear scalar fit
from replacing the actual frozen-continuation test.

## Classification and stop rule

Apply the first matching checkpoint classification:

| Outcome | Classification | Next action |
| --- | --- | --- |
| validity contract fails | `invalid` | repair only the digital or numerical contract under a new root |
| analytic state and specificity pass | `analytic_observable_residual_sufficient` | compare this analytic interface with a learned conditioned scalar and matched controls |
| analytic state passes but specificity fails | `analytic_observable_residual_nonspecific` | narrow correspondence; do not train yet |
| oracle passes but analytic state fails | `observable_semantic_target_not_frozen_equivalent` | stop frozen-writer sidecar optimization and use the calibrated invariant front end |
| frozen-covector oracle fails | `portable_covector_not_replicated` | audit group-action transport before any sidecar |

No topology scan, link-cobordism scan, richer writer, covector refit, probe
sweep, or TinyLLM training follows a negative result.

## Fixed artifacts

- runner: `experiments/structure_net/tinyllm_analytic_observable_residual.py`
- tests: `tests/structure_net/test_tinyllm_analytic_observable_residual.py`
- result root: `data/experiments/tinyllm_analytic_observable_residual/20260807_d6_existing_group`
- report: `docs/08 - Analysis/2026-08-07_tinyllm-analytic-observable-residual.md`
- meta hypothesis: `tinyllm-c2-analytic-observable-residual-v1`

The runner must preserve strict JSON, implementation and source hashes,
scientific fingerprints, per-result hashes, deterministic exact resume, and
the zero-training/zero-fitting evidence role.

## Method boundaries

The semantic estimate is analytic because the synthetic calibration reference
is exact. This is a positive-control mechanism, not a learned deployment
result. The portable covector and the tested checkpoints were selected after
prior outcomes. The diagnostic asks whether a frozen writer can be repaired,
not whether the calibrated invariant front end already solves the task. Three
selected checkpoints remain underpowered for population prevalence.

## Lifecycle disposition

The invalid systems artifact is retained at

```text
data/experiments/tinyllm_analytic_observable_residual/
    20260807_shakedown_cpu/campaign_results.json
SHA-256 74332bddd010ce4e82d8f9a086e89556bb806334bb613bef90c54ade10ce9e35
implementation f005553bc9113525d0fa247fa9739038a22c80cb154879be963ad5b02d108e71
```

The quarantined root is tracked by DVC directory object
`037165e6cec2ed8fe1373635950364dd.dir` (`2,022` files;
`39,819,881,860` bytes) in clean lakeFS snapshot
`9787cb9b8a375a79e97f3d254e7d83c5b83db18d4e88682dec334a4b40018563`.

The failure follows from the observation model rather than a digital replay
bug. The paired deck sheets share additive sensor noise *before* neutral
quadratic fusion. Cross terms therefore make their individual neutral carriers
different even though their noiseless semantic state is the same. The runner
uses an orbit average, but the preregistered contract correctly rejected the
claim that the single-sheet observed carrier itself was invariant.

Replacing the interface with an exact pairwise Reynolds average would remove
the failed contract only by requiring both orbit sheets at inference. That is
an orbit-level oracle, not the proposed deployable observable residual, and it
would duplicate the already validated calibrated invariant front end. No v2
repair or primary run is justified. The frozen-writer sidecar branch remains
closed by the scalar groupoid-defect stop rule.
