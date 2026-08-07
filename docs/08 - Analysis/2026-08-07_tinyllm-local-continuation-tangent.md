# TinyLLM activation-conditioned local continuation tangent

**Status:** PARTIAL SUPPORT — TASK TANGENT CLOSES 3/3; PRIMARY SPECIFICITY PASSES 2/3  
**Date:** 2026-08-07  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, `UNDERPOWERED`, post-outcome diagnostic  
**Hypothesis:** `tinyllm-c2-local-continuation-tangent-v1`  
**Preregistration:** [local continuation tangent preregistration](../07%20-%20Status%20Reports/2026-08-07_tinyllm-local-continuation-tangent-preregistration.md)

## Verdict

Task-conditioned activations clarify why a writer with excellent ordinary
coordinate fidelity can still fail causally. At the failed `context_m04`
state, the frozen continuation's scalar task gradient explains
`0.9641--0.9954` of the signed circular output error. Patching only the
writer residual projected onto that local task gradient passes all four
held-out composition/extrapolation cells in all three checkpoints. The
nominal first-order kernel changes the output by only `0.13--0.93%` of the
original writer gap.

The full preregistered hypothesis is nevertheless **not confirmed**. The
campaign required tangent sufficiency plus a `0.125`-bin advantage over a
norm-matched random correction in all `3/3` checkpoints. Seeds 7 and 53 pass;
seed 29's tangent correction still passes every cell but its random-control
advantage is `0.1196` bins, missing the fixed specificity margin by `0.0054`
bins. The locked conclusion is therefore
`checkpoint_stratified_local_continuation_geometry`.

The supported narrower finding is causal, not probe-only:

> Across these three selected checkpoints, almost all task-relevant error of
> the activation-conditioned writer lies in one decoder-conditioned local
> carrier direction, even though ordinary coordinate error remains
> three-dimensional.

## Preregistered gates

| Gate | Result | Required | Verdict |
| --- | ---: | ---: | --- |
| predecessor replay | **3/3** | 3/3 | pass |
| context and coordinate-scale numerics | **3/3** | 3/3 | pass |
| zero/exact/direct-rank-three controls | **3/3** | 3/3 | pass |
| coarse/fine local linearization | **3/3** | 3/3 | pass |
| tangent passes all four held-out cells | **3/3** | 3/3 | pass |
| tangent has declared random-direction specificity | **2/3** | 3/3 | fail |
| full local task-tangent checkpoint gate | **2/3** | 3/3 | fail |

No model, decoder, writer, probe, basis, or predictive observer was fitted.
All direct carrier, target, replay, and strict-JSON contracts passed.

## Results

| seed | signed-error R2 | residual MAE / observed | tangent mean shift | random mean shift | tangent advantage | kernel-change fraction | classification |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 7 | 0.9954 | 0.0282 | 0.0221 | 0.1511 | 0.1291 | 0.0045 | `local_task_tangent_sufficient` |
| 29 | 0.9641 | 0.0514 | 0.0502 | 0.1698 | 0.1196 | 0.0093 | `tangent_kernel_interaction_or_endpoint_curvature` |
| 53 | 0.9851 | 0.0374 | 0.0193 | 0.1503 | 0.1310 | 0.0013 | `local_task_tangent_sufficient` |

All shifts are in output-bin units. The seed-29 classification is a locked
first-match label, not positive evidence for curvature: its local model is
adequate, its tangent passes all cells, and its kernel-change fraction is
below the `0.10` materiality threshold. It fails only the declared random
specificity margin.

The numerical local model is substantially inside every registered bound:

- coarse/fine derivative cosine is `0.9999993--0.9999999`;
- coarse/fine relative L2 difference is `0.00034--0.00115`;
- signed-error sign agreement is `1.0` in every checkpoint; and
- linearization residual MAE is `2.82--5.14%` of observed error MAE.

This resolves the predecessor's apparent contradiction. The
activation-conditioned writer retained worst-cell coordinate `R2` near
`0.99`, yet failed the frozen continuation because ordinary coordinate
variance weights causally weak and causally sharp directions alike. The
decoder-conditioned task gradient isolates the sharp direction.

## Intervention

For standardized exact-minus-writer residual `e` and the frozen circular-angle
gradient `g` at the predicted carrier state, the runner computes

```text
e_task   = g (g dot e) / (g dot g)
e_kernel = e - e_task.
```

Centered finite differences use `0.025` coordinate standard deviations; a
`0.050` step is the convergence control. The frozen continuation is then run
from block-0 post-attention with the predicted write, task correction, kernel
correction, their exact sum, and deterministic norm-matched random controls.
The unchanged continuous endpoint measures circular alignment, mean/p95
moment shift, sampling resolution, and winding degree under two held-out
cohorts and both composition and extrapolation.

## Relationship to the full-moment attempt

A separate preregistered `2 x 3` circular-moment Jacobian campaign is retained
under
`data/experiments/tinyllm_local_continuation_tangent_kernel/20260807_d6_preregistered_diagnostic`.
It is **invalid scientific evidence**: all three checkpoints exceeded its
locked `0.05` central-finite-difference relative-error ceiling
(`0.0610--0.0936`). Its downstream patches must not be used to rescue or
strengthen this result. The scalar-angle study reported here was independently
preregistered before those outcomes and passes its full convergence and
linearization contract.

That study also has a separately versioned, explicitly post-outcome corrective
replication at `.../20260807_d6_corrective_v2`. Increasing only the numerical
step makes its derivative contract pass and yields a rank-two tangent that
closes every cell, an inert rank-one kernel, and cross-cell leading-direction
instability. It is useful triangulation, but its post-outcome status cannot
convert either original preregistered full gate into confirmation.

## Mechanistic update

The local task geometry is much more informative than activation variance:

1. a rank-three activation chart is descriptively accurate but not by itself
   a causal metric;
2. the frozen continuation supplies a locally almost one-dimensional task
   covector;
3. correcting the writer along that covector closes every tested causal cell;
4. the nominal scalar-gradient kernel is nearly task-inert at the measured
   residual scale; and
5. the exact `3/3` uniqueness claim remains too strong under the fixed random
   specificity margin.

The shortest next experiment is not another representation scan or larger
writer. Freeze this task covector on source/alignment data, test its stability
on a genuinely fresh cohort, and patch a source-predicted **scalar task error**
rather than three Euclidean carrier coordinates. Failure would reject a
portable task metric before any learned sidecar is trained.

## Campaign integrity

| Item | Value |
| --- | --- |
| requested / completed / failed | 3 / 3 / 0 |
| trained models / fitted observers / fitted writers | 0 / 0 / 0 |
| checkpoints | d6 seeds 7, 29, 53 |
| held-out cells | two cohorts under composition and extrapolation |
| exact orbits per cell | 64 |
| finite-difference state families | 144 |
| device | NVIDIA GeForce RTX 2060 SUPER, CUDA |
| peak allocated CUDA memory | 281,465,856 bytes |
| analysis time | 12.62 seconds |
| implementation SHA-256 | `9cf3d7cc51f19397792da9be2cbe163d573931d3452d31d44bfcaa5583cc4a8e` |
| campaign SHA-256 | `de69c302c6418d0b6e0cf7b8254c73fcc4b97fed57472a318e75fe9584ea9e85` |
| final DVC data root | `7053c5bcd5433ee6822ec9825b782b53.dir` (`1,847` files, `39,814,283,869` bytes) |
| lakeFS snapshot | `fd8392ef275fda3a4e98fbe957208151061cfb8c0aeb48451b6455ecc326ed55` |

The separate single-checkpoint CUDA lifecycle is explicitly
`systems_lifecycle_only_not_quality_evidence`; it is not pooled. Re-running
the completed primary root verified immutable aggregate reuse without
rewriting result bytes.

The final DVC root was pushed to the configured
`lakefs://artifacts/main/structure-net/` remote and is contained in the cited
clean lakeFS commit.

## Artifacts and reproduction

- Aggregate:
  `data/experiments/tinyllm_local_continuation_tangent/20260807_d6_preregistered_diagnostic/campaign_results.json`
- Per-checkpoint records:
  `data/experiments/tinyllm_local_continuation_tangent/20260807_d6_preregistered_diagnostic/runs/seed_*/result.json`
- Systems-only lifecycle:
  `data/experiments/tinyllm_local_continuation_tangent/20260807_shakedown_cuda/`
- Runner:
  `experiments/structure_net/tinyllm_local_continuation_tangent.py`
- Tests:
  `tests/structure_net/test_tinyllm_local_continuation_tangent.py`
- Meta-hypothesis:
  `data/meta_hypotheses/tinyllm-c2-local-continuation-tangent-v1.json`

The named hypothesis and all three direct experiment records passed
authoritative Chroma readback. Legacy NumPy-2.0 consumer and telemetry warnings
were non-fatal; the readback gate and strict JSON ledger are authoritative.

```bash
MPLCONFIGDIR=/tmp/matplotlib-structure-net \
pixi run python -m \
  experiments.structure_net.tinyllm_local_continuation_tangent \
  --device cuda:0 \
  --output \
  data/experiments/tinyllm_local_continuation_tangent/20260807_d6_preregistered_diagnostic
```

## Method boundaries

The circular-angle tangent is decoder-conditioned and checkpoint-local, not an
intrinsic representation metric. Exact residual coordinates and quotient
phase are diagnostic latent quantities. The carrier basis, context writer,
and held-out cells were selected or inspected in predecessor studies.
Finite-difference and residual patches are off-manifold causal interventions.
The three checkpoints are selected and underpowered; neither the `3/3`
tangent closure nor the failed uniqueness gate establishes population
prevalence or a deployable correction.
