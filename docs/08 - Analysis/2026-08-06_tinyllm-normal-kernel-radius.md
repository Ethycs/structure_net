# TinyLLM normal jet-kernel structural radius

**Status:** SUPPORTED IN THE DECLARED SINGLE-TRANSITION EDIT SPACE  
**Date:** 2026-08-06  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, REFINED PHASE GRID  
**Hypothesis:** `tinyllm-normal-jet-kernel-structural-radius-v1`  
**Preregistration:** `../07 - Status Reports/2026-08-06_tinyllm-normal-kernel-radius-preregistration.md`

## Verdict

The reduced normal kernel predicted the local distance to the d6 step-15 degree
discriminant in the declared 384-dimensional block-1 MLP output-bias edit space.
Its predicted radius was `0.22758`; direct nonlinear feasibility found `0.25450`,
for a ratio of `0.8942`. The predicted direction reached residual `3.81e-7`,
ranked in the best 10% of itself plus 32 isotropic random directions, and crossed
a resolved degree-zero to degree-one bracket. All preregistered local gates pass.

This confirms only the stated single-transition, single-seed, parameter-subspace,
Euclidean-metric claim. It is not a global distance theorem or population result.

## Campaign integrity

The study used the deterministic d6 seed-7 step-14 state and the same continuous
input lift as the defect analysis. It formed the moment Jacobian with respect to
the 384 block-1 MLP `c_proj.bias` coordinates, projected out the phase tangent,
and compared the minimum-norm predicted direction with a nonlinear Gauss–Newton
solve and 32 budget-matched directional searches.

The initial 1,024-phase bracket was explicitly unresolved: maximum adjacent angle
was `2.19 > π/2`. Per the preregistered held-out refinement rule, an append-only
4,096-point run was produced. Its maximum adjacent angles were `0.885` before and
`0.837` after, so both winding estimates resolved.

## Primary endpoints

| Endpoint | Value | Gate | Result |
| --- | ---: | --- | --- |
| projected normal rank | 1 | descriptive | — |
| predicted radius | 0.227577 | — | — |
| direct nonlinear radius | 0.254501 | — | — |
| predicted/direct ratio | 0.894211 | `[0.75,1.25]` | pass |
| direct residual | 6.40e-7 | `<=1e-5` | pass |
| predicted-direction residual | 3.81e-7 | `<=1e-4` | pass |
| predicted crossing scale | 0.260991 | `<=1.25 × direct radius` | pass |
| directional rank | best 10% of 33 | best 10% | pass |
| winding before / after | 0 / 1 | resolved transition | pass |

The direct optimizer reported `success: false` under its generic stopping flag,
but reached the preregistered feasibility residual in three Gauss–Newton iterations.
The scientific gate is the retained residual and radius, not that library flag.

## Interpretation and boundaries

Within this edit space, the linearized normal geometry predicts intervention cost
substantially better than generic directions and remains accurate after nonlinear
correction. This is real evidence for a local structural radius associated with
the observed degree defect.

Direct feasibility is nonconvex and does not prove a globally minimum distance.
The result covers one transition, seed, parameter subspace, and metric. Replication
across seeds, layers, and edit spaces is required before treating the radius as a
model-family invariant.

## Artifacts and reproduction

| Artifact | Path |
| --- | --- |
| refined result | `data/experiments/tinyllm_normal_kernel_radius/20260806_d6_step15_refined/results.json` |
| original unresolved run | `data/experiments/tinyllm_normal_kernel_radius/20260806_d6_step15/results.json` |
| refined result SHA-256 | `6aad11f848a6910054963a0c2a32d0d63cba79eceed09522ed331847d569ee3a` |

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
pixi run python -m experiments.structure_net.tinyllm_normal_kernel_radius \
  --device cuda:0 --degree-phase-points 4096 \
  --output data/experiments/tinyllm_normal_kernel_radius/20260806_d6_step15_refined
```
