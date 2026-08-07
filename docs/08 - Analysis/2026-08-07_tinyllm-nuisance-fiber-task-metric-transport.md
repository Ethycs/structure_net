# TinyLLM nuisance-fiber task-metric transport

**Status:** NOT CONFIRMED — CAUSAL TRANSPORT IS BROAD BUT NONSPECIFIC  
**Date:** 2026-08-07  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, `UNDERPOWERED`  
**Hypothesis:** `tinyllm-c2-nuisance-fiber-task-metric-transport-v1`  
**Preregistration:** [`2026-08-07_tinyllm-nuisance-fiber-task-metric-transport-preregistration.md`](../07%20-%20Status%20Reports/2026-08-07_tinyllm-nuisance-fiber-task-metric-transport-preregistration.md)

## Verdict

The local rank-two task tangent does **not** define a specific, stable
nuisance-fiber metric field under the preregistered test. Source-fiber
projectors causally repair all 12 matched target cells, almost exactly matching
the target-local projector. But shuffled projectors also pass all 12 cells and
random rank-two controls pass 94 of 96 cell interventions. The causal success
therefore cannot identify symmetry transport as its explanation.

The geometric result is support-relative:

```text
composition:   projector gate passes 6/6 cells
extrapolation: projector gate passes 2/6 cells
```

Median source/target kernel-line cosines remain high (`0.973--0.999`), but the
lower tail falls to `0.308` under extrapolation. Thus a typical local direction
is similar while a minority of orbits rotate enough to reject one global
quotient projector.

The complete hypothesis is not confirmed:

```text
exact neutral C2 type
  + source-fiber causal rescue
  does not imply
a unique nuisance-invariant continuation tangent.
```

## Primary gates

| Gate | Result | Required | Verdict |
| --- | ---: | ---: | --- |
| exact calibration-action contract | 2/3 checkpoints | 3/3 | fail |
| numerical and target controls | 1/3 | 3/3 | fail |
| geometric projector transport | 0/3 | 3/3 | fail |
| causal transported-tangent endpoint | **3/3, 12/12 cells** | 3/3 | pass |
| shuffled/random specificity | 0/3 | 3/3 | fail |
| complete hypothesis | 0/3 | 3/3 | not confirmed |

The validity misses are reported literally and do not get repaired after
outcome inspection. They also need interpretation:

- Seed 53's exact-action maximum projector error is `1.522e-4`, above the
  locked `1e-4` ceiling. The analytic carrier, propagated state, and Jacobian
  errors remain at most `2.98e-7`, `2.98e-7`, and `1.49e-7`; the pseudoinverse
  amplifies that floating-point perturbation. This is a locked numerical
  contract miss, not evidence of a macroscopic failure of the analytic group
  action.
- Seeds 7 and 53 fail the target-control gate because the phase-only predicted
  writer already passes their two fresh composition cells. Their inert kernel
  states consequently pass those same cells. Exact and full states still pass
  all 12 cells, and every Jacobian is finite rank two with decomposition error
  at most `3.79e-12`.

Only seed 29 satisfies both validity contracts. It is classified
`causal_equivalence_without_projector_equality`: causal transport passes, while
geometry and specificity do not.

## Causal intervention

For the target residual `e`, the target-local and source-transported patches
were

```text
local       = Pt e
transported = Ps e,
```

with both patched into the same frozen target continuation. Aggregate circular
moment shifts were:

| Seed | Local tangent | Transported tangent | Shuffled tangent | Median random |
| ---: | ---: | ---: | ---: | ---: |
| 7 | 0.02311 | **0.02289** | 0.05439 | 0.04895 |
| 29 | **0.03463** | 0.03589 | 0.07878 | 0.07779 |
| 53 | **0.01753** | 0.01788 | 0.04619 | 0.05364 |
| all cells | **0.02509** | 0.02555 | 0.05979 | — |

Transport changes the target-local result by only `-0.00022`, `+0.00126`, and
`+0.00035` bins across checkpoints. That is real frozen causal equivalence.
It is not specific: all eight random checkpoint-level controls pass in seeds 7
and 53, six of eight pass in seed 29, and every shuffled checkpoint control
passes. The preregistered `0.125`-bin advantage is nowhere close to satisfied.

This reveals a structural limitation of the rank-two test. In a rank-three
carrier, a generic rank-two projector retains most residual directions. At the
measured endpoint the continuation tolerance is broad enough that many such
patches close the task, so causal endpoint success alone cannot identify the
correct quotient tangent.

## Symmetry geometry

The exact calibration subgroup was declared before execution as

```text
Gcal = (R>0 x SO(2)) semidirect (R2_offset x R2_drift),
rho(g) = I3
```

on the neutral `C2` multiplicity carrier. The locked nonidentity action changes
orientation, amplitude, offset, and drift jointly in the signal and observed
calibration packet. Across all cells, maximum errors before the projector were:

| Quantity | Maximum absolute error |
| --- | ---: |
| analytic carrier | `1.79e-7` |
| propagated state | `2.98e-7` |
| continuation Jacobian | `1.49e-7` |
| row-space projector | `1.52e-4` |

So the analytic neutral representation law is numerically supported before
the ill-conditioned projector operation, while the preregistered projector
contract passes only two checkpoints.

The broader N3 pairing is an observation groupoid: paired examples share the
same quotient phase but independently vary speed, harmonics, noise, and the
calibrated nuisance fields. Its cellwise kernel-line results are:

| Shift | Median-cosine gate | p10 gate | Joint cell gate |
| --- | ---: | ---: | ---: |
| composition | 6/6 | 6/6 | **6/6** |
| extrapolation | 6/6 | 2/6 | **2/6** |

The failure is therefore concentrated in extrapolation tails, not in the
typical orbit or the known analytic calibration symmetry.

## Mechanistic conclusion

Symmetry groups remain useful here, but they constrain the architecture more
cleanly than they identify this post-hoc rank-two field.

The experiment supports three narrower claims:

1. The analytic calibrated carrier respects the declared target-preserving
   group to floating-point accuracy before pseudoinverse amplification.
2. Local continuation row spaces are usually close across matched nuisance
   fibers and remain causally interchangeable at the current endpoint.
3. That interchangeability is not unique and loses uniform geometric control
   in extrapolation tails.

It does **not** support a portable law of the form `P(gx)=P(x)` as the minimal
typed sidecar interface. The correct current abstraction is still an
orbit-local atlas, with a known neutral group type but context-dependent task
covectors.

An independently preregistered complementary campaign reached the same
universal-law boundary using exact acquisition-similarity arms applied before
tokenization. Its initial same-phase nuisance shuffle was later recognized as
a positive equivalence control, not a valid negative control. A locked
corrective uses semantic-phase shifts and random planes: those specificity
controls pass `3/3`, but exact paired tail geometry passes only seed 29 and the
universal identity law remains rejected at `1/3`. Thus the present conclusion
is not an artifact of pairing independent full-N3 nuisance draws, although the
correct reason is checkpoint-stratified tail geometry rather than failure of
same-phase nuisance equivalence. See the
[`local metric-field specificity corrective`](2026-08-07_tinyllm-local-metric-field-specificity.md).

This also limits the relevance of link cobordism. A cobordism of singular or
zero sets could describe how exceptional extrapolation orbits appear as the
nuisance path changes, but it would not fix the specificity failure: topology
cannot distinguish the intended rank-two projector when almost every matched
rank-two control is causally sufficient.

## Decision and next shortest test

Do not train the equivariant sidecar yet. The complementary
[`source task-covector portability`](2026-08-07_tinyllm-source-task-covector-portability.md)
study has already removed the generic rank-two loophole on a fresh cohort. Its
rank-one phase-conditioned covector is portable in all three checkpoints
(`R2 0.989--0.996`, mean row cosine `0.9969--0.9995`) and, when supplied the
fresh signed error, closes all six fresh cells. The phase-only signed amplitude
is not portable (`R2 -0.053--0.041`), so the fully source-predicted correction
passes only one of six cells.

The next shortest test is therefore an **observable scalar residual sensor**,
not another covector or projector transport. Freeze the portable covector and
test candidate scalar inputs in increasing-cost order:

1. continuation confidence and circular-posterior residual statistics;
2. the already observed calibration packet;
3. a minimal local activation summary.

Fit on source cohorts A/B and evaluate once on a fresh D cohort. If no scalar
predicts the sign and magnitude, stop attempting a deployable source-only
sidecar and retain the covector as an explanatory diagnostic. Any equivariant
sidecar after that is a prospective architectural intervention, not a recovered
universal field from these checkpoints.

## Campaign integrity

| Item | Value |
| --- | --- |
| requested / completed / failed / excluded | 3 / 3 / 0 / 0 checkpoints |
| fresh matched cells | 12 |
| exact `C2` orbits per cell | 64 |
| trained models / fitted writers / fitted observers | 0 / 0 / 0 |
| evaluated causal states | local, transported, shuffled, 8 random, exact/full/zero |
| device | NVIDIA GeForce RTX 2060 SUPER, CUDA |
| PyTorch / Python | 2.5.1+cu121 / 3.11.13 |
| peak allocated CUDA memory | 0.290 GiB |
| primary analysis time | 16.83 seconds |
| implementation SHA-256 | `2c18c1e330f974d59a6e07367b223961322b8ea973c9008374a5431f3b2d0e60` |
| campaign SHA-256 | `ccd4444a4091eda06ce699fdb219e751fac44cca91f47142dd73e73a8a6abc55` |
| final DVC data root | `9f9077c17fbbc668805088bf604deafc.dir` (`1,904` files, `39,816,811,567` bytes) |
| lakeFS snapshot | `8eccad2c763ea0230fde1e484b2d8c631dbe91524799c21920686bd23d704872` |

The writer campaign is locked to
`7ab6079d56bc8dc78cc5e1f4011c581b8f507bcd8ec20573bbf8b11227df8d1b`
and the corrected local-Jacobian method campaign to
`8b2f25dc61d1ffb93ba9dc66f1a85f36c07bda5a43ed3ed34a9627ca1485c12a`.
A fingerprint-matched resume preserved the primary campaign bytes exactly.
The final DVC root was pushed to the configured
`lakefs://artifacts/main/structure-net/` remote and is contained in the cited
clean lakeFS commit; the branch reports zero uncommitted objects.

The selected three-checkpoint cohort remains underpowered. The CUDA shakedown
used one checkpoint and eight orbits, is marked
`systems_lifecycle_only_not_quality_evidence`, and is not pooled.

## Artifacts and reproduction

- Aggregate:
  `data/experiments/tinyllm_nuisance_fiber_task_metric_transport/20260807_d6_preregistered_diagnostic/campaign_results.json`
- Per-checkpoint records and arrays:
  `data/experiments/tinyllm_nuisance_fiber_task_metric_transport/20260807_d6_preregistered_diagnostic/runs/seed_*/`
- Systems-only lifecycle:
  `data/experiments/tinyllm_nuisance_fiber_task_metric_transport/20260807_shakedown_cuda/`
- Runner:
  `experiments/structure_net/tinyllm_nuisance_fiber_task_metric_transport.py`
- Tests:
  `tests/structure_net/test_tinyllm_nuisance_fiber_task_metric_transport.py`
- Meta hypothesis:
  `data/meta_hypotheses/tinyllm-c2-nuisance-fiber-task-metric-transport-v1.json`

```bash
PYTHONPYCACHEPREFIX=/tmp/structure-net-nuisance-metric-primary-pyc \
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
MPLCONFIGDIR=/tmp/matplotlib-structure-net \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
pixi run python -m \
  experiments.structure_net.tinyllm_nuisance_fiber_task_metric_transport \
  --device cuda:0 \
  --output \
  data/experiments/tinyllm_nuisance_fiber_task_metric_transport/20260807_d6_preregistered_diagnostic
```

## Method boundaries

The exact subgroup contract is a positive numerical control, not the primary
nuisance test. The broader N3 relation is a groupoid, not one globally
invertible group. Exact target residuals are diagnostic causal inputs. Patches
remain off-manifold. The endpoint is permissive for rank-two subspaces. Three
selected checkpoints do not establish population prevalence, and the negative
specificity result does not rule out a prospectively constrained equivariant
architecture.
