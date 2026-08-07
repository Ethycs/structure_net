# TinyLLM group-anchored task-metric carrier transport

**Status:** NOT CONFIRMED — ONE PAIRWISE GAUGE CLASS, NO GLOBAL CHART  
**Date:** 2026-08-06  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, `UNDERPOWERED`  
**Hypothesis:** `tinyllm-c2-group-task-metric-carrier-transport-v1`  
**Preregistration:** [`2026-08-06_tinyllm-group-task-metric-transport-preregistration.md`](../07%20-%20Status%20Reports/2026-08-06_tinyllm-group-task-metric-transport-preregistration.md)

## Verdict

A label-free pullback Fisher metric recovers a causally transportable carrier
chart between checkpoints 7 and 53 in both directions, but it does not produce
one global chart across all three selected checkpoints. The two successful
maps pass all eight held-out cohort/shift cells. The four maps involving seed
29 fail the preregistered all-cell causal endpoint, including every
outside-range extrapolation cell.

The complete hypothesis is therefore not confirmed:

```text
same C2 invariant type
  + locally stable task metric
  does not imply
one global checkpoint-independent carrier gauge.
```

The diagnostic nevertheless identifies a real pairwise equivalence class.
Across the 7-to-53 and 53-to-7 cells, task-metric fitting reduces average
circular-moment shift from `0.158` bins for the better Euclidean affine
baseline to `0.081` bins, and all eight cells pass. Shuffled correspondence
fails decisively.

## Campaign integrity

The study reused three frozen 29,956,608-parameter d6 TinyLLM checkpoints, the
same exact `C2` Reynolds defects, the same rank-three bases, and the same 24
held-out cells as the preceding cross-seed campaign. It trained no model,
frontend, probe, observer, decoder, or calibration.

| Item | Value |
| --- | --- |
| requested / completed / failed / excluded | 6 / 6 / 0 / 0 directed pairs |
| trained models / predictive observers | 0 / 0 |
| newly fitted maps | 12: paired task-metric plus shuffled per direction |
| total evaluated maps | 30 |
| fit / held-out cells per pair | 2 / 4 |
| exact orbits per cell | 64 |
| device | NVIDIA GeForce RTX 2060 SUPER, CUDA |
| PyTorch / Python | 2.5.1+cu121 / 3.11.13 |
| implementation SHA-256 | `7e70d42d978fec1f05d2d7c5ee9d9c8d02acb70e9e27291f44942dcbdcd1cea8` |
| campaign SHA-256 | `dba5ed1e8304edfb0f314e1488d00cc9865fe58d5e9ce0fcdfaba76d87cb9138` |
| DVC data root | `25d2be3b471682646ba2fc4404de412a.dir` (`1,633` files, `39,807,359,282` bytes) |
| lakeFS commit | `3f87160c02c0fc051335706c383f682f7f8b7e2a0d4f2123ffc5b4d3d18c5509` |

The predecessor campaign is locked to SHA-256
`44707fd4bcd810e63614671aa491095fae735ee52359d464ab25abb10a2bc228`.
All six stored predecessor maps reproduced their continuous metrics with
maximum absolute error exactly zero. A completed resume left the aggregate
bytes unchanged. A separate two-seed, eight-orbit CUDA lifecycle is retained
as systems-only evidence and was not pooled.

## Primary gates

| Gate | Result | Required | Verdict |
| --- | ---: | ---: | --- |
| finite-difference metric contract | **6/6** | 6/6 | pass |
| continuous target controls | **6/6** | 6/6 | pass |
| predecessor replay contract | **6/6** | 6/6 | pass |
| task-metric coordinate transport | **3/6** | 6/6 | fail |
| task-metric continuous causal transport | **2/6** | 6/6 | fail |
| dominance over Euclidean baselines | **2/6** | 6/6 | fail |
| shuffled specificity | **6/6** | 6/6 | pass |
| complete hypothesis | **not confirmed** | every gate | fail |

The scalar-calibrated exact-bin endpoint remains secondary by preregistration.
This experiment tests the continuous carrier metric and does not reinterpret
seed 7's known fresh-cohort boundary-calibration instability.

## Directed-pair result

| Pair | Task-metric mean shift, bins | Better Euclidean mean | Held-out continuous cells | Coordinate gate | Pair verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| 7 -> 29 | 0.2871 | 0.2397 | 0/4 | fail | fail |
| 7 -> 53 | **0.0828** | 0.1560 | **4/4** | pass | pass |
| 29 -> 7 | 0.2044 | 0.2409 | 0/4 | fail | fail |
| 29 -> 53 | 0.1557 | 0.1662 | 2/4 | fail | fail |
| 53 -> 7 | **0.0792** | 0.1594 | **4/4** | pass | pass |
| 53 -> 29 | 0.1764 | 0.1973 | 1/4 | pass | fail |

Across all 24 cells, task-metric fitting raises continuous passes from `0` to
`11` and lowers mean shift from `0.194` for affine ridge to `0.164` bins. That
aggregate improvement is not the claim: it is concentrated in one checkpoint
pair and in composition.

For the sixteen cells involving seed 29, task-metric mean shift is `0.206`
versus `0.211` for affine ridge, with only `3/16` cells passing. Composition
improves (`0.146` versus `0.183`), but extrapolation does not (`0.266` versus
`0.240`) and passes `0/8`. The failure is therefore support-relative, not a
uniform lack of task weighting.

## Numerical metric contract

The Fisher geometry is well resolved at the declared finite-difference scale:

- target-seed median full/half-step relative errors are
  `0.00111`, `0.00161`, and `0.00232`;
- the maximum per-orbit relative error is below `0.0098`;
- minimum raw Fisher traces are `0.105--0.164`, far above the `1e-10` floor;
- raw minimum eigenvalues remain positive, about `1.5e-7--5.2e-7`;
- all shuffled maps fail while paired mean shifts beat them by several bins.

The weighted normal systems are anisotropic, with condition numbers around
`1.4e5--4.6e5`, but the preregistered isotropic floor and ridge make them finite
and deterministic. The half-step agreement and exact predecessor replay argue
against finite-difference noise as the cause of the pair split.

## Mechanistic interpretation

The preceding Euclidean result was too pessimistic for seeds 7 and 53: their
rank-three `C2` invariant carriers are causally gauge-equivalent once the
frozen continuation supplies the correct local metric. Their ordinary
coordinate residuals were concentrated in directions that Euclidean fitting
overweighted.

Seed 29 is different. When it is the source, task-metric fit sacrifices large
amounts of target-coordinate variance (`R2` about `0.53--0.70`) and still fails
extrapolation. When it is the target, 53-to-29 retains about `0.96` held-out
coordinate `R2` but still fails all extrapolation cells. Thus seed 29 cannot be
explained solely by an invertible linear gauge with a misweighted target
metric. Its chart relation is nonlinear, support-relative, or both.

The complementary
[`carrier-Jacobian axis audit`](2026-08-06_tinyllm-carrier-jacobian-axis-audit.md)
showed that each individual transport error is locally task-linear in all six
directions. Taken together, the results localize the failure more tightly: the
continuation is locally predictable, but no single affine source-to-target map
satisfies the orbit-varying task metric across both supports when seed 29 is
involved.

This is a representation-theoretic limitation worth preserving:

```text
irrep type is architectural;
multiplicity-space coordinates and causal metric need not be universal.
```

## Decision and next architecture

Per preregistration, stop post-hoc carrier alignment here. Another polynomial,
neural, or held-out-adapted map could fit seed 29 but would no longer test a
portable causal interface.

The constructive next step is an explicitly typed `C2` sidecar whose gauge is
fixed during training:

1. expose trivial and signed character channels before the block-0 synthesis
   front;
2. allow only declared neutral fusion, including the `c1 tensor c1 -> c0`
   bilinear path;
3. emit three normalized invariant carrier channels in a fixed shared metric;
4. use the already separated scalar decoder calibration after the carrier;
5. compare the sidecar with the frozen/raw TinyLLM control across composition
   and extrapolation, with cross-seed channel transport as an architectural
   endpoint rather than a post-hoc fit.

This is now justified as an architecture experiment, not another attempt to
persuade unconstrained checkpoints into sharing coordinates after training.

## Artifacts and reproduction

- Aggregate:
  `data/experiments/tinyllm_group_task_metric_transport/20260806_d6_preregistered/campaign_results.json`
- Per-pair records:
  `data/experiments/tinyllm_group_task_metric_transport/20260806_d6_preregistered/runs/source_*_target_*/result.json`
- Systems-only lifecycle:
  `data/experiments/tinyllm_group_task_metric_transport/20260806_shakedown_cuda/`
- Runner:
  `experiments/structure_net/tinyllm_group_task_metric_transport.py`
- Tests:
  `tests/structure_net/test_tinyllm_group_task_metric_transport.py`
- Meta-hypothesis:
  `data/meta_hypotheses/tinyllm-c2-group-task-metric-carrier-transport-v1.json`

```bash
PYTHONPYCACHEPREFIX=/tmp/structure-net-task-metric-primary-pyc \
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
MPLCONFIGDIR=/tmp/matplotlib-structure-net \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
pixi run python -m \
  experiments.structure_net.tinyllm_group_task_metric_transport \
  --device cuda:0 \
  --output \
  data/experiments/tinyllm_group_task_metric_transport/20260806_d6_preregistered
```

## Method boundaries

Only three selected stable checkpoints are tested. The metric is label-free
but requires paired access to the target checkpoint on fit orbits. The maps are
diagnostic gauge fixes, not independently computable encoders. Patches remain
off-manifold interventions. Pairwise success does not establish population
prevalence or a transitive global atlas, and the negative global result does
not rule out a deliberately gauge-fixed equivariant architecture.
