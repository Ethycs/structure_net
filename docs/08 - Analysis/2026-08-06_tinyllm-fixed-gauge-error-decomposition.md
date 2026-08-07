# TinyLLM fixed-gauge error decomposition

**Status:** VALID NEGATIVE — LINEAR WRITER, NOT SENSOR PRECISION, LIMITS 3/3  
**Date:** 2026-08-06  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, `UNDERPOWERED`, post-outcome diagnostic  
**Hypothesis:** `tinyllm-c2-fixed-gauge-error-decomposition-v1`  
**Preregistration:** [`2026-08-06_tinyllm-fixed-gauge-error-decomposition-preregistration.md`](../07%20-%20Status%20Reports/2026-08-06_tinyllm-fixed-gauge-error-decomposition-preregistration.md)

## Verdict

Perfect latent-phase substitution does not rescue the fixed-gauge causal
writer. All three frozen checkpoints satisfy the replay, oracle-carrier, and
target-control contracts, yet both observed-fit/oracle-evaluated and
oracle-fit/oracle-evaluated writers fail the all-cell continuous endpoint in
`3/3` checkpoints. Every checkpoint is classified
`writer_or_carrier_limited`.

The sensor-only branch is therefore falsified:

```text
replace observed carrier with exact quotient coordinate
    -> 0/3 checkpoint causal gates
    -> do not train a sensor-only sidecar.
```

Because `(cos(2 phi), sin(2 phi))` completely parameterizes the declared
degree-two semantic quotient and the direct rank-three target control passes,
the remaining candidate is not missing phase information or activation rank.
It is the restricted quotient-to-write relation: a single linear,
nuisance-blind map does not reproduce the checkpoint's task-effective write.

## Four-way intervention

The diagnostic keeps all primary data, bases, thresholds, readouts, and frozen
continuations unchanged and compares:

| Fit carrier | Evaluation carrier | Purpose |
| --- | --- | --- |
| observed | observed | exact primary replay |
| observed | oracle | sensor substitution only |
| oracle | observed | fit/evaluation mismatch |
| oracle | oracle | ideal fixed-gauge linear-writer capacity |

The oracle is `(cos(2 phi), sin(2 phi), 1)` computed from latent phase. It is a
diagnostic only and is not deployable. One new no-intercept ridge writer is
fitted per checkpoint; no model or observer is trained.

## Campaign integrity

| Item | Value |
| --- | --- |
| requested / completed / failed / excluded | 3 / 3 / 0 / 0 checkpoints |
| trained models / predictive observers | 0 / 0 |
| newly fitted writers | 3 oracle writers |
| held-out cells | 12 per evaluated state |
| device | NVIDIA GeForce RTX 2060 SUPER, CUDA |
| PyTorch / Python | 2.5.1+cu121 / 3.11.13 |
| implementation SHA-256 | `37175d78dd4768e448bea482a271309e67f00863ffa6202a799346fad008ac38` |
| campaign SHA-256 | `be4cf87248550c8ace7fc474efff2ce168bce3c9002932ecf6063977c91f0fa6` |
| primary campaign SHA-256 | `de80e30c23e06801c75d6fae899c67d0da82b86fdaff9158d94270597df8379c` |
| final DVC data root | `f29e1f0e920aff74661e2a64d7ec56c1.dir` (`1,796` files, `39,812,097,258` bytes) |
| lakeFS snapshot | `71cda38c5b84bfa364c136a0741dd4ff6e77040395f4e24b5d50d8419c11a648` |

All per-result hashes match, the current implementation digest matches the
stored campaign, and the observed/observed continuous and coordinate records
replay the primary campaign with maximum absolute error exactly zero. Runtime
was `11.70` seconds. The final DVC root is current locally, was pushed to the
configured `lakefs://artifacts/main/structure-net/` remote, and is contained in
the cited clean lakeFS commit.

## Contracts and classifications

| Gate | Result | Required | Verdict |
| --- | ---: | ---: | --- |
| primary replay | **3/3** | 3/3 | pass |
| exact oracle carrier | **3/3** | 3/3 | pass |
| continuous target controls | **3/3** | 3/3 | pass |
| observed-fit / oracle-eval | **0/3** checkpoints | 3/3 | fail |
| oracle-fit / oracle-eval | **0/3** checkpoints | 3/3 | fail |

Oracle phase error is at most `6.24e-16` output bins, far below the `1e-8`
contract. Thus the classification cannot be attributed to numerical phase
error. Each oracle writer still predicts held-out target coordinates well:
worst `R2 = 0.96869`, mean `R2 = 0.98420`.

## Causal result

| Seed | Oracle-fit R2 on fit | Oracle/oracle mean shifts across cells | Passing cells | Classification |
| ---: | ---: | --- | ---: | --- |
| 7 | 0.98538 | 0.2478, 0.2591, 0.2073, 0.2498 | 0/4 | writer/carrier limited |
| 29 | 0.97409 | 0.1696, 0.2556, 0.1647, 0.3368 | 0/4 | writer/carrier limited |
| 53 | 0.99071 | 0.1563, 0.1729, 0.1233, 0.1917 | 1/4 | writer/carrier limited |

Across the 12 oracle/oracle cells, mean causal shift is `0.21124` bins and
only seed-53 heldout-B composition passes. The observed-fit/oracle-evaluated
condition is nearly identical: mean `0.20841` and the same `1/12` cell pass.
Improving the sensor therefore does not even provide a consistent directional
benefit.

## Mechanistic interpretation

The combined controls remove several tempting explanations:

- It is not observation-side phase quantization: the exact oracle fails.
- It is not a missing three-dimensional target subspace: direct rank-three
  patches pass in every cell.
- It is not an arbitrary marginal effect: shuffled writers fail in the
  primary campaign.
- It is not poor descriptive correspondence: held-out coordinate `R2` remains
  about `0.97--0.99`.

The live ambiguity is now smaller. The task-effective defect coordinates may
form a curved function of quotient phase, or the frozen continuation may
require invariant nuisance/context conditioning at the write. Both violate
the tested single linear, nuisance-blind interface.

This strengthens the overall interpretability conclusion: the carrier is a
causal chart, not merely a representation vector. Its write map and local task
metric are part of the mechanism.

## Decision and shortest next test

Do not train the proposed sensor-only equivariant sidecar. Before any new
model training, run one frozen nested capacity diagnostic on these exact
artifacts:

1. quotient-only Fourier writers of fixed increasing harmonic order;
2. the same writers augmented with declared observation-derived invariant
   nuisance/context summaries;
3. a shuffled-correspondence control at each capacity;
4. the unchanged four held-out causal cells and direct-rank-three controls.

If quotient-only nonlinear writers pass, the limitation is curvature and the
minimal sidecar needs a nonlinear neutral synthesis. If only context-augmented
writers pass, the interface must be a typed conditional write. If neither
passes at a small preregistered capacity, stop fitting sidecars and map the
downstream nonlinear continuation directly.

## Artifacts and reproduction

- Aggregate:
  `data/experiments/tinyllm_fixed_gauge_error_decomposition/20260806_d6_preregistered_diagnostic/campaign_results.json`
- Per-checkpoint records:
  `data/experiments/tinyllm_fixed_gauge_error_decomposition/20260806_d6_preregistered_diagnostic/runs/seed_*/result.json`
- Systems-only lifecycle:
  `data/experiments/tinyllm_fixed_gauge_error_decomposition/20260806_shakedown_cuda/`
- Runner:
  `experiments/structure_net/tinyllm_fixed_gauge_error_decomposition.py`
- Tests:
  `tests/structure_net/test_tinyllm_fixed_gauge_error_decomposition.py`
- Meta-hypothesis:
  `data/meta_hypotheses/tinyllm-c2-fixed-gauge-error-decomposition-v1.json`

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
MPLCONFIGDIR=/tmp/matplotlib-structure-net \
pixi run python -m \
  experiments.structure_net.tinyllm_fixed_gauge_error_decomposition \
  --device cuda:0 \
  --output \
  data/experiments/tinyllm_fixed_gauge_error_decomposition/20260806_d6_preregistered_diagnostic
```

## Method boundaries

The oracle uses latent phase and cannot be deployed. This is a post-outcome,
three-checkpoint diagnostic on reused held-out cells. Target-local writers have
alignment-fit access, patches are off manifold, and the experiment does not
test a nonlinear writer, context-conditioned write, learned encoder, or
population prevalence.
