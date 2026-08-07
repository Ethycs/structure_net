# TinyLLM defect-subspace decoder-boundary audit

**Status:** NOT CONFIRMED — MIXED BOUNDARY REPAIR AND A DISTINCT THIRD SEMANTIC DIRECTION  
**Date:** 2026-08-06  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, `UNDERPOWERED`  
**Hypothesis:** `tinyllm-c2-defect-boundary-correction-v1`  
**Preregistration:** [`2026-08-06_tinyllm-defect-boundary-audit-preregistration.md`](../07%20-%20Status%20Reports/2026-08-06_tinyllm-defect-boundary-audit-preregistration.md)

## Verdict

The hypothesis that every extra causal defect rank is only an exact-bin margin
correction is falsified. The three preregistered rank near misses split into
one `boundary_only` cell and two `continuous_map_distortion` cells.

Seed 29 is the clean boundary case. Its rank-4 miss differs from the exact
posterior on only one of 64 held-out-B extrapolation orbits, by one adjacent
bin. Mean continuous moment displacement is `0.0149` bin widths, and the one
disagreement lies in the bottom quartile of exact decoder margins. Adding only
25% of the omitted defect, or source singular direction 5 alone, repairs the
task gate. Rank 5 passes every held-out cohort/shift cell.

Seed 53 is qualitatively different. Rank 2 changes 27/64 and 31/64 predictions
in the two composition cells and shifts the posterior moment by `0.364` and
`0.408` bins on average. Only `22.6--29.6%` of disagreements are in the bottom
margin quartile. Source singular direction 3 alone restores near-exact
posteriors and passes all four held-out cells; direction 4 alone does not
repair either composition cell. This is a specific missing semantic coordinate,
not a cloud of decoder-margin corrections.

The refined stable causal ranks are therefore:

```text
seed 7: rank 2  (already exact in the predecessor grid)
seed 29: rank 5 (refined from the dyadic rank-8 bracket)
seed 53: rank 3 (refined from the dyadic rank-4 bracket)
```

## Campaign integrity

The audit reused the source-fitted bases, frozen checkpoints, orbit generator,
and held-out cohorts from the rank campaign. It conditioned on the three
declared failures and is a mechanistic follow-up, not an independent front
replication.

| Item | Value |
| --- | --- |
| requested / completed / failed / reused | 2 / 2 / 0 / 0 |
| primary failure cells / matched held-out cells | 3 / 8 |
| trained models / fitted predictive observers | 0 / 0 |
| interpolation coefficients | 0, .125, .25, .5, .75, .875, 1 |
| individual directions | 5--8 for seed 29; 3--4 for seed 53 |
| device | NVIDIA GeForce RTX 2060 SUPER, CUDA |
| PyTorch / Python | 2.5.1+cu121 / 3.11.13 |
| summed per-seed analysis time | 7.8 seconds |
| maximum decomposition relative error | `2.09e-7` |
| implementation SHA-256 | `20e2d7123313004f9664850a85c26204060020afb4548bb37fcd4f4df4d51521` |
| campaign SHA-256 | `67b338a222ca965f193933194af04fbb110a3b92ce2f982d3a68b3bd064ecd0a` |

Exact and previously sufficient-rank patches passed, while the registered near
rank failed, in every primary cell. Predecessor identities and exact
decomposition passed for both checkpoints. A separate eight-orbit CUDA
lifecycle was systems-only and was not pooled.

The NAL draft recommends five seeds. This targeted audit reused the only two
stable checkpoints with dyadic-rank near misses, so it is marked `UNDERPOWERED`
and makes a mechanism claim only for those frozen cells.

## Preregistered endpoint

| Seed | Cohort / shift | Near rank | Classification | Prediction disagreements | Mean moment shift, bins | Low-margin capture |
| ---: | --- | ---: | --- | ---: | ---: | ---: |
| 29 | held-out B / extrapolation | 4 | **boundary only** | 1/64 | 0.0149 | 1.000 |
| 53 | held-out A / composition | 2 | **continuous distortion** | 27/64 | 0.3644 | 0.296 |
| 53 | held-out B / composition | 2 | **continuous distortion** | 31/64 | 0.4077 | 0.226 |

The registered hypothesis required `boundary_only` in 3/3 cells. It achieved
1/3 and is not confirmed. All prediction disagreements were adjacent-bin, but
adjacency alone is too weak: seed 53 changes a broad fraction of the orbit and
its continuous posterior map, not just examples sitting on decision margins.

## Causal interpolation

Seed 29 crosses the task boundary immediately. Its exact-bin accuracy rises
from `0.6719` to `0.6875` at interpolation coefficient `0.25`; prediction
disagreement simultaneously falls from 1/64 to zero. Direction 5 alone is
sufficient in all four cells, although direction 6 also repairs the primary
cell. The tiny tail is redundant as a margin control.

Seed 53 changes continuously and broadly:

| Coefficient | Held-out A comp. accuracy / disagreement | Held-out B comp. accuracy / disagreement |
| ---: | ---: | ---: |
| 0.000 | 0.609 / 0.422 | 0.609 / 0.484 |
| 0.250 | 0.656 / 0.312 | 0.719 / 0.344 |
| 0.500 | 0.734 / 0.234 | 0.828 / 0.203 |
| 0.750 | 0.797 / 0.109 | 0.891 / 0.078 |
| 1.000 | 0.781 / 0.000 | 0.875 / 0.000 |

More than half of held-out-A disagreements require coefficient `>=0.75` to
match the exact prediction. The smooth trajectory and widespread moment
movement reject a one-example threshold explanation.

## Singular-direction localization

The individual-direction intervention is more decisive than the dyadic rank
grid. In seed 53, adding direction 3 to rank 2:

- passes all four held-out task gates;
- preserves at least `0.999922` of the exact Fisher effect;
- leaves only 1--2 prediction disagreements per cell;
- limits mean moment displacement to `0.0170--0.0178` bins.

Adding direction 4 without direction 3 leaves both composition cells near the
rank-2 failure (`0.594--0.609` accuracy and `0.964--0.970` Fisher
preservation). Direction 3 increases cumulative source defect energy from
`0.8632` to `0.9962`; it is a large, ordered semantic component.

In seed 29, direction 5 raises cumulative energy only from `0.998812` to
`0.999262`. Its effect is exactly the opposite: negligible continuous geometry
but enough boundary-normal displacement to correct one prediction.

## Interpretation and boundaries

The stable quotient front has a small vector-valued core plus a
checkpoint-dependent decoder-margin tail:

```text
semantic defect core: 2--3 source singular directions
decoder-sensitive correction: sometimes one additional tiny direction
stable exact-bin causal rank: 2--5 directions across checkpoints.
```

This resolves the apparent conflict between smooth Fisher rank and hard task
rank. Seed 53 genuinely needs a third semantic coordinate; seed 29's fifth
coordinate should not be promoted into the semantic dimension of the
quotient. For architecture work, a three-dimensional typed carrier is the
smallest currently supported common semantic target, with a separate decoder
margin calibration mechanism.

The basis and directions remain checkpoint-local. The intervention uses the
exact omitted activation and does not establish that a new architecture can
compute it. The classification thresholds are preregistered for this frozen
16-bin task and need not transfer to a different decoder resolution.

The next shortest diagnostic is decoder-side: evaluate a continuous circular
endpoint and a margin-calibrated readout on the same frozen residuals. If ranks
2/3 remain sufficient while only discrete accuracy changes, semantic carrier
dimension and decoder calibration separate without retraining the transformer.

## Artifacts and reproduction

- Aggregate: `data/experiments/tinyllm_defect_boundary_audit/20260806_d6_preregistered/campaign_results.json`
- Per-seed records: `data/experiments/tinyllm_defect_boundary_audit/20260806_d6_preregistered/runs/seed_*/result.json`
- Disposable lifecycle: `data/experiments/tinyllm_defect_boundary_audit/20260806_shakedown_cuda/`
- Runner: `experiments/structure_net/tinyllm_defect_boundary_audit.py`
- Tests: `tests/structure_net/test_tinyllm_defect_boundary_audit.py`

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
MPLCONFIGDIR=/tmp/matplotlib-structure-net \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
pixi run python -m experiments.structure_net.tinyllm_defect_boundary_audit \
  --device cuda:0 \
  --output data/experiments/tinyllm_defect_boundary_audit/20260806_d6_preregistered
```

