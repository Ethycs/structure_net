# TinyLLM defect-subspace decoder-boundary audit

**Status:** NOT CONFIRMED — CORRECTIVE REPLICATION FINDS MIXED BOUNDARY REPAIR AND A DISTINCT THIRD SEMANTIC DIRECTION
**Date:** 2026-08-06  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED` WITH POST-OUTCOME IMPLEMENTATION CORRECTION, `UNDERPOWERED`
**Hypothesis:** `tinyllm-c2-defect-boundary-correction-v1`  
**Preregistration:** [`2026-08-06_tinyllm-defect-boundary-audit-preregistration.md`](../07%20-%20Status%20Reports/2026-08-06_tinyllm-defect-boundary-audit-preregistration.md)

## Verdict

The hypothesis that every extra causal defect rank is only an exact-bin margin
correction is **not confirmed**. The schema-v1.1 post-outcome corrective
replication split the three registered rank near misses into one
`boundary_only` cell and two `continuous_map_distortion` cells. This supports
rejection of the uniform boundary-only mechanism, but it is not fresh
confirmatory evidence because the classifications were visible before the
reproduction-gate implementation was corrected.

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
| summed per-seed analysis time | 8.9 seconds |
| maximum decomposition relative error | `2.09e-7` |
| maximum predecessor Fisher drift | `5.71e-9` |
| implementation SHA-256 | `f6c91f13a539acd64a925b3176bdbfbe77a47d932c992fbd147d207cf9e43ae4` |
| campaign SHA-256 | `869cd0bd6160164e2a83810e7088a4232a278767d1e76bec1fae59247ada8490` |

Exact and previously sufficient-rank patches passed, while the registered near
rank failed, in every primary cell. Predecessor identities, all 24 near/
sufficient/exact causal-label comparisons, Fisher reproduction, and exact
decomposition passed for both checkpoints. Auxiliary moment/increment
diagnostics drifted by at most `6.33e-8` and were reported but were never a
declared predecessor gate. A separate eight-orbit schema-v1.1 CUDA lifecycle
was systems-only and was not pooled.

The NAL draft recommends five seeds. This targeted audit reused the only two
stable checkpoints with dyadic-rank near misses, so it is marked `UNDERPOWERED`
and makes a mechanism claim only for those frozen cells.

## Correction and evidence status

Three roots are preserved. The original `20260806_d6_preregistered` root was
produced concurrently by digest `20e2d7...` without the required predecessor-
endpoint or joint aggregate controls; its producing source state is not the
audited runner and it is excluded. The `_v2` root added those controls, but its
implementation incorrectly applied the declared `1e-8` Fisher tolerance to
auxiliary derived diagnostics. It consequently failed its implemented
reproduction gate at `6.33e-8` even though every causal label matched and the
Fisher errors were below the declared tolerance.

The authoritative `_v3` correction uses schema
`nal.tinyllm-defect-boundary-audit.v1.1`, gates exactly the preregistered causal
labels and Fisher values, and labels itself
`post_outcome_corrective_replication_evidence`. It preserved the same
classification pattern, passed predecessor reproduction 2/2, and remained
byte-identical across fingerprint-matched resume. No boundary threshold,
registered cell, interpolation coefficient, or classification rule changed.

## Preregistered endpoint

| Seed | Cohort / shift | Near rank | Classification | Prediction disagreements | Mean moment shift, bins | Low-margin capture |
| ---: | --- | ---: | --- | ---: | ---: | ---: |
| 29 | held-out B / extrapolation | 4 | **boundary only** | 1/64 | 0.0149 | 1.000 |
| 53 | held-out A / composition | 2 | **continuous distortion** | 27/64 | 0.3644 | 0.296 |
| 53 | held-out B / composition | 2 | **continuous distortion** | 31/64 | 0.4077 | 0.226 |

The registered hypothesis required `boundary_only` in 3/3 cells. The
corrective replication achieved 1/3 and the hypothesis is not confirmed. All
prediction disagreements were adjacent-bin, but
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

- Corrective aggregate: `data/experiments/tinyllm_defect_boundary_audit/20260806_d6_preregistered_v3/campaign_results.json`
- Corrective per-seed records: `data/experiments/tinyllm_defect_boundary_audit/20260806_d6_preregistered_v3/runs/seed_*/result.json`
- Preserved failed-gate attempt: `data/experiments/tinyllm_defect_boundary_audit/20260806_d6_preregistered_v2/`
- Preserved nonconforming original: `data/experiments/tinyllm_defect_boundary_audit/20260806_d6_preregistered/`
- Disposable schema-v1.1 lifecycle: `data/experiments/tinyllm_defect_boundary_audit/shakedown_20260806_v4/`
- Meta-hypothesis record: `data/meta_hypotheses/tinyllm-c2-defect-boundary-correction-v1.json`
- Runner: `experiments/structure_net/tinyllm_defect_boundary_audit.py`
- Tests: `tests/structure_net/test_tinyllm_defect_boundary_audit.py`

The meta-hypothesis write was verified by authoritative Chroma readback of the
named hypothesis and both experiment records. The legacy Chroma dependency
emitted NumPy-2.0 consumer and telemetry warnings during the write; these are
transport diagnostics, not failed evidence gates, and the JSON ledger remains
the portable source record.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
MPLCONFIGDIR=/tmp/matplotlib-structure-net \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
pixi run python -m experiments.structure_net.tinyllm_defect_boundary_audit \
  --device cuda:0 \
  --output data/experiments/tinyllm_defect_boundary_audit/20260806_d6_preregistered_v3
```
