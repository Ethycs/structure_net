# TinyLLM source-fitted decoder-boundary defect basis

**Status:** NOT CONFIRMED — ONE-DIRECTION REPAIR IS NOT BOUNDARY-SPECIFIC  
**Date:** 2026-08-06  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, `UNDERPOWERED`  
**Hypothesis:** `tinyllm-c2-defect-boundary-basis-v1`  
**Preregistration:** [`2026-08-06_tinyllm-defect-boundary-basis-preregistration.md`](../07%20-%20Status%20Reports/2026-08-06_tinyllm-defect-boundary-basis-preregistration.md)

## Verdict

A single source-fitted decoder-boundary direction repairs every frozen
near-rank state, but the repair is not specific to local boundary geometry.
`G + B_1` passed all four held-out cells in all three checkpoints. The next
ordinary geometric singular direction and a direction fitted after shuffling
source residual/normal membership did exactly the same. The preregistered
specificity gate therefore passed `0/3`, and the boundary-correction
hypothesis is **not confirmed**.

The random residual controls matter. Matched random directions inside the
remaining source span did not become sufficient in any checkpoint. The result
does not say that every added direction works. It says the successful
direction is a stable ordered residual component that can be recovered by
geometric SVD or by several highly correlated source-weighting schemes; the
per-example decoder-boundary pairing is not the causal explanation.

This independently confirms the adjacent-rank result from the decoder-boundary
audit:

```text
seed 7:  1 + 1 = rank 2
seed 29: 4 + 1 = rank 5
seed 53: 2 + 1 = rank 3
```

The predecessor's ranks `2/8/4` were minima on a dyadic test grid, not exact
minimum ranks.

## Campaign integrity

The campaign reused the three stable block-0 checkpoints, their exact `C2`
orbit generator, and the predecessor's source and held-out cohort seeds. It
computed one batched frozen-decoder backward pass per source cell, fitted no
predictive observer, used no target label, and trained no model.

| Item | Value |
| --- | --- |
| requested / completed / failed / excluded | 3 / 3 / 0 / 0 |
| trained models / predictive observers | 0 / 0 |
| source boundary normals | 128 per checkpoint |
| held-out cells | 2 cohorts x 2 shifts x 3 checkpoints |
| correction counts | 1, 2, 4 |
| device | NVIDIA GeForce RTX 2060 SUPER, CUDA |
| PyTorch / Python | 2.5.1+cu121 / 3.11.13 |
| summed per-seed analysis time | 12.6 seconds |
| implementation SHA-256 | `ae578ba04b9eb1b4b44a155295f159e6c9b35f1eab799a95d32d817f845b1fd5` |
| campaign SHA-256 | `c4cbcec89a0f64a94da29c4add90f61bdbc4b6a618caa70779663c1afbdd1a39` |
| scientific payload SHA-256 | `54acd44cd92584e65b557b3d0197a76f7d65232857798c4f10cf75cb0d2b3267` |

The exact/zero endpoints and predecessor failure cells reproduced in all three
checkpoints. Every inherited head decomposition error remained below `1e-6`,
all combined bases were orthonormal within `1e-8`, and all Fisher denominators
were nondegenerate. Ten focused new/predecessor tests passed before launch. A
separate eight-orbit CUDA lifecycle was labeled systems-only and not pooled.

A post-outcome systems amendment repaired tuple/list canonicalization in the
aggregate resume guard. Deterministic replay preserved the complete
source-basis, held-out, and gate payload byte-for-byte under its declared
scientific hash. A subsequent fingerprint-matched resume returned the existing
aggregate without changing campaign SHA-256.

The campaign is marked `UNDERPOWERED` because it conditions on the three
previously selected stable checkpoints. It supports a mechanism statement for
those checkpoints, not a prevalence claim over training seeds.

## Preregistered gates

Every gate required all three checkpoints.

| Gate | Result | Required | Verdict |
| --- | ---: | ---: | --- |
| basis, decomposition, and nondegeneracy contracts | **3/3** | 3/3 | pass |
| exact/zero endpoint replication | **3/3** | 3/3 | pass |
| predecessor failure-cell replication | **3/3** | 3/3 | pass |
| one paired boundary direction sufficient everywhere | **3/3** | 3/3 | pass |
| shuffled and random controls remain specific | 0/3 | 3/3 | **fail** |

The successful repair cannot rescue the failed mechanism-specificity gate.

## Held-out causal comparison

Worst Fisher preservation is over both held-out cohorts and both shifts. Pass
count is the number of held-out cells clearing the full causal conjunction.

| Seed | Base rank | Base pass / worst Fisher | Geometric +1 | Boundary +1 | Shuffled +1 | Random +1 |
| ---: | ---: | --- | --- | --- | --- | --- |
| 7 | 1 | `0/4 / 0.4600` | `4/4 / 0.9990` | `4/4 / 0.9992` | `4/4 / 0.9992` | `0/4 / 0.4608` |
| 29 | 4 | `3/4 / 0.999866` | `4/4 / 0.999991` | `4/4 / 0.999978` | `4/4 / 0.999993` | `3/4 / 0.999866` |
| 53 | 2 | `2/4 / 0.9634` | `4/4 / 0.999922` | `4/4 / 0.999821` | `4/4 / 0.999839` | `2/4 / 0.9643` |

The seed-29 failure is especially sharp. On held-out-B extrapolation, base
rank 4 and the random correction remain at `0.671875` exact-bin accuracy. The
geometric, paired-boundary, and shuffled-boundary directions all restore
`0.6875` and pass. All three structured interventions agree with the exact
winner on every orbit in that cell.

For seed 53, the next geometric direction performs at least as well as the
task-weighted direction. It raises the two failed composition accuracies from
`0.609/0.609` to `0.766/0.891`; the boundary basis yields `0.797/0.891`, and
the shuffled basis yields `0.781/0.891`. These are small numerical differences
inside the same causal regime, not evidence for unique boundary localization.

## What the source normals measured

The first boundary-weighted singular direction captured a large fraction of
the source boundary-residual energy:

| Seed | First direction energy | Mean residual/normal cosine, comp. / extrap. | Exact winner changed from base, comp. / extrap. |
| ---: | ---: | ---: | ---: |
| 7 | 0.956 | `0.782 / 0.757` | `82.8% / 84.4%` |
| 29 | 0.755 | `0.020 / -0.021` | `0.0% / 1.6%` |
| 53 | 0.764 | `0.267 / 0.262` | `50.0% / 43.8%` |

Seed 29 is incompatible with a strong local-normal account: its omitted
residual is nearly orthogonal on average to the source top-two normals and
almost never changes the source winner. Nevertheless, SVD of the small signed
normal components recovers a held-out-effective direction. Shuffling which
normal is paired with which residual still works, showing that a coherent
global source direction—not example-specific margin pairing—drives the result.

The large first-direction energies are therefore descriptive concentration,
not causal specificity.

## Reconciliation with the direct boundary audit

The earlier direct audit classified seed 29's fifth direction as a one-example
decoder-margin tail and seed 53's third direction as a broad semantic
coordinate. This experiment does not overturn that distinction. It asks a
different constructive question: can source boundary normals identify the
needed direction more specifically than ordinary source geometry?

They cannot. In both cases the next geometric direction is already sufficient,
and shuffled normal pairing retains sufficiency. The supported common account
is:

```text
distributed multi-head synthesis
  -> an ordered 2--3 dimensional semantic carrier
  -> a checkpoint-specific small residual tail
  -> a discrete decoder that can expose the tail at its margins.
```

For architecture work, retain a three-dimensional typed neutral carrier. Do
not add a learned boundary-normal sidecar on the evidence here. Seed 29's tiny
fifth-direction effect belongs more naturally in decoder calibration than in
the semantic carrier dimension.

## Next shortest diagnostic

Stop task-weighted activation-basis fitting. Evaluate a continuous circular
readout and a fixed margin-calibrated discrete readout on the same frozen
rank-2/3 states. If the continuous endpoint remains stable while calibration
alone repairs seed 29, semantic carrier dimension and decoder discretization
separate without retraining the transformer.

## Artifacts and reproduction

- Aggregate:
  `data/experiments/tinyllm_defect_boundary_basis/20260806_d6_preregistered/campaign_results.json`
- Per-seed evidence:
  `data/experiments/tinyllm_defect_boundary_basis/20260806_d6_preregistered/runs/seed_*/result.json`
- Disposable systems-only lifecycle:
  `data/experiments/tinyllm_defect_boundary_basis/shakedown_20260806/`
- Runner: `experiments/structure_net/tinyllm_defect_boundary_basis.py`
- Tests: `tests/structure_net/test_tinyllm_defect_boundary_basis.py`

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
MPLCONFIGDIR=/tmp/matplotlib-structure-net \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
pixi run python -m experiments.structure_net.tinyllm_defect_boundary_basis \
  --device cuda:0 \
  --output data/experiments/tinyllm_defect_boundary_basis/20260806_d6_preregistered
```

## Method boundaries

The basis is conditioned on a frozen decoder and a first-order local normal.
Held-out patches use the exact held-out defect and therefore establish
representational sufficiency, not independent computability. Bases are
checkpoint-local, projected states may be off manifold, and the selected
three-checkpoint cohort remains underpowered.
