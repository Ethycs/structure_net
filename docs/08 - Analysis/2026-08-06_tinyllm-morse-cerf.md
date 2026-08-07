# TinyLLM Equivariant Morse–Cerf Quotient-Front Scan

**Status:** NOT CONFIRMED — NEAR-FRONT EVENTS ARE REAL BUT NOT A STABLE QUOTIENT NORMAL FORM  
**Date:** 2026-08-06  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`  
**Hypothesis:** `tinyllm-equivariant-morse-cerf-quotient-front-v1`  
**Preregistration:** `../07 - Status Reports/2026-08-06_tinyllm-morse-cerf-preregistration.md`

## Verdict

The frozen TinyLLMs do exhibit symmetry-constrained changes in their causal
orbit-mixture landscapes near the exact orbit-averaging front, but the complete
Morse–Cerf hypothesis is not confirmed. The events are not a stable,
degree-independent normal form of quotient formation.

For `k=2`, early cover structure and a near-front index, barrier, component, or
saddle event appeared in 5/5 seeds under both shifts. The stronger mature-basin
gate passed 2/5 seeds on composition and 5/5 on extrapolation. For `k=3`, early
structure and near-front events reached 4/5 on composition and 5/5 on
extrapolation, but mature basins reached only 2/5 and 3/5. Controls frequently
developed similar mature fronts, especially under extrapolation.

The autonomous-closure gate failed 0/5 for both degrees because its
preregistered Morse-index condition is structurally miscalibrated. The
commutator potential is nonnegative and exactly zero at every simplex vertex;
as a sublayer becomes globally Reynolds-compatible it approaches the flat-zero
function, not a nondegenerate index-zero barycenter basin. This gate is retained
unchanged in the primary verdict. A clearly labeled descriptive threshold scan
shows `C(center)<=0.01` early in every cell, but that result cannot rescue the
frozen hypothesis.

## Campaign integrity

All ten retained d6 checkpoints (`k=2,3`; seeds `7,17,29,41,53`) completed
without training or parameter changes. Each cell validated the checkpoint and
the frozen deck-action and Reynolds–Koopman comparators. The primary task
potential was target-posterior KL, exact cyclic symmetrization was applied, and
accuracy remained descriptive.

The first CUDA shakedown selected the wrong physical card because PyTorch's
default ordinal order differed from `nvidia-smi`. It failed while loading the
model and produced no scientific cell. Setting `CUDA_DEVICE_ORDER=PCI_BUS_ID`
mapped physical GPU 2 to logical `cuda:0`; the disposable shakedown and all ten
confirmatory cells then completed. The producing implementation was not edited
during the campaign.

| Item | Value |
| --- | --- |
| requested / completed / failed | 10 / 10 / 0 |
| exact orbits per shift and cell | 32 |
| `k=2` grid | 49-point interval |
| `k=3` grid | 12-subdivision triangle, 91 points |
| primary / control alpha samples per sublayer | 9 / 5 |
| inspected residual sublayers | 6: attention and MLP in blocks 0–2 |
| shifts | composition and extrapolation |
| device | NVIDIA GeForce RTX 2060 SUPER, CUDA |
| PyTorch / Python | 2.5.1+cu121 / 3.11.13 |
| implementation SHA-256 | `4b1cb2ebff5a5b40ddd45528e89dd21869ce77c3569ae101cef289b8b41ce35e` |
| campaign SHA-256 | `2f4987e7c57d76dd5cb9c7f84ab38df1ec7e3735e7fc2ac0e862ec9e1a760185` |
| DVC data root | `92baa9c38d07560be31614facd739289.dir` |
| lakeFS backup commit | `22a5d0501b20ba9a78431f0313d915957b598a3673567f088155d9096ab1d96d` |

## Preregistered gates

Each count is evaluated separately by degree and shift. A gate requires at least
four of five seeds. The full hypothesis requires every row to pass everywhere.

| Gate | `k=2` composition | `k=2` extrapolation | `k=3` composition | `k=3` extrapolation |
| --- | ---: | ---: | ---: | ---: |
| early cover structure | **5/5** | **5/5** | **4/5** | **5/5** |
| near-front Morse event | **5/5** | **5/5** | **4/5** | **5/5** |
| mature barycenter basin | 2/5 | **5/5** | 2/5 | 3/5 |
| control specificity | **4/5** | 3/5 | **4/5** | 2/5 |
| task/closure distinction | 0/5 | 0/5 | 0/5 | 0/5 |

The task/closure row is a primary failure, not a missing measurement. Every
commutator front is null under the frozen requirement that its barycenter be a
nondegenerate index-zero critical point.

## Task fronts and controls

Depth is measured in residual sublayers: `0→1` is block-0 attention, `1→2` its
MLP, and so on. A null task front means the mature condition did not persist at
all later discrete endpoints. Control entries are random pairing / phase
shuffling.

| Degree | Seed | Shift | causal depth | task Morse front | control fronts |
| --- | ---: | --- | ---: | ---: | --- |
| `k=2` | 7 | composition / extrapolation | 1 / 1 | 2.000 / 0.250 | null, 2.250 / 4.250, 1.250 |
| `k=2` | 17 | composition / extrapolation | 5 / 5 | null / 2.625 | 1.250, 1.250 / 3.250, 1.250 |
| `k=2` | 29 | composition / extrapolation | 1 / 1 | 0.750 / 0.625 | 1.500, 3.750 / 2.250, 1.750 |
| `k=2` | 41 | composition / extrapolation | 3 / 3 | 2.250 / 2.250 | null, null / 5.750, null |
| `k=2` | 53 | composition / extrapolation | 1 / 1 | null / 0.375 | 4.250, 6.000 / 4.250, 5.500 |
| `k=3` | 7 | composition / extrapolation | 3 / 3 | 2.875 / 2.250 | 1.000, 1.250 / 2.250, 0.750 |
| `k=3` | 17 | composition / extrapolation | 4 / 1 | 1.750 / 1.625 | 0.750, 0.750 / 1.000, 0.750 |
| `k=3` | 29 | composition / extrapolation | 3 / 3 | 4.375 / 4.375 | null, null / null, null |
| `k=3` | 41 | composition / extrapolation | 2 / 2 | 4.125 / 1.750 | 0.500, 0.500 / 0.500, 2.250 |
| `k=3` | 53 | composition / extrapolation | null / 3 | 3.000 / 3.000 | null, 3.500 / null, null |

Seed 53 has no composition causal front at `k=3`: its frozen deck campaign did
not classify exact orbit averaging as preserved at any inspected cut. That is
correctly treated as a primary failure rather than assigned a synthetic depth.

## Mechanistic measurements

The finite grids recorded 40 task-landscape events across the ten `k=2`
shift-cells and 141 across the ten `k=3` shift-cells. Near the causal fronts,
`k=2` showed 16 center-index changes, 9 merge-barrier crossings, 9
sheet-component changes, and 25 saddle-census changes. `k=3` showed 7, 8, 8,
and 40 respectively.

This supports a limited claim: causal fronts often occur in locally changing
intervention geometry. It does not support one universal handle attachment.
Degree three is especially implementation-specific, averaging 14.1 detected
events per shift-cell versus 4.0 for degree two, with controls often acquiring
their own early basin.

Seed 7 illustrates both the signal and its limitation. In `k=3` composition,
the vertex components merge between depths `2.75` and `2.875`, just before the
frozen causal depth `3.0`; at `2.875` the barycenter has positive Hessian
eigenvalues approximately `1.088` and `3.264`, excess KL `0.0181`, merge barrier
`0.0382`, and three discrete saddles. Under extrapolation, the corresponding
merge occurs between `2.125` and `2.25`, substantially earlier. Its critical
census also changes repeatedly around the window. The event is therefore not a
single shift-stable critical candidate.

## Post-outcome commutator diagnostic

Ignoring Morse index and asking only when `C(center)<=0.01` persists at later
discrete endpoints gives depths `0.75–0.875` for all `k=2` cells and
`0.50–0.875` for all `k=3` cells. This is descriptive and was computed only
after the primary failure was known.

It says that the remaining local sublayer becomes nearly affine on the exact
orbit mixture early in each sublayer timing family. It does not establish an
autonomous quotient state through subsequent blocks. At alpha `1`, the
remaining-sublayer map is the identity by construction and the commutator is
exactly the degenerate zero function. Future work should define closure by a
small commutator norm over the simplex, not by a Morse minimum at its center.

## Certification decision

The preregistration allowed interval certification only for `k=3`, seed 7, and
only if numerical localization exposed an isolated, stable critical event. It
did not. Composition and extrapolation merge at different depths, the center is
already nondegenerate on both sides, and multiple saddle-census changes occur in
the surrounding windows. Certification was therefore not attempted. Selecting
another seed or certifying only the cleaner shift would violate the frozen rule.

## Interpretation and boundaries

The strongest warranted conclusion is:

> Exact orbit averaging often crosses a changing low-dimensional task-loss
> landscape near its causal front, but the surrounding Morse geometry is too
> seed-, shift-, and pairing-dependent to define a universal quotient-formation
> normal form in these TinyLLMs.

The scan is exhaustive only on its finite interval and triangular lattices. The
discrete critical census does not prove continuum nondegeneracy or connectedness
between grid points. The declared vertex-exact timing homotopy fixes ordinary
sheet outputs while varying only mixture behavior, but it is an intervention
family, not a neural-ODE flow. No claim is made about hidden states outside the
exact deck-orbit simplex.

## Artifacts and reproduction

Primary aggregate:
`data/experiments/tinyllm_morse_cerf/20260806_d6_preregistered/campaign_results.json`

Per-cell records and compressed landscapes:
`data/experiments/tinyllm_morse_cerf/20260806_d6_preregistered/runs/k{2,3}/seed_<seed>/`

Disposable systems shakedown:
`data/experiments/tinyllm_morse_cerf/shakedown_20260806/`

The full DVC data root is committed at
`lakefs://artifacts/main/structure-net/` in lakeFS commit
`22a5d0501b20ba9a78431f0313d915957b598a3673567f088155d9096ab1d96d`.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
pixi run python -m experiments.structure_net.tinyllm_morse_cerf \
  --output data/experiments/tinyllm_morse_cerf/20260806_d6_preregistered \
  --degrees 2,3 --device cuda:0
```
