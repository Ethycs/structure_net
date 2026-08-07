# TinyLLM Reynolds character-coupling causal decomposition

**Status:** PARTIALLY SUPPORTED — EXACT SYNTHESIS LOCALIZES `k=2`; QUADRATIC SUFFICIENCY DOES NOT REPLICATE  
**Date:** 2026-08-06  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`  
**Hypothesis:** `tinyllm-reynolds-character-coupling-synthesis-v1`  
**Preregistration:** `../07 - Status Reports/2026-08-06_tinyllm-reynolds-character-coupling-preregistration.md`

## Verdict

The exact Reynolds/Jensen defect identifies a strong causal mechanism for
degree two: in all five `k=2` seeds and both shifts, the first sublayer whose
actual next barycenter passes while its propagated barycenter fails occurs at
exactly the frozen causal quotient front. The complete twelve-sublayer causal
classification is also identical between composition and extrapolation in all
five seeds. Neither control reproduces the mechanism.

The full hypothesis is nevertheless not confirmed. The neutral quadratic term
explains at least 70% of the downstream Fisher effect in only 3/5 `k=2` seeds.
It succeeds strongly in seeds 7, 29, and 53, whose quotient front is block-0
attention, but not in the later-front seeds 17 and 41. For `k=3`, causal regimes
are shift-stable in only 2/5 seeds, localization passes 3/5, and quadratic
sufficiency passes 0/5.

The experiment therefore supports the causal synthesis statement more strongly
than a universal low-order Taylor statement:

> In degree-two TinyLLMs, the causal quotient front is the sublayer where
> branch-bearing cover states first synthesize a quotient-sufficient next
> barycenter. Which local Taylor order captures that synthesis depends on the
> model and front depth.

## Campaign integrity

All ten retained d6 checkpoints completed without retraining. Each cell used 64
new exact nuisance-matched orbits under composition and extrapolation and
scanned every attention and MLP residual sublayer in blocks 0--5. Source model,
checkpoint, deck-action comparator, and Reynolds–Koopman comparator identities
were validated before analysis.

The derivative scale, task-effect floor, quadratic threshold, controls, and
four-regime decision rule were frozen before the confirmatory run. A disposable
eight-orbit CUDA lifecycle completed first and was not pooled with evidence.

| Item | Value |
| --- | --- |
| requested / completed / failed | 10 / 10 / 0 |
| exact orbits per shift and cell | 64 |
| residual sublayers | 12: attention and MLP in blocks 0–5 |
| finite-difference scale | `eta=0.25` |
| quadratic Fisher-effect gate | `>=0.70` on both shifts |
| task-effect degeneracy floor | `1e-6` |
| controls | shuffled membership; norm-matched random directions |
| device | NVIDIA GeForce RTX 2060 SUPER, CUDA |
| PyTorch / Python | 2.5.1+cu121 / 3.11.13 |
| implementation SHA-256 | `eeb46a9e1748e32d1316e764b4bccea14b1b647addfe39097d89d39fc951ae0b` |
| campaign SHA-256 | `a7ccda0d8a36a5c96de96045a32400deaf3cdbdb0856d969164df8d6a455495b` |
| DVC data root | `cb6ebfe2d688de4490faf0882211ef3d.dir` |
| lakeFS commit | `483bf0f3f89eaca522333aa0d8bdbb1f871f0996610e1a885b3d62e2951ad02b` |

## Preregistered gates

Each gate requires at least four of five seeds separately by degree.

| Gate | `k=2` | `k=3` | Required |
| --- | ---: | ---: | ---: |
| shift-stable twelve-sublayer causal regime | **5/5** | 2/5 | 4/5 |
| synthesis front within one sublayer of causal front | **5/5** | 3/5 | 4/5 |
| neutral quadratic explains `>=0.70` Fisher effect on both shifts | 3/5 | 0/5 | 4/5 |
| neither control reproduces synthesis | **5/5** | **5/5** | 4/5 |

Passing secondary gates cannot promote the full result because quadratic
sufficiency fails for both degrees and the `k=3` causal gates also fail.

## Exact causal synthesis fronts

Depth counts residual sublayers from one. The entries are composition /
extrapolation.

| Degree | Seed | frozen causal depth | measured synthesis depth | distance |
| --- | ---: | --- | --- | --- |
| `k=2` | 7 | 1 / 1 | 1 / 1 | 0 / 0 |
| `k=2` | 17 | 5 / 5 | 5 / 5 | 0 / 0 |
| `k=2` | 29 | 1 / 1 | 1 / 1 | 0 / 0 |
| `k=2` | 41 | 3 / 3 | 3 / 3 | 0 / 0 |
| `k=2` | 53 | 1 / 1 | 1 / 1 | 0 / 0 |
| `k=3` | 7 | 3 / 3 | 7 / 3 | 4 / 0 |
| `k=3` | 17 | 4 / 1 | 3 / 1 | 1 / 0 |
| `k=3` | 29 | 3 / 3 | 3 / 3 | 0 / 0 |
| `k=3` | 41 | 2 / 2 | 2 / 2 | 0 / 0 |
| `k=3` | 53 | 12 / 3 | 3 / 3 | 9 / 0 |

For every `k=2` cell, all sublayers before the front are
`cover_required_after_sublayer`, the front is `invariant_synthesis`, and every
later sublayer is `quotient_already_closed`. This sequence is a substantially
cleaner causal signature than the surrounding Morse landscapes.

Degree three is different. Several models become sufficient, lose sufficiency,
and synthesize it again. Seed 29 has a second synthesis event after a closed
sublayer under both shifts; seed 41 does so only under extrapolation. Seeds 7,
17, and 53 disagree across shifts in front location or subsequent regime
sequence. The mechanism is therefore support-relative for degree three.

## Neutral quadratic contribution

Downstream Fisher-effect explained fractions at the first synthesis event are:

| Degree | Seed | composition | extrapolation | joint gate |
| --- | ---: | ---: | ---: | --- |
| `k=2` | 7 | 0.984 | 0.985 | pass |
| `k=2` | 17 | 0.364 | -0.384 | fail |
| `k=2` | 29 | 0.956 | 0.949 | pass |
| `k=2` | 41 | 0.377 | 0.326 | fail |
| `k=2` | 53 | 0.963 | 0.973 | pass |
| `k=3` | 7 | -23.709 | -13.995 | fail |
| `k=3` | 17 | -17.185 | 0.151 | fail |
| `k=3` | 29 | -0.370 | -0.248 | fail |
| `k=3` | 41 | -6.411 | -4.568 | fail |
| `k=3` | 53 | -0.417 | -0.198 | fail |

Negative values are untruncated: in those cells, inserting the quadratic
approximation moves the downstream posterior farther from the actual next-
barycenter posterior than inserting no defect at all.

The approximation is task-directed rather than a faithful residual
reconstruction. Median quadratic downstream effect explained is `0.953` for
`k=2`, while median residual explained fraction is `-4.124`. Thus the successful
degree-two quadratic term lands in a downstream-equivalent task direction even
while missing most of the high-dimensional exact residual defect. This is
consistent with the program's repeated distinction between readable,
task-effective, and state-space-exact representations.

## Cubic diagnostic and character selection

Adding the declared cubic stencil does not materially change `k=2`. For `k=3`
it improves some cells but the median Fisher-effect explained fraction remains
`-2.091`; no cell reaches the primary quadratic threshold, and the cubic term
does not provide a stable rescue.

The character measurements confirm the theoretical correction to the earlier
degree-based prediction. Real `k=3` carriers contain equal conjugate `r=1` and
`r=2` energies at every synthesis front. Each mode carries approximately
`0.10–0.29` of total Fourier energy, so the neutral quadratic coupling
`1+2=0 mod 3` is both allowed and populated. Its empirical failure is therefore
not caused by an absent character. It means a local quadratic expansion at the
barycenter does not approximate the finite cover-to-barycenter defect over the
observed radius.

## Controls

Shuffled membership is `cover_required_after_sublayer` in all 120 evaluated
degree-two and all 120 degree-three shift/sublayer cells. It never creates a
task-valid actual barycenter.

Matched random directions are usually `quotient_already_closed` because their
generic zero-mean variations do not reproduce the learned cover computation.
They produce four isolated `k=3` synthesis labels, but none occurs near the
exact front with at least 70% quadratic Fisher-effect explanation. Control
specificity therefore passes 5/5 for both degrees.

This matters: the exact `k=2` localization is not merely generic Jensen
curvature. It depends on the trained orbit-sheet organization.

## Interpretation and boundaries

The strongest result is causal rather than perturbative. The identity

`chi_l = mean_j F_l(h_j) - F_l(mean_j h_j)`

isolates exactly what the sublayer manufactures from cover variation. For
degree two, adding this exact defect is precisely what changes the frozen
continuation from failing to passing at the previously measured quotient front.

The finite-difference expansion is local at `eta=0.25`, while the causal defect
uses full observed sheet displacements. Poor residual approximation—especially
for `k=3`—does not disprove character selection; it rejects a universal
second/third-order truncation at this radius. No global Hessian or interval
certificate was constructed.

Within-orbit branch chance is exact because every patched barycenter is repeated
across all fiber members. It does not claim that arbitrary off-fiber states lack
branch information.

## Artifacts and reproduction

Primary aggregate:
`data/experiments/tinyllm_reynolds_character_coupling/20260806_d6_preregistered/campaign_results.json`

Per-cell records:
`data/experiments/tinyllm_reynolds_character_coupling/20260806_d6_preregistered/runs/k{2,3}/seed_<seed>/result.json`

Disposable shakedown:
`data/experiments/tinyllm_reynolds_character_coupling/shakedown_20260806/`

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
pixi run python -m experiments.structure_net.tinyllm_reynolds_character_coupling \
  --output data/experiments/tinyllm_reynolds_character_coupling/20260806_d6_preregistered \
  --device cuda:0
```
