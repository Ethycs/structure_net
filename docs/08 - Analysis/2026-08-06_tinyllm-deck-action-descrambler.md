# TinyLLM deck-action descrambler and causal quotient front

**Status:** NOT CONFIRMED — CAUSAL QUOTIENT FRONT WITHOUT A CLEAN LINEAR ISOTYPIC SPLIT  
**Date:** 2026-08-06  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`  
**Hypothesis:** `tinyllm-deck-action-carrier-cover-v1`  
**Preregistration:** `../07 - Status Reports/2026-08-06_tinyllm-deck-action-descrambler-preregistration.md`

## Verdict

No residual cut passed the complete deck-action gate in four of five seeds for
either `k=2` or `k=3`. In particular, held-out Procrustes transport failed at
every residual cut, and the nominal invariant component remained conditionally
branch-decodable. The preregistered linear deck-action/isotypic-decomposition
hypothesis is therefore not confirmed.

The causal intervention produced a separate, reproducible result. Exact
nuisance-matched orbit averaging destroyed the task map before attention in all
five `k=2` and all five `k=3` models, but preserved it at full depth in all ten
models while forcing patched branch accuracy to chance. The transition occurred
earlier for `k=2` than for `k=3`. This localizes a **causal quotient front**:
early computation requires the finite cover, whereas the mature network carries
a quotient-sufficient representation alongside redundant branch information.

## Campaign integrity

The study reused all 15 frozen d6 degree-ladder checkpoints (`k=1,2,3`; seeds
`7,17,29,41,53`) without retraining. Each result records and validates its source
checkpoint SHA-256 and the analysis implementation SHA-256. Baseline continuation
from every captured cut reproduced the frozen output with zero cross-cut spread.

| Item | Value |
| --- | --- |
| requested / completed / failed | 15 / 15 / 0 |
| analysis device | NVIDIA GeForce RTX 2060 SUPER, CUDA |
| PyTorch / Python | 2.5.1+cu121 / 3.11.13 |
| fit / evaluation orbits | 512 / 256 |
| map points | 192 |
| analyzed transformer blocks | 3 |
| analysis implementation SHA-256 | `9a15ceaf735555cdd0aa843247e7f142e7585632ecd98f2a2c73b60a341c3a48` |
| campaign result SHA-256 | `a3c14ce7022b7301344beaca876e0d454445c972a57de69c9cd4cd89098036b3` |

## Preregistered deck gates

Counts below require a seed to pass the named gate on both composition and
extrapolation. A stable cut required at least four of five seeds to pass all four
geometry gates; none did.

| Degree and cut | transport | closure | branch localization | invariant branch control | complete gate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `k=2` frontend | 2/5 | 5/5 | 5/5 | 3/5 | 2/5 |
| `k=2` block-0 post-attention | 0/5 | 5/5 | 5/5 | 0/5 | 0/5 |
| `k=2` block-1 post-attention | 0/5 | 5/5 | 5/5 | 0/5 | 0/5 |
| `k=2` block-2 post-attention | 0/5 | 5/5 | 5/5 | 0/5 | 0/5 |
| `k=2` full | 0/5 | 5/5 | 5/5 | 0/5 | 0/5 |
| `k=3` frontend | 3/5 | 5/5 | 5/5 | 0/5 | 0/5 |
| `k=3` block-0 post-attention | 0/5 | 0/5 | 5/5 | 0/5 | 0/5 |
| `k=3` block-1 post-attention | 0/5 | 0/5 | 5/5 | 0/5 | 0/5 |
| `k=3` block-2 post-attention | 0/5 | 0/5 | 5/5 | 0/5 | 0/5 |
| `k=3` full | 0/5 | 3/5 | 5/5 | 0/5 | 0/5 |

The branch component always matched the full activation's strong branch probe,
but this was not an exclusive split. At full depth, mean conditional branch
accuracy for `k=2` was `98.59%` on the full representation, `98.28%` on the
nontrivial component, and `78.83%` on the nominal invariant component
(composition). For `k=3`, the corresponding values were `98.15%`, `98.07%`, and
`98.15%`. The extrapolation values were respectively `97.97%`/`98.16%`/`76.99%`
and `96.93%`/`96.95%`/`96.88%`.

## Causal quotient front

Each count requires the same exact-orbit classification on both held-out shifts.
Four matching seeds constitute a reproducible classification. Blank remainder is
partial, unresolved, or shift-inconsistent.

| Cut | `k=2` exact orbit average | `k=3` exact orbit average |
| --- | --- | --- |
| frontend | destroyed 5/5 | destroyed 5/5 |
| block-0 pre-attention | destroyed 5/5 | destroyed 5/5 |
| block-0 post-attention | preserved 3/5; destroyed 1/5 | destroyed 4/5 |
| block-0 post-MLP | preserved 3/5 | preserved 1/5; destroyed 3/5 |
| block-1 post-attention | preserved 4/5 | preserved 3/5 |
| block-1 post-MLP | preserved 4/5 | preserved 3/5 |
| block-2 post-attention | preserved 5/5 | preserved 4/5 |
| block-2 post-MLP | preserved 5/5 | preserved 4/5 |
| full depth | preserved 5/5 | preserved 5/5 |

At full depth, exact orbit averaging improved mean exact-bin accuracy while
retaining the target winding and high circular alignment:

| Degree | regime | original accuracy / alignment | orbit-average accuracy / alignment | patched branch accuracy |
| --- | --- | ---: | ---: | ---: |
| `k=2` | composition | 0.7961 / 0.9956 | 0.8562 / 0.9982 | 0.5000 |
| `k=2` | extrapolation | 0.6648 / 0.9868 | 0.8219 / 0.9968 | 0.5000 |
| `k=3` | composition | 0.6352 / 0.9878 | 0.7625 / 0.9957 | 0.3333 |
| `k=3` | extrapolation | 0.5154 / 0.9677 | 0.7531 / 0.9941 | 0.3333 |

The orbit-averaged mature maps retained posterior normalized Fisher `H1`
lifetimes of `0.7864/0.7874` for `k=2` and `0.6822/0.7358` for `k=3` on
composition/extrapolation.

## Intervention controls

The mature-map preservation is specific to the exact task fibers. Mean full-depth
exact-bin accuracies are shown below.

| Degree and regime | exact orbit | random same-rank projection | random orbit pairing | phase-shuffled average | equal-norm ablation |
| --- | ---: | ---: | ---: | ---: | ---: |
| `k=2` composition | 0.8562 | 0.5973 | 0.1703 | 0.1629 | 0.7762 |
| `k=2` extrapolation | 0.8219 | 0.5297 | 0.1688 | 0.1688 | 0.6504 |
| `k=3` composition | 0.7625 | 0.5057 | 0.1799 | 0.1898 | 0.6096 |
| `k=3` extrapolation | 0.7531 | 0.4294 | 0.1677 | 0.1753 | 0.4932 |

Phase-shuffled averaging can retain high circular alignment while destroying
absolute task-bin accuracy; alignment alone is therefore not a sufficient causal
endpoint. Random orbit pairing also destroys the mature map, ruling out generic
activation averaging as the explanation.

## Cross-seed atlas

Real-Schur root multiplicities were consistent at residual width 384: `k=2`
produced approximately 192 invariant and 192 nontrivial dimensions, while `k=3`
produced approximately 128 invariant and 256 nontrivial dimensions. At full
depth their five-seed means were `192.8/191.2` and `127.8/256.2` respectively.

Energy moved strongly toward the nominal invariant component with depth. At full
depth its mean composition/extrapolation fraction was `0.9934/0.9935` for `k=2`
and `0.9795/0.9797` for `k=3`. This numerical regularity does not overcome the
failed transport and invariant-control gates: repeated Schur blocks are not
cross-seed neuron alignments, and the approximate group-average projector does
not produce a clean information partition.

## Interpretation and boundaries

The supported claim is causal and depth-localized, not a successful linear
descrambling theorem. Before attention, removing sheet identity by exact orbit
averaging removes information needed for the task. Later, the same averaging
removes branch information without damaging the degree-`k` output. The quotient
front moves deeper as harmonic degree increases.

The study tests one orthogonal Procrustes action, the finite average of that
approximate action, one conditioned linear ridge-probe family, and synthetic exact
orbits. Orthogonal descrambling never erases information. The projector need not
be exactly idempotent, nonlinear deck actions remain untested, and coordinates
inside repeated isotypic blocks are non-identifiable across seeds.

## Artifacts and reproduction

| Artifact | Path |
| --- | --- |
| campaign result | `data/experiments/tinyllm_deck_action_descrambler/20260806_d6_preregistered/campaign_results.json` |
| per-seed results | `data/experiments/tinyllm_deck_action_descrambler/20260806_d6_preregistered/runs/k*/seed_*/result.json` |
| fitted action/projector arrays | `data/experiments/tinyllm_deck_action_descrambler/20260806_d6_preregistered/runs/k*/seed_*/*.npz` |
| source checkpoints | `data/experiments/tinyllm_degree_k_ladder/20260806_d6_preregistered` |

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
pixi run python -m experiments.structure_net.tinyllm_deck_action_descrambler \
  --device cuda:0 \
  --output data/experiments/tinyllm_deck_action_descrambler/20260806_d6_preregistered
```
