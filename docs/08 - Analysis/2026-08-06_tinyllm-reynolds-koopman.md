# TinyLLM Reynolds–Koopman quotient-closure scan

**Status:** NOT CONFIRMED — PREDICTIVE BARYCENTER FRONT PRECEDES CAUSAL QUOTIENT FRONT  
**Date:** 2026-08-06  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`  
**Hypothesis:** `tinyllm-reynolds-koopman-quotient-closure-v1`  
**Preregistration:** `../07 - Status Reports/2026-08-06_tinyllm-reynolds-koopman-preregistration.md`

## Verdict

The Reynolds barycenter develops a sharp, reproducible predictive closure
transition, but that transition is not the causal quotient front. The full
hypothesis is not confirmed.

Low-rank cover invariants added substantial final-task prediction before block-0
attention and negligible gain afterward in all five seeds for both `k=2` and
`k=3`. Nevertheless, the barycenter-only task front matched the frozen exact-
orbit causal front within one cut in only 3/5 `k=2` seeds and 1/5 `k=3` seeds.
For many models a linear observer could read the final task from an early
barycenter that the frozen transformer could not itself use after causal orbit
averaging. This is decoder-relative predictive sufficiency, not autonomous
quotient dynamics.

The stricter checks reinforce that conclusion. Autonomous one-step closure near
the declared task front passed in 1/5 `k=2` seeds, and unseen cover-scaling
response prediction passed in only 1/5 seeds for each degree. `k=3` did pass the
autonomous gate in 5/5 because its task front was uniformly post-attention, but
its front still preceded the causal front in four seeds. The degree-three front
was not later than degree two: both median joint indices were block-0
post-attention.

## Campaign integrity

All ten frozen d6 degree-ladder checkpoints (`k=2,3`; seeds
`7,17,29,41,53`) completed without retraining. Each cell validated the source
model digest, checkpoint hash, prior causal-comparator result, and zero-spread
baseline replay recorded by that comparator. The initial full launch encountered
host BLAS oversubscription; restricting BLAS to one thread changed no model,
cohort, dictionary, gate, or result schema, and the append-only runner resumed
completed cells by implementation digest.

| Item | Value |
| --- | --- |
| requested / completed / failed | 10 / 10 / 0 |
| fit / held-out / response orbits | 384 / 192 / 96 |
| fixed-map members | 192 |
| barycenter PCA / character sketch rank | 48 / 24 |
| ridge | 0.001 |
| device | NVIDIA GeForce RTX 2060 SUPER, CUDA |
| PyTorch / Python | 2.5.1+cu121 / 3.11.13 |
| implementation SHA-256 | `1f15f9f5de15561fde815baa855de86076dd561d7a41b8fd8814ab5e93aac40c` |
| campaign SHA-256 | `2c2dbe77327df6d0cecc47e5a2c8684a1d32bcad5d2d8b4a6fbb03a4dfeda514` |

## Primary gates

Counts require simultaneous composition and extrapolation success in one seed.
The preregistration required at least four of five seeds for each per-degree gate.

| Gate | `k=2` | `k=3` | Required |
| --- | ---: | ---: | ---: |
| task-front agreement within one cut | 3/5 | 1/5 | 4/5 |
| substantial-to-negligible cover-gain transition | 5/5 | 5/5 | 4/5 |
| autonomous one-step closure near task front | 1/5 | 5/5 | 4/5 |
| unseen-`lambda` response prediction | 1/5 | 1/5 | 4/5 |
| median `k=3` front no earlier than `k=2` | block-0 post-attention | block-0 post-attention | pass, equality |

Passing degree ordering by equality does not support the proposed later
degree-three mechanism. Because several required gates fail, no secondary result
can promote the full hypothesis.

## Predictive and causal fronts

The entries show composition / extrapolation. `b0 pre`, `b0 attn`, `b0 MLP`,
`b1 attn`, `b1 MLP`, `b2 attn`, and `full` abbreviate the corresponding residual
cuts.

| Degree | Seed | frozen causal front | barycenter task-closure front | within one cut on both shifts |
| --- | ---: | --- | --- | --- |
| `k=2` | 7 | b0 attn / b0 attn | b0 attn / b0 pre | yes |
| `k=2` | 17 | b2 attn / b2 attn | b0 attn / b0 attn | no |
| `k=2` | 29 | b0 attn / b0 attn | b0 attn / b0 pre | yes |
| `k=2` | 41 | b1 attn / b1 attn | b0 attn / b0 pre | no |
| `k=2` | 53 | b0 attn / b0 attn | b0 pre / b0 pre | yes |
| `k=3` | 7 | b1 attn / b1 attn | b0 attn / b0 attn | no |
| `k=3` | 17 | b1 MLP / b0 attn | b0 attn / b0 attn | no |
| `k=3` | 29 | b1 attn / b1 attn | b0 attn / b0 attn | no |
| `k=3` | 41 | b0 MLP / b0 MLP | b0 attn / b0 attn | yes |
| `k=3` | 53 | full / b1 attn | b0 attn / b0 attn | no |

At the analytic frontend, barycenter-only final-moment `R2` was negative on
average, while the full character dictionary reached `0.9989/0.9943` for `k=2`
and `0.9985/0.9941` for `k=3` on composition/extrapolation. This is the clean
cover-dependent regime.

At block-0 pre-attention, barycenter-only `R2` had already risen to
`0.9819/0.9804` for `k=2` and `0.9651/0.9565` for `k=3`. Positive cover gain was
still `0.0174/0.0177` and `0.0336/0.0379`, respectively. After block-0 attention,
barycenter-only `R2` was at least `0.9975` in every degree/regime mean and cover
gain was at most `0.0001`.

That early linear readability is not enough to identify computation available to
the frozen downstream network. The causal intervention is the discriminating
evidence: several networks still required the cover for one or two later blocks.

## Autonomous one-step closure

The held-out next-barycenter model makes the mismatch explicit.

| Transition | degree | barycenter `R2`, composition / extrapolation | cover gain, composition / extrapolation |
| --- | --- | ---: | ---: |
| b0 pre → b0 post-attention | `k=2` | 0.9553 / 0.9544 | 0.0431 / 0.0427 |
| b0 pre → b0 post-attention | `k=3` | 0.9024 / 0.8624 | 0.0734 / 0.0850 |
| b0 post-attention → b0 post-MLP | `k=2` | 1.0000 / 1.0000 | 0.0000 / 0.0000 |
| b0 post-attention → b0 post-MLP | `k=3` | 1.0000 / 1.0000 | 0.0000 / 0.0000 |

All later transition means were at least `0.9995` with cover gain at most
`0.0001`. Thus block-0 attention marks a strong observable one-step closure
transition. For `k=2`, however, four extrapolation task fronts were selected at
pre-attention; those fronts failed the autonomous gate even though an external
linear decoder already predicted the final task. For `k=3`, all task fronts were
post-attention, so all five passed the nearby one-step gate.

## Controls

The cover lift is predictive but not uniquely Koopman-specific. At the frontend,
a random same-size nonlinear cover dictionary achieved mean moment `R2` of
`0.9999/0.9989` for `k=2` and `0.9991/0.9946` for `k=3`, matching or slightly
exceeding the structured dictionary. Shuffling character rows or orbit membership
destroyed frontend prediction, confirming that correct fiber association matters.

At block-0 post-attention, barycenter ridge, random dictionary, random-Fourier
kernel ridge, phase-shuffled lift, and the barycenter MLP were all near saturation.
The proposed lift therefore offers a symmetry-readable decomposition of the
cover-gain transition, but not superior predictive performance over generic
nonlinear features. This fails the attachment's stated usefulness criterion for
a uniquely stable Koopman account.

## Cover-scaling interventions

Response models were trained on `lambda={0,0.5,1}` and evaluated on unseen
`{0.25,0.75,1.25}`. At each seed's declared task front, the mean results were:

| Degree | regime | `lambda` | moment `R2` | circular cosine |
| --- | --- | ---: | ---: | ---: |
| `k=2` | composition | 0.25 / 0.75 / 1.25 | 0.858 / 0.991 / 0.987 | 0.932 / 0.996 / 0.994 |
| `k=2` | extrapolation | 0.25 / 0.75 / 1.25 | 0.420 / 0.942 / 0.963 | 0.863 / 0.972 / 0.983 |
| `k=3` | composition | 0.25 / 0.75 / 1.25 | 0.681 / 0.966 / 0.198 | 0.879 / 0.984 / 0.615 |
| `k=3` | extrapolation | 0.25 / 0.75 / 1.25 | 0.634 / 0.929 / 0.072 | 0.863 / 0.962 / 0.556 |

The model interpolates well near `lambda=0.75` but is not a stable intervention
model across contraction and extrapolating amplification. The poor `k=3,
lambda=1.25` result is especially inconsistent with a closed low-order lifted
operator.

## Harmonic-synthesis diagnostic

The fixed-map complex harmonic response supports quadratic dominance for `k=2`
only after block-0 attention: it passed in 4/5 seeds by block-0 post-MLP and 5/5
from block-1 attention onward on both shifts. The predicted cubic dominance for
`k=3` never passed in four seeds at any cut; the quadratic coefficient was most
often the largest nonconstant term. The proposed quadratic-versus-cubic account
of the degree-dependent causal delay is not supported.

## Interpretation and boundaries

The supported result is an exact-symmetry observable decomposition:

`cover-dependent prediction → barycenter-readable prediction`

occurs sharply around block-0 attention. It does not follow that the transformer's
remaining computation closes on that barycenter. Predictive access by a newly fit
decoder, autonomous next-state closure, causal sufficiency under intervention,
and response prediction are distinct; this campaign measured all four and found
them non-equivalent.

The study supplies a time-varying finite dictionary, not a stationary Koopman
operator. It does not claim eigenfunction status for the barycenter, exact
finite-dimensional invariance, maximal invariant-subspace recovery, cross-seed
hidden-coordinate transfer, or behavior outside the declared synthetic shifts.

## Artifacts and reproduction

| Artifact | Path |
| --- | --- |
| campaign result | `data/experiments/tinyllm_reynolds_koopman/20260806_d6_preregistered/campaign_results.json` |
| per-seed result | `data/experiments/tinyllm_reynolds_koopman/20260806_d6_preregistered/runs/k*/seed_*/result.json` |
| frozen dictionaries and fitted operators | `data/experiments/tinyllm_reynolds_koopman/20260806_d6_preregistered/runs/k*/seed_*/koopman_models.npz` |
| causal comparator | `data/experiments/tinyllm_deck_action_descrambler/20260806_d6_preregistered` |
| frozen checkpoints | `data/experiments/tinyllm_degree_k_ladder/20260806_d6_preregistered` |

```bash
MPLCONFIGDIR=/tmp/matplotlib-structure-net \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
pixi run python -m experiments.structure_net.tinyllm_reynolds_koopman \
  --device cuda:0 \
  --output data/experiments/tinyllm_reynolds_koopman/20260806_d6_preregistered
```
