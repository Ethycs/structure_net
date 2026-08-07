# TinyLLM invariant front-end causal test

**Status:** FAILED FULL PREREGISTERED GATE — compositional quotient in four
learned seeds, no extrapolating cosine base

**Date:** 2026-08-06

**Hypothesis:** `tinyllm-invariant-frontend-stable-cosine-quotient-v1`

**Conformance:** `PREREGISTERED`; the design and exact N3 gauge limitation are
recorded in the
[preregistration](../07%20-%20Status%20Reports/2026-08-06_tinyllm-invariant-frontend-causal-preregistration.md).

## Verdict

Architectural nuisance control did **not** make the requested absolute-cosine
quotient stable under extrapolation. Neither the analytic canonicalizer nor the
learned equivariant encoder had a seed pass the full four-cut, two-shift
conjunction; four of five were required.

The learned encoder nevertheless produced a sharp partial result. Four of five
seeds passed the joint `(cosine >= 0.90, branch <= 0.55)` endpoint at every cut
on held-out composition. At full depth their mean composition values were
`(0.757, 0.492)`, depressed by an optimization collapse in seed 17; the other
four full-depth cosine correlations ranged from 0.938 to 0.958. Under
outside-range extrapolation, all five failed cosine at every cut. Full-depth
mean cosine fell to 0.373 while branch accuracy remained at chance, 0.494.

This is the preregistered “composition succeeds but extrapolation fails” branch:
the learned architecture constructs a support-relative quotient, not a stable
absolute-cosine quotient. The exact observation gauge makes the stronger target
unidentifiable in principle. The intervention cleanly contracts the tested
branch, but outside training support it contracts semantic base and fiber
together.

## Campaign integrity

The campaign completed 15/15 cells with no failures: five retained raw N3
controls and ten newly trained structured models. Every cell used implementation
SHA-256
`8784c19836f9175c03a11da112f399160435dfbe6722cdcbdd9b30dfe9dc8317`.
Within each seed, all three arms record identical training-data and minibatch
schedule hashes.

Six focused contract tests and seven predecessor regression tests passed before
launch. A three-cell CUDA shakedown completed first. The confirmatory run used
two workers on PyTorch logical GPU 1, the RTX 2060 SUPER. Peak allocated memory
was 0.547 GiB for raw analysis and 1.024 GiB for each structured worker.

## Fixed intervention

- d8 TinyLLM: 50,964,992 transformer parameters;
- seeds 7, 17, 29, 41, and 53;
- N3 support, 4,096 paired examples, batch size 64, 600 updates;
- AdamW at `3e-4`, weight decay `0.01`, gradient clip `1.0`;
- task cross-entropy only—no residual, contrastive, adversarial, or equivariance
  penalty;
- raw control: retained ordinary N3 checkpoints;
- analytic arm: discard the harmonic channel, fit drift plus a planar sinusoid
  on a fixed speed grid, normalize amplitude, infer direction, and advance to
  the prediction horizon;
- learned arm: fixed constant/affine-trend annihilation, scale normalization,
  shared planar temporal channels, invariant Gram-conditioned scalar mixing,
  and a laboratory-frame coordinate projection;
- 1,024-example exact-fiber tests and fresh width-128 nonlinear conditional
  probes trained for at most 240 steps.

The analytic arm adds 1,024 trainable scalar-embedding parameters after its
fixed estimator. The learned arm adds 9,904 trainable front-end and embedding
parameters. Both use the same transformer size and task head.

## Primary endpoints

Each entry is the five-seed mean `(cosine correlation, conditional branch
balanced accuracy)` followed by the number of seeds passing both gates.

| Arm and cut | Composition | Extrapolation |
| --- | --- | --- |
| Raw frontend | `(0.961, 0.998)`, 0/5 | `(0.571, 0.886)`, 0/5 |
| Raw post-attention | `(0.974, 0.769)`, 0/5 | `(0.461, 0.592)`, 0/5 |
| Raw post-MLP | `(0.976, 0.558)`, 2/5 | `(0.459, 0.516)`, 0/5 |
| Raw full | `(0.971, 0.603)`, 1/5 | `(0.452, 0.522)`, 0/5 |
| Analytic frontend | `(0.602, 0.506)`, 0/5 | `(0.366, 0.468)`, 0/5 |
| Analytic post-attention | `(0.645, 0.504)`, 0/5 | `(0.360, 0.465)`, 0/5 |
| Analytic post-MLP | `(0.646, 0.494)`, 0/5 | `(0.360, 0.469)`, 0/5 |
| Analytic full | `(0.647, 0.488)`, 0/5 | `(0.359, 0.480)`, 0/5 |
| Learned frontend | `(0.726, 0.488)`, 4/5 | `(0.350, 0.507)`, 0/5 |
| Learned post-attention | `(0.764, 0.498)`, 4/5 | `(0.367, 0.488)`, 0/5 |
| Learned post-MLP | `(0.754, 0.500)`, 4/5 | `(0.365, 0.495)`, 0/5 |
| Learned full | `(0.757, 0.492)`, 4/5 | `(0.373, 0.494)`, 0/5 |

The learned composition means obscure the seed structure. Seed 17 collapsed;
its full-depth composition correlation was `-0.012`. Seeds 7, 29, 41, and 53
were respectively `0.958`, `0.938`, `0.950`, and `0.951`, with branch
accuracies between 0.486 and 0.504. The same four seeds therefore pass all four
composition cuts, satisfying that half of the seedwise design exactly.

Conditional log-loss gain agrees with the accuracy result. It was approximately
zero throughout both structured arms; on learned full depth it averaged
`-0.001` for composition and `-0.001` for extrapolation. Tested branch removal
was real relative to a fresh probe, rather than an online-adversary artifact.

## Positive-control failure

The analytic canonicalizer erased branch information but failed the base even
in distribution: frontend mean cosine was 0.654 in distribution, 0.602 on
composition, and 0.366 on extrapolation. Transformer depth did not repair it.
This does not show that analytic invariants are generally ineffective. It shows
that this observation-only sinusoid estimator is not an oracle under eight
quantized, noisy samples—and, more fundamentally, that the current absolute
target is not constant on the exact observation gauge.

For harmonic order `k`, the transformation

```text
phase' = phase + alpha
orientation' = orientation - alpha
harmonic_phase' = harmonic_phase - k * alpha
```

leaves every observed sensor value unchanged while changing absolute cosine.
No deterministic observation-only canonicalizer can recover both targets for
such an identical input. The analytic failure therefore activates the
preregistered diagnosis: either add an observed gauge reference, or redefine
the target on an identifiable quotient.

## Task behavior

| Arm | In-distribution accuracy | Composition accuracy | Extrapolation accuracy |
| --- | ---: | ---: | ---: |
| Raw | 0.491 | 0.390 | 0.131 |
| Analytic | 0.112 | 0.116 | 0.093 |
| Learned | 0.304 | 0.296 | 0.126 |

The analytic bottleneck materially damaged the task. The learned bottleneck
retained useful compositional task information but remained below raw, and its
extrapolation accuracy was essentially tied with raw while cosine geometry
remained far below the representation gate. Task performance does not rescue
the failed joint endpoint.

## What the causal comparison establishes

1. The raw transformer preserves compositional cosine but carries a strongly
   decodable phase branch at its input and still leaks it at depth.
2. A structurally equivariant scalar bottleneck removes tested branch
   information at the front end and throughout the transformer.
3. Four independent learned seeds simultaneously retain cosine and erase branch
   on composition at every declared cut.
4. The same architecture erases branch but loses cosine under extrapolation in
   every seed. Architectural symmetry is therefore sufficient for a
   support-relative quotient here, not for the requested absolute-cosine map.
5. Because the generator admits identical observations with different targets,
   further loss tuning cannot solve the full stated problem.

## Next experiment

Repair identifiability before comparing more representation losses. The
cleanest causal follow-up adds an observed calibration reference that fixes the
planar orientation gauge, while leaving phase and target hidden. Repeat the raw,
analytic, and learned-equivariant arms with the same endpoints. A second valid
fork changes the semantic target from laboratory-frame `cos(phi)` to a
gauge-invariant coordinate such as cosine relative to the observed reference.

Do not tune the current learned encoder against the visible extrapolation set.
Without a gauge repair, a better score would remain support-specific estimator
selection rather than construction of the claimed quotient.

## Artifacts and reproduction

- Aggregate: `data/experiments/tinyllm_invariant_frontend_causal/20260806_d8_preregistered/campaign_results.json`
- Per-seed results and weights: `data/experiments/tinyllm_invariant_frontend_causal/20260806_d8_preregistered/runs/`
- Preregistration: `docs/07 - Status Reports/2026-08-06_tinyllm-invariant-frontend-causal-preregistration.md`
- Runner: `experiments/structure_net/tinyllm_invariant_frontend_causal.py`
- Tests: `tests/structure_net/test_tinyllm_invariant_frontend_causal.py`
- Meta-hypothesis: `data/meta_hypotheses/tinyllm-invariant-frontend-stable-cosine-quotient-v1.json`

The frozen implementation SHA-256 is
`8784c19836f9175c03a11da112f399160435dfbe6722cdcbdd9b30dfe9dc8317`; the
aggregate SHA-256 is
`25a24c4331887395ca14922218986368893137f63283a3a93d954450b6e1d071`.
The failed meta-hypothesis and all 15 direct experiment records were read back
from ChromaDB. The focused experiment/predecessor gate completed with 23 tests
passing; the full repository gate completed with 411 passed, 1 skipped, and 0
failed.

```bash
pixi run python -m \
  experiments.structure_net.tinyllm_invariant_frontend_causal \
  --gpus 1 --slots-per-gpu 2 --max-parallel 2 \
  --output data/experiments/tinyllm_invariant_frontend_causal/20260806_d8_preregistered
```

Logical GPU ordinals are PyTorch ordinals, not necessarily NVIDIA-SMI indices.
