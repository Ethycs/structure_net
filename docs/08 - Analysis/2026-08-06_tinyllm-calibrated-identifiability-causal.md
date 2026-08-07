# TinyLLM calibrated identifiability causal test

**Status:** PASSED PREREGISTERED GATE — gauge repair makes the quotient
constructible

**Date:** 2026-08-06

**Hypothesis:** `tinyllm-calibrated-reference-stable-cosine-quotient-v1`

**Conformance:** `PREREGISTERED`; the observation contract, proof obligation,
arms, and thresholds were fixed in the
[preregistration](../07%20-%20Status%20Reports/2026-08-06_tinyllm-calibrated-identifiability-preregistration.md).

## Verdict

Fixing the observation gauge made the absolute-cosine quotient constructible.
Both the analytic canonicalizer and the learned calibrated-equivariant encoder
passed the complete seedwise gate in **5/5 seeds**; four were required. The raw
calibrated TinyLLM passed in 0/5.

At full depth under outside-range extrapolation, the analytic arm achieved mean
`(cosine, branch) = (0.992, 0.496)` and the learned arm achieved
`(0.987, 0.510)`. Both also passed at the front-end output and on held-out
composition in every seed. Conditional log-loss gain was approximately zero.

This is the causal result predicted by the preceding negative study. An
unrestricted transformer did not learn to use the calibration packet robustly;
the fixed and structurally invariant front ends did. The desired quotient
therefore becomes stable when the target is made identifiable and the nuisance
symmetry is respected by construction.

## Pre-training identifiability contract

Training was hard-gated on

```text
O_cal(z) = O_cal(z')  =>  T(z) = T(z').
```

The observed phase-independent pilot record contained orientation vector,
signed speed, positive amplitude, planar offset, and planar affine drift. It
contained no phase, branch, harmonic phase, target cosine, or task label.

The proof fixes the acquisition transform from equality of calibration records,
then recovers current planar phase from equality of corrected observation
distributions and advances it using the observed direction. The previous gauge
action still produced sensor means equal to `1.07e-15`, but every one of 12,096
target-changing gauge pairs had a distinct calibration observation. There were
zero violations, and the minimum calibration distance was `0.04999`.

The implementation repeated this contract at campaign construction and inside
every worker before training.

## Campaign integrity

The campaign completed 15/15 cells with no failures. Every result records
implementation SHA-256
`73e152b9b6c7ab51ad08e6a80786c32a71566f9d4fb410ff2cf3df53761faf77`.
Within each seed, all three arms have identical training-data and minibatch
schedule hashes.

Seven new contract tests and thirteen predecessor regression tests passed
before launch. A three-cell CUDA shakedown completed first. The confirmatory run
used two workers on PyTorch logical GPU 1, the RTX 2060 SUPER. Peak allocated
memory was 1.525 GiB for raw and 1.023 GiB for each structured arm.

Before this experiment, the complete 20.62 GB research data tree was tracked by
DVC and committed to `lakefs://artifacts/main/structure-net/` as lakeFS commit
`105b3c97fdb710653685ec117b3c45164c7b22193fb293b09de9b75dfda90ae7`.

## Fixed design

- d8 TinyLLM with 50,965,504 transformer parameters, including one extra
  positional embedding shared by all arms;
- seeds 7, 17, 29, 41, and 53;
- N3 support, 4,096 paired examples, batch size 64, 600 task-only updates;
- AdamW at `3e-4`, weight decay `0.01`, gradient clipping at `1.0`;
- raw arm: original sensor tokens plus a learned embedding of the calibration
  packet;
- analytic arm: reference-defined offset/drift removal, scale normalization,
  inverse orientation, endpoint normalization, and horizon advancement;
- learned arm: shared planar temporal channels, invariant Gram-conditioned
  vector mixing, and dot/cross invariants with the orientation reference;
- fresh nonlinear conditional probes on 2,048 / 512 / 1,024 disjoint
  train/validation/test examples;
- primary cuts: front-end output and full transformer depth.

The system parameter totals were 50,970,112 raw, 50,966,528 analytic, and
50,976,625 learned. No representation, adversarial, contrastive, or
equivariance loss was used.

## Primary endpoints

Each cell reports the five-seed mean `(cosine correlation, conditional branch
balanced accuracy)` and seed pass count. A seed passes only when cosine is at
least 0.90 and branch accuracy is at most 0.55.

| Arm and cut | Composition | Extrapolation |
| --- | --- | --- |
| Raw frontend | `(0.959, 0.998)`, 0/5 | `(0.001, 0.498)`, 0/5 |
| Raw full | `(0.972, 0.622)`, 0/5 | `(0.481, 0.521)`, 0/5 |
| Analytic frontend | `(0.972, 0.501)`, 5/5 | `(0.964, 0.499)`, 5/5 |
| Analytic full | `(0.998, 0.499)`, 5/5 | `(0.992, 0.496)`, 5/5 |
| Learned frontend | `(0.972, 0.498)`, 5/5 | `(0.960, 0.501)`, 5/5 |
| Learned full | `(0.999, 0.501)`, 5/5 | `(0.987, 0.510)`, 5/5 |

The learned full-depth extrapolation seed range was 0.973–0.992 for cosine and
0.488–0.532 for branch accuracy. Thus the result is not driven by a favorable
mean: every individual learned seed passed every declared primary cell.

The analytic front-end range confirms that the positive control itself passed
before transformer processing. Transformer depth then sharpened its cosine
correlation from 0.964 to 0.992 on extrapolation. The learned encoder tracked
the control closely, losing only 0.0047 mean full-depth extrapolation
correlation.

## Task behavior

| Arm | In-distribution accuracy | Composition accuracy | Extrapolation accuracy |
| --- | ---: | ---: | ---: |
| Raw calibrated | 0.490 | 0.389 | 0.130 |
| Analytic calibrated | 0.751 | 0.745 | 0.616 |
| Learned calibrated-equivariant | 0.738 | 0.717 | 0.492 |

Calibration alone did not rescue the unrestricted raw transformer: its task and
representation behavior remained close to the prior raw N3 baseline. Putting
the calibration into a fixed or structurally invariant map substantially
improved both quotient geometry and task generalization. The learned arm
approached the analytic control on representation geometry but retained a
12.5-point extrapolation task gap, leaving room for optimization within the
now-correct function class.

## Causal interpretation

The sequence of studies now separates three claims:

1. More nuisance coverage and residual penalties do not repair a target that is
   undefined on observational equivalence classes.
2. A structurally invariant bottleneck without a gauge reference contracts the
   phase branch but loses absolute cosine under extrapolation.
3. Supplying a phase-independent gauge-fixing reference makes the target
   identifiable; both analytic and learned invariant maps then preserve cosine
   and contract branch under the same shift.

The raw arm is essential. It received exactly the same calibration information
yet failed the gate, showing that information availability alone was
insufficient. The successful intervention was the combination of identifiable
observation and an architecture that used the reference according to the
symmetry.

The result supports the design principle:

> First quotient latent states by observational indistinguishability. Only then
> define the semantic task quotient—or supply a reference that fixes the gauge.

## Limits and next experiment

The calibration packet is an exact strong control. Real instruments estimate
orientation, scale, offset, drift, and timing with error. This experiment does
not establish robustness to calibration noise or missing calibration fields.
It also treats additive sensor noise probabilistically; an arbitrary noise
realization is not allowed to cancel an arbitrary semantic signal in the
identifiability statement.

The next experiment should degrade the calibration packet along a preregistered
noise curve and identify where the joint quotient gate breaks. A complementary
study should test the gauge-invariant relative target without any calibration
record. Neither should retune against the current extrapolation set.

## Artifacts and reproduction

- Aggregate: `data/experiments/tinyllm_calibrated_frontend_causal/20260806_d8_preregistered/campaign_results.json`
- Per-seed weights and results: `data/experiments/tinyllm_calibrated_frontend_causal/20260806_d8_preregistered/runs/`
- Preregistration: `docs/07 - Status Reports/2026-08-06_tinyllm-calibrated-identifiability-preregistration.md`
- Runner: `experiments/structure_net/tinyllm_calibrated_frontend_causal.py`
- Tests: `tests/structure_net/test_tinyllm_calibrated_frontend_causal.py`

```bash
pixi run python -m \
  experiments.structure_net.tinyllm_calibrated_frontend_causal \
  --gpus 1 --slots-per-gpu 2 --max-parallel 2 \
  --output data/experiments/tinyllm_calibrated_frontend_causal/20260806_d8_preregistered
```
