# TinyLLM repeated-reference acquisition

**Status:** PRIMARY GATE INVALID; DIRECT ACQUISITION SUBRESULT SUPPORTED  
**Date:** 2026-08-07  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, frozen-system acquisition intervention  
**Hypothesis:** `tinyllm-reference-acquisition-replicates-v1`  
**Schema:** `nal.tinyllm-reference-acquisition-replicates.v1`  
**Preregistration:** [repeated-reference acquisition preregistration](../07%20-%20Status%20Reports/2026-08-07_tinyllm-reference-acquisition-replicates-preregistration.md)

## Verdict

Repeated acquisition repairs the unchanged end-to-end task exactly where the
inverse-square precision law predicts. At `sigma=0.175` radians, neither the
analytic circular mean nor the learned equivariant moment denoiser passes a
checkpoint at `m=1`, `4`, or `16`. At `m=64`, both methods pass composition
and extrapolation simultaneously in **5/5 analytic** and **5/5 learned**
checkpoints. The same-reference misgrouping control passes `0/5` in both arms.

The full preregistered campaign is nevertheless **invalid**. The retained
one-step true-coordinate task-gradient write passes only `0/5` analytic and
`2/5` learned checkpoints, below its required four-of-five positive-control
gate. The locked classification is therefore:

```text
invalid
```

The acquisition result is a strong, preregistered subresult, not confirmation
of the complete registered hypothesis. It shows that a coherent input-side
reference correction restores the frozen computation; it simultaneously
falsifies the assumption that the earlier one-step local residual write is a
portable causal ceiling at this larger orientation error.

## Primary outcome matrix

Each cell is the number of checkpoints whose unchanged answer head stays
within three accuracy points of its own exact-reference clean baseline on both
composition and extrapolation.

| Acquisition method | Repeats | Analytic front end | Learned equivariant front end |
| --- | ---: | ---: | ---: |
| analytic circular mean | `1` | `0/5` | `0/5` |
| analytic circular mean | `4` | `0/5` | `0/5` |
| analytic circular mean | `16` | `0/5` | `0/5` |
| analytic circular mean | `64` | **`5/5`** | **`5/5`** |
| learned equivariant moment | `1` | `0/5` | `0/5` |
| learned equivariant moment | `4` | `0/5` | `0/5` |
| learned equivariant moment | `16` | `0/5` | `0/5` |
| learned equivariant moment | `64` | **`5/5`** | **`5/5`** |

The registered grid therefore brackets complete frozen-task recovery between
effective angular standard errors

```text
0.175 / sqrt(16) = 0.04375 radians  (2.51 degrees)
0.175 / sqrt(64) = 0.021875 radians (1.25 degrees).
```

This is consistent with the preceding orientation titration, where the first
registered failure appeared near two degrees.

## Accuracy recovery

Mean loss from the exact-reference clean accuracy is shown below for the
analytic circular mean. The learned denoiser is numerically indistinguishable
at the registered precision.

| Front end | Repeats | Composition loss | Extrapolation loss |
| --- | ---: | ---: | ---: |
| analytic | `1` | `35.55` points | `27.21` points |
| analytic | `4` | `20.31` | `14.26` |
| analytic | `16` | `7.77` | `3.85` |
| analytic | `64` | **`1.80`** | **`1.31`** |
| learned | `1` | `34.24` | `17.89` |
| learned | `4` | `18.71` | `6.48` |
| learned | `16` | `6.93` | `1.60` |
| learned | `64` | **`1.70`** | **`0.80`** |

The worst checkpoint at `m=64` remains within the gate: analytic maximum loss
is `2.73` points on composition and `1.66` on extrapolation; learned maximum
loss is `2.64` and `2.73` points respectively.

## The inverse-square mechanism is exact enough

The analytic circular-mean angular-RMSE slopes are:

| Shift | `d log(RMSE) / d log(m)` | Registered interval |
| --- | ---: | ---: |
| composition | `-0.4837` | `[-0.60,-0.40]` |
| extrapolation | `-0.4944` | `[-0.60,-0.40]` |

Both pass. The nested repeats preserve the predecessor's first draw exactly,
all repeats are shared across both sheets of each exact fiber, and both arms
and every checkpoint receive identical observations.

The learned denoiser adds no material value. At `m=64`, its angular-RMSE excess
over the analytic mean is only `4.85e-7` radians on composition and `3.30e-7`
on extrapolation. Its fitted coefficients are

```text
(1.00915, 0.00544, 0.00074),
```

close to the analytic first moment `(1,0,0)`. The higher-moment correction is
unnecessary under unbiased independent Gaussian orientation error.

## Why the complete gate is invalid

The one-step true-coordinate residual write was intended as a causal positive
control. It fails systematically:

| Causal write | Analytic front end | Learned front end |
| --- | ---: | ---: |
| true target cosine | **`0/5`** | **`2/5`** |
| shuffled target cosine | `0/5` | `0/5` |

All writes are finite, and the shuffled control behaves correctly. The failure
is concentrated on composition: analytic true-coordinate losses are
`3.91--6.84` points and learned losses `2.15--9.86` points. Extrapolation often
improves, so this is not simple absence of task information.

The required patch norms are large—roughly `41--65` residual units. A local
task gradient can set the posterior's ordered first moment to first order, but
that does not keep a large displacement on the trained residual manifold or
guarantee the correct argmax bin after nonlinear continuation. By contrast,
repeated acquisition changes the observed reference and lets the frozen front
end and transformer follow a coherent on-manifold computation.

The experiment therefore rejects this stronger assumption:

```text
one-step local true-coordinate write
    == universal positive-control ceiling for acquisition repair.
```

It does not reject the direct acquisition intervention.

## Controls and integrity

- all ten exact-reference and `m=1` task metrics replay the predecessor
  exactly (`maximum error = 0`);
- the inherited `sigma=0.175` representation gate passes in every system;
- pair-shared repeat error and source-first-draw error are both exactly zero;
- misgrouped same-reference membership passes `0/5` in both arms;
- the shuffled-coordinate residual write passes `0/5` in both arms;
- every result is finite and every system state hash remains unchanged;
- no front end, TinyLLM, answer head, or representation probe is trained;
- only four shared three-parameter calibration denoisers are fit; and
- exact resume leaves campaign and artifact bytes unchanged.

| Item | Value |
| --- | --- |
| systems requested/completed | `10/10` |
| fitted acquisition parameters | `12` |
| implementation SHA-256 | `6c3cc4463b2c515280c778461e4606dca6ffb19f08d11cb5b7a852a942c7df77` |
| campaign SHA-256 | `269fd948f0d6fee8916bbe3cb94c1d87f76572e43c103b52fc8775fa9653031e` |
| result-manifest SHA-256 | `8181d48f99850d5e487b5209d648373410bea3b46b2eafcb58f57118ed898c1c` |
| repeat-array SHA-256 | `8886c1f6ad0fd307720748e44b1741edb656fdb9e29e913c7ad059b72397eef4` |
| denoiser SHA-256 | `7373577fd3a6535c9183eeeac393389dd54f5bf9bb64f5baeb6f4b371e66f702` |
| device | NVIDIA GeForce RTX 2060 SUPER (`cuda:1`) |
| peak CUDA allocation | `646,493,184` bytes |
| analysis time | `125.31` seconds |

## Artifacts and reproduction

- primary campaign:
  `data/experiments/tinyllm_reference_acquisition_replicates/20260807_d8_preregistered/campaign_results.json`
- per-checkpoint results:
  `data/experiments/tinyllm_reference_acquisition_replicates/20260807_d8_preregistered/runs/*/seed_*/result.json`
- nested repeat arrays:
  `data/experiments/tinyllm_reference_acquisition_replicates/20260807_d8_preregistered/reference_repeat_arrays.npz`
- fitted moment denoisers:
  `data/experiments/tinyllm_reference_acquisition_replicates/20260807_d8_preregistered/equivariant_moment_denoisers.pt`
- valid systems-only shakedown:
  `data/experiments/tinyllm_reference_acquisition_replicates/20260807_shakedown_cuda_v2/`
- preserved pre-lifecycle-correction shakedown:
  `data/experiments/tinyllm_reference_acquisition_replicates/20260807_shakedown_cuda/`
- runner and tests:
  `experiments/structure_net/tinyllm_reference_acquisition_replicates.py`,
  `tests/structure_net/test_tinyllm_reference_acquisition_replicates.py`

```bash
MPLCONFIGDIR=/tmp/matplotlib-reference-acquisition-primary \
pixi run python -m \
  experiments.structure_net.tinyllm_reference_acquisition_replicates \
  --device cuda:1 \
  --output \
  data/experiments/tinyllm_reference_acquisition_replicates/20260807_d8_preregistered
```

## Interpretation and next shortest diagnostic

The direct intervention supports measurement precision as the practical
bottleneck under this unbiased synthetic noise model. Sixty-four independent
observations recover both structured systems without learned denoising or any
model change. That is an acquisition-cost result, not a representation-
learning result.

The invalid positive control also sharpens the interpretability question. The
next diagnostic should be a frozen **reference-path versus residual-tangent
transport** audit:

1. interpolate the orientation reference along the shortest circular path
   from the stored `m=1` estimate to the `m=64` estimate;
2. record the actual frozen residual and task output along that input-side
   path;
3. compare the true residual displacement with the one-step task-gradient
   write and with a multi-step locally relinearized write; and
4. use a matched shuffled reference path as the negative control.

If path integration succeeds where the one-step write fails, the previous
ceiling was outside its local validity radius. If it still fails while the
actual reference path succeeds, the scalar task-gradient chart is not
integrable into the trained residual manifold. No retraining, nonlinear
readout, topology scan, or link-cobordism analysis should precede this audit.

## Scope boundary

The repeats are independent, unbiased Gaussian angular observations of a
synthetic calibration reference. The result does not establish robustness to
correlated noise, systematic bias, real sensing cost, natural-language tasks,
or an architecture population. Because the preregistered causal ceiling
failed, the full hypothesis remains unconfirmed despite the clean acquisition
subresult.
