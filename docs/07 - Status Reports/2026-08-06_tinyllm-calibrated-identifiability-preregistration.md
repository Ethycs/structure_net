# TinyLLM calibrated identifiability causal test preregistration

**Status:** PREREGISTERED — TRAINING BLOCKED UNTIL IDENTIFIABILITY CONTRACT PASSES  
**Date:** 2026-08-06  
**Profile:** NAL-STD-EXPERIMENT `PREREGISTERED`  
**Hypothesis:** `tinyllm-calibrated-reference-stable-cosine-quotient-v1`  
**Schema:** `nal.tinyllm-calibrated-frontend-causal.v1`

## Question

Does an observed calibration reference that fixes the acquisition gauge allow
an analytic or structurally invariant sensor front end to retain absolute
cosine while removing the conditional phase branch under composition and
outside-range extrapolation?

## Observation and calibration contract

The ordinary N3 sensor remains unchanged. Each example additionally carries an
observed calibration packet produced by a known phase-independent pilot:

```text
C = (cos orientation, sin orientation, signed angular speed,
     amplitude, planar offset_x, planar offset_y,
     planar drift_x, planar drift_y)
```

The packet is acquisition metadata measured from the calibration pilot. It
contains no latent phase, future phase, target cosine, branch label, harmonic
phase, or task-bin label. Every condition receives the same packet.

This is a strong positive-control intervention, not a claim that such metadata
is free in real systems. It answers whether the previous failure was caused by
the observation gauge. Later work may replace the exact packet with a noisy or
learned pilot estimator only if this control succeeds.

## Identifiability gate before training

Let `z` denote semantic phase and target-preserving acquisition nuisance, let
`O_cal(z)` be the distribution of the quantized N3 sensor together with `C`,
and let `T(z) = cos(phi_future)`. Training is forbidden unless the implementation
establishes

```text
O_cal(z) = O_cal(z')  =>  T(z) = T(z').
```

The declared proof is:

1. equality of the calibration packets fixes positive amplitude, planar
   offset/drift, signed speed, and the orientation vector;
2. because the declared orientation support lies inside one `2*pi` chart,
   equality of `(cos orientation, sin orientation)` fixes orientation;
3. equality of observation distributions fixes the de-offset, de-drift,
   de-scaled, de-rotated planar mean at the final history step;
4. that unit vector fixes current phase modulo `2*pi`;
5. signed direction and the fixed future horizon therefore fix future phase
   and absolute cosine.

For the old gauge action,

```text
phase' = phase + alpha
orientation' = orientation - alpha
harmonic_phase' = harmonic_phase - k * alpha,
```

sensor means remain equal, but calibration equality implies
`exp(-i*alpha) = 1`. On the declared chart this forces `alpha = 0`, so the
target is unchanged.

The runner must also execute an exhaustive numerical contract over fixed phase,
orientation, harmonic-order, and nonzero gauge-shift grids. It must verify that
every target-changing old-gauge pair has distinct calibration observations.
The test uses observation distributions: sampled additive noise is not treated
as an adversarial latent capable of cancelling arbitrary signal differences.

## Three matched arms

Seeds 7, 17, 29, 41, and 53 compare:

1. `raw_calibrated`: raw quantized N3 tokens plus a learned embedding of `C`;
2. `analytic_calibrated`: fixed calibration-based canonicalization, one scalar
   feature token, and the same transformer;
3. `learned_calibrated_equivariant`: a learned planar-equivariant encoder whose
   vector output is paired with the orientation reference through dot/cross
   invariants, followed by one scalar feature token and the same transformer.

All models are newly trained. They share transformer shape and initialization
seed, 4,096 paired examples, paired minibatch schedule, 600 updates, batch size
64, AdamW learning rate `3e-4`, weight decay `0.01`, and gradient clipping at
`1.0`. The only objective is cosine-interval task cross-entropy. No
representation, adversarial, contrastive, or equivariance penalty is used.

## Fixed analytic front end

The positive control:

1. decodes the observed sensor bins;
2. subtracts calibrated planar offset and affine drift;
3. divides by calibrated positive amplitude;
4. rotates by the inverse calibrated orientation;
5. normalizes the final observed planar vector;
6. advances it by the calibrated direction over the fixed prediction horizon;
7. emits the laboratory-frame cosine coordinate.

It uses neither phase nor target. Quantization and sampled sensor noise remain.

## Learned invariant front end

The learned arm first applies the same reference-defined offset, drift, and
positive-scale correction. Shared temporal weights form planar vector channels;
all mixing coefficients depend only on rotation-invariant Gram features. The
resulting vector is equivariant. Its dot product and signed area with the
calibration orientation vector are invariant under joint planar rotation.
Together with signed speed, those invariants feed a learned scalar map.

The numerical architecture contract is

```text
F(a R x + o + d*t, R c, transformed calibration) = F(x, c, calibration)
```

for positive `a`, planar rotation `R`, constant offset `o`, and affine drift
`d*t`, with arbitrary discarded harmonic channel.

## Representation cuts and endpoint

Frozen nonlinear probes evaluate:

- `frontend`: flattened raw sensor plus calibration packet for the raw arm, and
  the emitted scalar for the two structured arms;
- `full`: final query residual after all eight transformer blocks.

At each cut, a seed passes a regime only when both hold:

```text
cosine Pearson correlation >= 0.90
conditional branch balanced accuracy <= 0.55
```

Conditional log-loss gain remains a secondary diagnostic. A condition succeeds
only if the same four of five seeds pass both cuts on both held-out composition
and outside-range extrapolation. In-distribution and task metrics are controls,
not alternate success paths.

## Interpretation

| Outcome | Meaning |
| --- | --- |
| analytic and learned pass | fixing the observation gauge makes the desired quotient constructible |
| analytic passes, learned fails | the target is identifiable but the learned invariant family or optimization is insufficient |
| composition passes, extrapolation fails | gauge repair is necessary but not sufficient for support-independent estimation |
| branch erased, cosine lost | the intervention still creates compression rather than the task quotient |
| analytic fails | stop learned optimization and audit calibration sufficiency or implementation |

## Fixed execution plan

- d8 TinyLLM;
- seeds 7, 17, 29, 41, and 53;
- 4,096 training examples and 600 task-only updates;
- probe train/validation/test sizes 2,048 / 512 / 1,024;
- at most two CUDA workers after a representative shakedown;
- artifact root:
  `data/experiments/tinyllm_calibrated_frontend_causal/20260806_d8_preregistered`.

The implementation digest is frozen before primary workers launch. Any change
after primary outcomes are visible creates a new exploratory campaign.
