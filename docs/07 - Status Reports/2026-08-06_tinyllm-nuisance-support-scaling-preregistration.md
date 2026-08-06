# TinyLLM Nuisance-Support Scaling Preregistration

**Status:** PREREGISTERED; RESULTS NOT YET INTERPRETED

**Date:** 2026-08-06

**Hypothesis:** `tinyllm-nuisance-invariant-internal-cosine-quotient-v1`

**Depends on:** `../08 - Analysis/2026-08-05_tinyllm-conditional-branch-depth-scan.md`

## Question

Does increasing nuisance diversity move the learned internal cosine quotient
from distribution-relative fiber collapse to a nuisance-invariant
factorization, while training sample count, optimization steps, architecture,
and seed remain fixed?

## Factorial design

The campaign uses a d8 TinyLLM, seeds 7/17/29/41/53, 4,096 fixed training
examples, 600 AdamW updates, and batch size 64 in every cell. It crosses four
nested nuisance supports with four objectives, for 80 independently trained
models:

| Support | Added nuisance axes |
| --- | --- |
| N0 | Fixed amplitude, offset, orientation, speed, harmonic, direction, and noise |
| N1 | Amplitude and three-channel offset |
| N2 | N1 plus orientation and angular speed |
| N3 | N2 plus harmonic strength/order/phase, direction, and observation noise |

The objective arms are ordinary final-depth, discrete multi-exit, continuous
gate, and an explicit quotient-contrastive objective. Ordinary N1--N3 cells
are the nuisance-augmented ordinary arm. Every objective is run at every
support level so data coverage and objective pressure remain separable.

N2 and N3 training use anti-diagonal amplitude/orientation combinations. The
composition test uses the held-out diagonal combinations while retaining
familiar marginal values. N0 and N1 results on that split measure transfer
into unseen axes and are not described as pure compositional generalization.

## Evaluation

Every model is evaluated on its training support, every larger support, the
shared N3 compositional split, and an outside-range extrapolation family. A
frozen nonlinear probe is fit only on fresh samples from the model's training
support. It jointly predicts cosine and the phase branch conditional on exact
cosine.

The primary representation cuts are real depth 0.005 and full depth. Block-1
query, post-attention, and post-MLP cuts repeat the same measurements to
localize shifted failure.

For each cut and family, report the two primary coordinates separately:

1. base retention: Pearson correlation between probe-predicted and true cosine;
2. fiber leakage: balanced branch accuracy conditional on exact cosine.

Successful invariant quotienting requires cosine correlation at least 0.90
and branch accuracy at most 0.55.

## Branch-conditioned fiber geometry

For exact interior cosine fibers, compute mean cross-branch distance divided by
mean within-branch nuisance distance:

`gamma_fiber = D_cross / D_within`.

The primary distance is Fisher--Rao distance between task posteriors decoded
from the selected residual cut. Hidden-space Euclidean and cosine distances
are controls. A fiber is operationally merged when gamma is at most 1.25. This
is a finite sampled statistic, not a certified Reeb graph, cosheaf, or
conditional mutual-information estimate.

## Preregistered conclusions

- **Strong success:** one arm at N3 passes base retention, fiber leakage, and
  Fisher gamma at both primary cuts on composition and extrapolation in all
  five seeds.
- **Coverage-dependent success:** the paired seedwise base-retention slope is
  positive across N0--N3 on both shifted regimes, N3 improves on N0 by at least
  0.05, branch leakage remains at most 0.55, but strong success is not reached.
- **Objective-dependent success:** the explicit quotient or a joint-depth arm
  reaches strong success at N3 while ordinary final-depth N3 does not.
- **Failure despite broad coverage:** no N3 arm reaches 0.90 cosine retention
  under both shifted regimes. This redirects the next intervention toward an
  equivariant/invariant encoder rather than stronger topology claims.

Shared-run wall time is not a benchmark. Per-cell checkpoints and detailed
JSON records are retained, and the canonical spawned NAL runner supplies
logical-GPU scheduling, retry, and fingerprinted completed-result resume.

## Protocol amendment: primary-35 design

After 29 full-factorial cells had completed, the remaining design was reduced
before any N2 or N3 result was observed. The primary analysis now contains 35
models:

- ordinary final-depth training at N0, N1, N2, and N3 for all five seeds;
- discrete multi-exit, continuous gate, and quotient-contrastive training at
  N3 for all five seeds.

This retains the ordinary nuisance-coverage scaling curve and the N3 objective
comparison needed by every preregistered conclusion. Objective-by-support
interaction cells at N0--N2 are supplementary rather than required. Existing
completed cells remain reported and are not discarded.

The amendment was motivated by experimental deduplication and an implementation
memory defect, not an N2/N3 outcome. The original multi-depth loop retained all
four transformer graphs until a combined backward pass, causing two of the N0
continuous-gate cells to exhaust an 8 GiB GPU. Immediate per-depth backward is
algebraically the same mean objective, released each graph promptly, and passed
a two-worker d8 CUDA calibration. Endpoints, thresholds, seeds, update count,
sample count, and nuisance generators are unchanged.
