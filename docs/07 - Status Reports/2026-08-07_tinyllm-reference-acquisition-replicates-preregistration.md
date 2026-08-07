# TinyLLM repeated-reference acquisition preregistration

**Status:** COMPLETED — `invalid`; acquisition `5/5` both arms, causal ceiling `0/5` and `2/5`  
**Date:** 2026-08-07  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, frozen-system acquisition intervention  
**Hypothesis:** `tinyllm-reference-acquisition-replicates-v1`  
**Schema:** `nal.tinyllm-reference-acquisition-replicates.v1`

**Pre-primary lifecycle amendment:** the first systems-only seed-7 root
completed both frozen arms with exact source replay, pair-shared repeats,
finite causal writes, unchanged state hashes, and valid inverse-square scaling.
Its campaign aggregate nevertheless reported `valid=false` because the runner
incorrectly included the one-seed causal-ceiling quality outcome in systems
validity. The root is preserved. Before any `m=64` primary outcome existed or
was inspected, the runner was changed only so `allow_underpowered` reports
systems validity independently of scientific population gates. The primary
oracle, control, scaling, task, threshold, and classification rules below are
unchanged. The clean systems root is `20260807_shakedown_cuda_v2`.

## Decision question

The orientation-only campaign found that every registered quotient-
representation cell still passed at `0.175` radians, while the frozen task
interface had already failed. The subsequent causal decomposition showed that
a true-coordinate write can restore the frozen continuation at larger joint
calibration error. The readout-only intervention repaired the learned arm at
`0.035` radians but not the analytic arm at population level.

Test the shortest remaining acquisition-side question before any front-end,
TinyLLM, answer-head, nonlinear-readout, or representation training:

> Can repeated independent observations of the same orientation reference
> restore frozen task utility at `sigma=0.175` through the predicted
> inverse-square sample-efficiency law?

## Locked sources

Use the exact ten analytic and learned calibrated systems, datasets, first
noise draw, and source metrics from:

```text
data/experiments/tinyllm_calibration_orientation_noise/
    20260807_d8_preregistered/campaign_results.json
SHA-256 876af062f9d0bdd365e2f1ffbb959d9ff2e5e4e277321b510bbe6a956456877f
implementation 990975365c7f0c468c3895029c1a87a98fa14d5a28853a1fc445070769218c70
noise arrays b8d959169f2f249d635037a362c70522e514ace13d3bf656a91bdaea99ef43e7
```

The underlying calibrated campaign remains fixed at SHA-256
`80623a7283fed3d902a1cbf9fb58afe2a2fef82ab86406212f907de40a96c501`.
Use seeds `7,17,29,41,53`, the exact source cohorts, and only the analytic
calibrated and learned calibrated-equivariant arms. Freeze both front ends,
every TinyLLM parameter, scalar embedding, layer norm, and answer head.

The `m=1` condition must replay the source orientation campaign at
`sigma=0.175` within `2e-6` for every stored task metric. Its inherited
front-end/full-depth representation gate must already pass. Failure of either
contract invalidates the campaign.

## Repeated-reference generator

Use `sigma=0.175` radians and the nested prefix counts

```text
m = {1, 4, 16, 64}.
```

The first deviate for every exact cosine fiber is the predecessor's stored
orientation-noise value. Generate the remaining 63 independent standard-
normal deviates once from the locked experiment seed, share every deviate
across the two sheets of its fiber, and use identical arrays across front-end
arms and checkpoints. Save the arrays.

For clean orientation `z` on the unit circle, repeat `j` is

```text
z_j = z exp(i sigma epsilon_j).
```

Report the measured circular angular RMSE and resultant length for every
cohort and count. The analytic standard error prediction is

```text
sigma_eff(m) = sigma / sqrt(m).
```

On both composition and extrapolation, the fitted slope of
`log(angular RMSE)` against `log(m)` must lie in `[-0.60,-0.40]`. This scaling
gate is fixed independently of task performance.

## Acquisition arms

### Analytic circular mean — positive-control mechanism

Normalize the complex sample mean:

```text
z_hat = sum_j z_j / |sum_j z_j|.
```

This computation sees only repeated observed calibration vectors. It never
uses phase, cosine, task bins, or target labels.

### Learned equivariant moment denoiser

Fit one three-coefficient, reflection-compatible equivariant denoiser per
repeat count on the training calibration cohort only. From

`M_r = mean_j(z_j^r)`, construct the charge-one features

```text
v_1 = M_1
v_2 = M_2 conjugate(M_1)
v_3 = M_3 conjugate(M_1)^2.
```

Fit three **real** coefficients by deterministic ridge regression toward
`(1,0,0)`, with ridge ratio `0.01`, to the known clean calibration orientation.
Normalize the resulting charge-one vector. This head is exactly rotation-
equivariant and reflection-compatible by construction. It uses calibration
supervision, not phase or task labels. Save all twelve scalar coefficients.

At `m=64`, its angular RMSE may exceed the analytic circular mean by at most
`0.002` radians on each shifted cohort. This comparison is a declared
mechanism gate, not a post-outcome model choice.

### Misgrouped-repeat negative control

At the largest count, preserve the first correct observation but cyclically
reassign observations `2..64` between distinct exact fibers before averaging.
This preserves marginal observation statistics and computation while
destroying same-reference membership. It may pass the frozen task gate in at
most one of five checkpoints per arm.

## Frozen task endpoint

Evaluate the unchanged answer-token argmax on composition and extrapolation.
For each checkpoint, compare every repeated-reference condition with that
checkpoint's exact-reference clean accuracy. A method/count passes only when
accuracy loss is at most `0.03` absolute on both shifts and every provenance,
noise, finite-value, frozen-state, and replay contract passes.

A population arm passes in at least four of five checkpoints. The primary
constructive endpoint requires both analytic and learned denoisers to pass
both structured front-end arms at `m=64`. Report the first passing count per
checkpoint without changing the locked grid.

Also report front-end cosine correlation/RMSE and target cross-entropy. These
are descriptive and cannot replace the exact-bin task gate. No new
representation probe is fit.

## Retained causal ceiling

At `m=1`, apply the predecessor's local task-gradient write to the frozen full
residual, once with the true target cosine and once with a fixed shuffled
target cosine. The true-coordinate write must pass the same three-point task
gate in at least four of five checkpoints per arm. The shuffled write may pass
at most one seed per arm. This is a label-using causal ceiling, not a deployable
acquisition method.

## Locked classification

| Outcome | Classification | Decision |
| --- | --- | --- |
| provenance, source replay, scaling, finite, or causal-ceiling gate fails | `invalid` | repair lifecycle only under a new root |
| analytic and learned denoisers pass both front-end arms | `acquisition_precision_sufficient` | retain all frozen systems; quantify acquisition cost |
| analytic passes both arms but learned does not | `analytic_reference_averaging_sufficient` | use the analytic estimator; reject the learned denoiser |
| learned passes both arms but analytic does not | `learned_denoising_required` | localize the useful higher-moment correction |
| exactly one structured front-end arm passes | `arm_stratified_acquisition_repair` | keep the remaining failure front-end specific |
| neither acquisition method passes | `reference_averaging_insufficient` | test reference-model bias or the bounded analytic nonlinear-readout ceiling |

No repeat count, noise scale, ridge value, feature order, threshold, cohort,
checkpoint, or classification may change after primary outcomes are inspected.

## Fixed artifacts

- runner:
  `experiments/structure_net/tinyllm_reference_acquisition_replicates.py`
- tests:
  `tests/structure_net/test_tinyllm_reference_acquisition_replicates.py`
- shakedown root:
  `data/experiments/tinyllm_reference_acquisition_replicates/20260807_shakedown_cuda_v2`
- preserved pre-correction systems root:
  `data/experiments/tinyllm_reference_acquisition_replicates/20260807_shakedown_cuda`
- primary root:
  `data/experiments/tinyllm_reference_acquisition_replicates/20260807_d8_preregistered`
- report:
  `docs/08 - Analysis/2026-08-07_tinyllm-reference-acquisition-replicates.md`
- meta hypothesis:
  `tinyllm-reference-acquisition-replicates-v1`

## Scope boundary

This experiment uses unbiased independent Gaussian angular errors in a
synthetic calibration reference. It does not estimate a real instrument's
sample cost, correlated errors, systematic bias, or natural-language
behavior. The selected checkpoints are replication units rather than a random
architecture population.
