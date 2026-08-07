# TinyLLM noisy-reference readout recalibration preregistration

**Status:** COMPLETED — `arm_stratified_readout_repair`; learned `5/5`, analytic `3/5`  
**Date:** 2026-08-07  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, frozen-system readout intervention  
**Hypothesis:** `tinyllm-noisy-reference-readout-recalibration-v1`  
**Schema:** `nal.tinyllm-noisy-reference-readout-recalibration.v1`

**Pre-aggregate lifecycle amendment:** the first primary launch under
`20260807_d8_preregistered` used `cuda:1` and terminated after writing all five
analytic records but before starting the learned arm or writing a campaign
aggregate. The exact terminating stderr was not retained. An initial diagnosis
incorrectly matched host `nvidia-smi` physical index 1 to PyTorch logical
`cuda:1`; a subsequent `cuda:2` attempt failed before loading any checkpoint
and proved that the logical ordinals are remapped. PyTorch `cuda:1` is the free
8 GiB RTX 2060 SUPER. The original interruption occurred near the command
execution window and is not labelled an OOM. The five analytic gate booleans
were inspected during diagnosis. They are therefore preserved as exposed
outcomes, and the analytic arm of the clean relaunch is explicitly corrective
replication. No learned-arm outcome was produced or inspected. The learned arm
retains preregistered unseen-outcome status. The failed `cuda:2` root contains
no result and is systems-only failure evidence.

The clean, single-device authoritative root is
`20260807_d8_preregistered_cuda1_persistent_mixed_pedigree` on PyTorch logical
`cuda:1`, executed in a persistent terminal session. The intervention, cohorts,
seeds, corruption, ridge value, target transform, controls, thresholds,
classification, and stop rules are unchanged. The campaign must record
`preregistered_mixed_pedigree_partial_analytic_exposure`, with arm-level
evidence roles distinguishing corrective analytic replication from the
preregistered learned result.

## Decision question

The orientation-reference titration found a complete-system radius of zero on
the registered grid even though all representation cells continued to pass at
the first nonzero error. At `sigma=0.035` radians (about two degrees), the
analytic arm passed `0/5` task gates and the learned arm `2/5`, while cosine
retention and conditional branch contraction passed in every checkpoint.

Test the shortest remaining localization question before retraining a front
end or TinyLLM:

> Can refitting only the frozen system's existing-capacity answer-token
> readout restore noisy-reference task utility without damaging clean utility?

The target is cosine binned on the ordered interval `[-1,1]`. This is not a
circular-output problem. All calibration and evaluation below therefore use
the declared ordered bin centers.

## Locked sources

Use only the two structured arms and five checkpoints from the completed
orientation-only campaign:

```text
data/experiments/tinyllm_calibration_orientation_noise/
    20260807_d8_preregistered/campaign_results.json
SHA-256 876af062f9d0bdd365e2f1ffbb959d9ff2e5e4e277321b510bbe6a956456877f
implementation 990975365c7f0c468c3895029c1a87a98fa14d5a28853a1fc445070769218c70
noise arrays b8d959169f2f249d635037a362c70522e514ace13d3bf656a91bdaea99ef43e7
```

The underlying calibrated-system campaign remains fixed at SHA-256
`80623a7283fed3d902a1cbf9fb58afe2a2fef82ab86406212f907de40a96c501`.
Reload the analytic and learned calibrated-equivariant systems at seeds
`7,17,29,41,53`. Freeze every TinyLLM, front end, scalar embedding, layer norm,
and residual activation. Only detached final-residual readouts may be fit.

The predecessor's representation result at `sigma=0` and `0.035` is inherited
only after every source result hash, scientific fingerprint, and declared
representation gate replays. No new representation probe is fit.

## Exact cohorts and corruption

Regenerate the predecessor's five cohorts exactly:

| Split | Seed | Samples | Use |
| --- | ---: | ---: | --- |
| train | `19184` | `2048` | readout fitting only |
| validation | `19294` | `512` | descriptive fit audit only |
| in distribution | `19390` | `1024` | descriptive evaluation |
| composition | `20399` | `1024` | primary evaluation |
| extrapolation | `21408` | `1024` | primary evaluation |

Evaluate only `sigma=0` and `sigma=0.035`. Reuse the exact stored standard-
normal value for every fiber from the predecessor's noise array. The same
angular corruption remains shared by both C2 sheets, arms, checkpoints, and
readout conditions. Dataset and noise-array bytes must replay before fitting.

## Frozen representation supplied to the readout

For every example, extract the frozen full residual `r` and apply the frozen
final layer norm:

```text
x = LN_f(r).
```

Let `W_0` be the sixteen answer-token rows of the frozen language-model head.
The untouched baseline is

```text
p_0(y | x) = softmax(W_0 x).
```

Its clean and noisy exact-bin accuracy and target cross-entropy must reproduce
the predecessor metrics. The fit never sees composition or extrapolation
labels.

## Readout arms

Fit each arm separately on clean train residuals and on noisy train residuals.
Evaluate every fitted arm on both clean and noisy composition/extrapolation,
forming a fixed `fit condition × evaluation condition` matrix.

### Existing-capacity linear answer head — primary

Fit a new `16 × d_model` weight matrix with no bias, exactly matching the
answer-token slice of the existing head. Use deterministic ridge regression
toward `W_0` on centered log target posteriors:

```text
Y = log(P_target) - mean_class(log(P_target))
lambda = 0.01 * trace(X^T X) / d_model
W* = argmin_W ||X W^T - Y||_F^2 + lambda ||W-W_0||_F^2.
```

The closed-form solve has no optimizer, minibatch, early-stopping, or
post-outcome hyperparameter choice. Save every fitted weight matrix.

### Affine scalar interval calibrator — low-capacity control

From the untouched posterior, compute its expected ordered coordinate using
the sixteen equally spaced centers on `[-1,1]`. Fit only slope and intercept by
least squares to the target posterior's expected coordinate. Quantize the
affine prediction to the nearest ordered center. This arm can repair a global
scale or offset but cannot select a new hidden-state direction.

### Target-shuffled linear head — negative control

Apply one fixed deterministic permutation to noisy training targets and run
the identical linear solve. It must remain below `0.20` exact-bin accuracy on
both noisy shifted regimes in every checkpoint. The shuffle is checkpoint-
specific but arm-independent and is fixed by `8_100_019 + seed`.

## Primary endpoint

All accuracy losses are measured against the untouched frozen head at
`sigma=0` in the same checkpoint, arm, cohort, and shift.

A noisy-fitted readout passes a checkpoint only when all of the following hold:

1. the inherited `sigma=0.035` representation gate passes;
2. on noisy composition and extrapolation, exact-bin accuracy loss is at most
   `0.03` absolute;
3. on clean composition and extrapolation, exact-bin accuracy loss is at most
   `0.03` absolute; and
4. every source, feature-width, finite-value, and immutable-artifact contract
   passes.

The clean-fitted linear head is a protocol positive control: it must remain
within `0.03` of the untouched clean baseline on both clean shifted regimes.
A population arm passes in at least four of five checkpoints. Report the
linear and affine-scalar population gates separately. The primary constructive
claim requires the existing-capacity linear head to pass both the analytic and
learned arms.

Also report target cross-entropy, expected-cosine correlation and RMSE,
weight displacement from `W_0`, and all four clean/noisy transfer cells. These
are descriptive and cannot replace the exact-bin gate.

## Locked classification

| Outcome | Classification | Decision |
| --- | --- | --- |
| provenance, replay, clean-fit control, or shuffled control fails | `invalid` | repair lifecycle only under a new root |
| scalar and linear readouts pass both arms | `scalar_interval_calibration_sufficient` | retain the quotient and deploy a calibrated scalar boundary |
| linear passes both arms but scalar does not | `linear_readout_recalibration_sufficient` | retain the quotient; the answer head needs hidden-state reweighting |
| exactly one arm passes the linear population gate | `arm_stratified_readout_repair` | localize the remaining failure to that front-end/readout pairing |
| noisy utility passes but clean compatibility fails | `condition_specific_readout_tradeoff` | preregister a clean/noisy mixture head; do not claim repair |
| neither linear arm passes despite valid controls | `probe_readout_gap_persists` | the probe-defined quotient is not sufficient for this linear task interface; test a bounded nonlinear ceiling before model training |

No noise level, head width, ridge value, target transform, threshold, or arm
may be changed after primary outcomes are inspected. No TinyLLM or front-end
retraining follows before this frozen readout test is resolved.

## Fixed artifacts

- runner:
  `experiments/structure_net/tinyllm_noisy_reference_readout_recalibration.py`
- tests:
  `tests/structure_net/test_tinyllm_noisy_reference_readout_recalibration.py`
- primary result root:
  `data/experiments/tinyllm_noisy_reference_readout_recalibration/20260807_d8_preregistered_cuda1_persistent_mixed_pedigree`
- report:
  `docs/08 - Analysis/2026-08-07_tinyllm-noisy-reference-readout-recalibration.md`
- meta hypothesis:
  `tinyllm-noisy-reference-readout-recalibration-v1`

## Scope boundary

This is supervised readout calibration on one synthetic two-degree reference-
error distribution and five selected successful checkpoints. It does not
estimate a sub-two-degree boundary, train a reference estimator, repair other
calibration fields, or establish natural-language behavior.
