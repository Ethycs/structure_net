# TinyLLM calibration-degradation causal preregistration

**Status:** COMPLETED — PRIMARY ROBUSTNESS GATE FAILED; `exact_calibration_required`  
**Date:** 2026-08-07  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, `UNDERPOWERED`, frozen-checkpoint causal diagnostic  
**Hypothesis:** `tinyllm-calibrated-reference-robustness-curve-v1`  
**Schema:** `nal.tinyllm-calibration-degradation-causal.v1`

**Result:** The frozen confirmatory campaign completed `10/10` checkpoint
evaluations. The analytic robust radius was `0.00` and the learned radius was
`0.05`, below the locked `0.20` target. The measured report is
[TinyLLM calibration-degradation causal test](../08%20-%20Analysis/2026-08-07_tinyllm-calibration-degradation-causal.md).
The final data snapshot is DVC root
`ca2abd03b528233760a5f1cb23686dca.dir`, committed to lakeFS as
`4fca75c78b6ec1f9e83c1548bb15aca79d076de909b9e0dc7547c55223bb816f`.

## Decision question

The calibrated analytic and learned-equivariant TinyLLM front ends passed the
joint composition/extrapolation quotient gate in all five seeds when supplied
an exact phase-independent calibration packet. The frozen-writer repair branch
has since been closed by an exact two-term action-defect result.

The next constructive question is therefore:

> How much error can the gauge-fixing reference tolerate before semantic-base
> retention, branch contraction, or downstream task utility fails—and does the
> learned equivariant encoder approach the analytic control?

No model, front end, or task head is trained or fine-tuned. This study perturbs
only the calibration packet at inference while holding sensor tokens, targets,
checkpoints, examples, and clean-trained diagnostic observers fixed.

## Locked source

Reuse the successful calibrated-front-end campaign:

```text
data/experiments/tinyllm_calibrated_frontend_causal/
    20260806_d8_preregistered/campaign_results.json
SHA-256 80623a7283fed3d902a1cbf9fb58afe2a2fef82ab86406212f907de40a96c501
implementation 73e152b9b6c7ab51ad08e6a80786c32a71566f9d4fb410ff2cf3df53761faf77
```

Use the stored `analytic_calibrated` and
`learned_calibrated_equivariant` systems for seeds `7`, `17`, `29`, `41`, and
`53`. The deterministic raw-file manifest over their 30 result/model/front-end
files is

```text
23598aaef5a0d16825ff9a928de57857e7bd58b10feab8853739448acfd983fc
```

Every loaded model and full system state must also reproduce the state digests
in its source result. The ten condition/checkpoint pairs are the repeated
systems; the five seeds are the replication units.

## Fixed data and observers

Regenerate the exact analysis datasets and seeds used by the source campaign:

| Role | Samples | Seed expression |
| --- | ---: | --- |
| clean observer train | `2,048` | `83 + 101` |
| clean observer validation | `512` | `83 + 211` |
| composition evaluation | `1,024` | `83 + 1,316` |
| extrapolation evaluation | `1,024` | `83 + 2,325` |

Fit one deterministic nonlinear conditional branch/cosine observer at each of
`frontend` and `full` using only clean training-support activations. Fit the
cosine-only conditional null on the same clean split. Evaluate those same
frozen observers at every perturbation level; no observer may adapt to a noise
level or ablation.

Because the predecessor did not store observer weights, deterministic refit is
unavoidable. Clean evaluation metrics must replay the source campaign before
any noisy metric is valid. This is observer fitting, not TinyLLM training, and
is recorded separately.

## Calibration intervention

The clean packet is

```text
(cos orientation, sin orientation, signed speed,
 amplitude, offset_x, offset_y, drift_x, drift_y).
```

Use common random numbers across arms, checkpoints, and noise levels. For one
fixed standard-normal draw per example, apply the following physically typed
errors at scalar level `sigma`:

```text
orientation angle += sigma * 0.40 rad * xi_orientation
signed speed      += sigma * 0.50     * xi_speed
log amplitude     += sigma * 0.50     * xi_amplitude
offset_x/y        += sigma * 0.30     * xi_offset_x/y
drift_x/y         += sigma * 0.16     * xi_drift_x/y
```

Renormalize the orientation vector, exponentiate and clamp amplitude to
`[0.10, 4.0]`, and leave sensor tokens unchanged. The locked curve is

```text
sigma in {0.00, 0.05, 0.10, 0.20, 0.40, 0.80}.
```

The errors are nested: every level scales the same example-specific draw.

Secondary causal ablations replace exactly one calibration group by its neutral
default: orientation `(1,0)`, signed speed `+0.35`, amplitude `1`, offset `0`,
or drift `0`. An `all_default` control replaces the entire packet. Ablations
cannot promote the primary robustness hypothesis.

## Measurements

At `frontend` and `full`, on composition and extrapolation, report:

- cosine correlation;
- conditional branch balanced accuracy;
- conditional branch log-loss gain over the cosine-only null;
- clean-to-perturbed row-cosine similarity;
- normalized activation RMSE; and
- centered linear CKA against the clean activation on the fixed first `128`
  evaluation rows.

At full depth also report exact-bin task accuracy, circular error, and target
cross-entropy. Sensor/input/target hashes must be identical across every
calibration intervention.

## Joint endpoint and robust radius

A representation cell passes only when all three hold:

```text
cosine correlation >= 0.90
conditional branch balanced accuracy <= 0.55
conditional log-loss gain <= 0.02.
```

Task utility passes a shift only when exact-bin accuracy falls by no more than
`0.03` absolute from the same checkpoint/arm's clean value.

A checkpoint passes a noise level only when both representation cuts and task
utility pass on both composition and extrapolation. An arm passes a level in at
least `4/5` seeds. Its robust radius is the largest consecutive `sigma`,
starting at zero, for which every preceding level also passes `4/5`.

The primary hypothesis passes only if:

1. the analytic robust radius is at least `0.20`;
2. the learned robust radius is at least `0.20`;
3. the learned and analytic radii differ by at most one adjacent grid level;
   and
4. `all_default` fails the joint checkpoint gate in at least `4/5` seeds for
   both arms, establishing dynamic range.

No mean can rescue fewer than four seedwise joint passes.

## Fixed classifications

Apply the first matching campaign label:

| Outcome | Classification |
| --- | --- |
| provenance, source-state, data-identity, clean-replay, finite, or observer contract fails | `invalid` |
| complete primary hypothesis passes | `learned_calibration_robustness_tracks_analytic_control` |
| analytic reaches `0.20`, learned does not | `learned_calibration_brittle` |
| both fail before `0.20` | `exact_calibration_required` |
| learned exceeds analytic by more than one level | `learned_more_robust_than_analytic_control` |
| analytic fails before `0.20`, learned does not | `analytic_canonicalizer_brittle` |
| otherwise | `mixed_calibration_robustness` |

## Outcome-directed decisions

| Outcome | Next action |
| --- | --- |
| learned tracks analytic | use the learned front end as the constructive architecture and test naturalistic calibration estimators |
| learned brittle | keep the analytic control and train only a calibration-aware front end with explicit perturbation augmentation |
| exact calibration required | stop model optimization; improve the measurement/reference process first |
| learned more robust | localize which learned invariant suppresses calibration error before changing training |
| analytic brittle only | audit the endpoint/phase-advance formula before any new model |
| invalid | repair only the digital contract under a new root |

No writer sidecar, residual penalty, topology scan, or TinyLLM retraining is
licensed by this experiment.

## Fixed artifacts

- runner: `experiments/structure_net/tinyllm_calibration_degradation_causal.py`
- tests: `tests/structure_net/test_tinyllm_calibration_degradation_causal.py`
- result root: `data/experiments/tinyllm_calibration_degradation_causal/20260807_d8_existing_checkpoints`
- report: `docs/08 - Analysis/2026-08-07_tinyllm-calibration-degradation-causal.md`
- meta hypothesis: `tinyllm-calibrated-reference-robustness-curve-v1`

The runner must preserve strict JSON, source and implementation hashes,
deterministic fingerprints, per-result/activation hashes, exact resume, and
explicit counts of trained models (`0`) and fitted diagnostic observers.

## Method boundaries

The calibration-noise scale is a synthetic, dimensionless stress coordinate;
it is not a calibrated real-instrument error model. Diagnostic observers are
refit on the predecessor's clean split because their weights were not stored.
The source models and exact calibration success were selected after prior
outcomes. Five seeds are adequate for the declared replicated gate but do not
establish architecture-population prevalence.
