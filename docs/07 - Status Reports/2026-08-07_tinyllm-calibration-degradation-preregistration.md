# TinyLLM calibration-degradation breakpoint preregistration

**Status:** COMPLETED — SUPPORTED REPRESENTATION-ONLY BREAKPOINTS; END-TO-END ROBUSTNESS NOT CLAIMED  
**Date:** 2026-08-07  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, sequential frozen-checkpoint robustness diagnostic  
**Hypothesis:** `tinyllm-calibration-noise-breakpoint-v1`  
**Schema:** `nal.tinyllm-calibration-degradation.v1`

**Post-outcome disposition (2026-08-07):** The locked campaign completed all
`10/10` checkpoint cells and supported bounded stable representation
breakpoints in both arms: analytic `[1, 2]` and learned `[0.5, 1]`. Campaign
SHA-256 is
`87a556e61db4a584b9cd423af9cb9d663f3f0225757a31a69a7158788137ef86`.
The result and its endpoint boundary are reported in the
[calibration-degradation analysis](../08%20-%20Analysis/2026-08-07_tinyllm-calibration-degradation.md)
and stored under meta hypothesis `tinyllm-calibration-noise-breakpoint-v1`.
The methods below remain the original preregistration. Later task-inclusive
campaigns do not invalidate this result; they show that its probe-defined
representation endpoint is not an end-to-end task-robustness endpoint.

## Question and prediction

How much error in an observed gauge-fixing reference can the already trained
analytic and learned calibrated TinyLLM front ends tolerate before the joint
cosine-retention/branch-contraction endpoint fails?

The directional prediction is a bounded, ordered robustness transition:

1. both arms replay their exact-calibration source gate at zero corruption;
2. both tolerate at least one nonzero registered corruption level;
3. both fail by the maximum registered level; and
4. neither arm re-enters the campaign gate after its first failure.

This study identifies a grid interval, not a universal noise constant. It makes
no preregistered superiority claim between the learned and analytic arms.

## Locked predecessor

Reuse the completed five-seed calibrated-identifiability campaign:

```text
data/experiments/tinyllm_calibrated_frontend_causal/
    20260806_d8_preregistered/campaign_results.json
SHA-256 80623a7283fed3d902a1cbf9fb58afe2a2fef82ab86406212f907de40a96c501
implementation 73e152b9b6c7ab51ad08e6a80786c32a71566f9d4fb410ff2cf3df53761faf77
```

The source completed all `15/15` cells. The analytic and learned structured
arms passed the joint quotient gate in `5/5` seeds; the raw arm passed `0/5`.
This follow-up therefore reuses only:

- `analytic_calibrated`;
- `learned_calibrated_equivariant`;
- seeds `7`, `17`, `29`, `41`, and `53`.

Every source result, model checkpoint, and front-end checkpoint must exist and
replay its stored state digest before evaluation. The source campaign and
producing implementation hashes are hard gates. No TinyLLM or front end is
trained or updated.

## Intervention and fixed controls

Only the observed eight-channel calibration packet changes at inference:

```text
(cos orientation, sin orientation, signed speed,
 amplitude, offset_x, offset_y, drift_x, drift_y).
```

The quantized sensor history, target, architecture, model weights, front-end
weights, evaluation examples, probe split sizes, probe family, probe seeds,
representation cuts, and task decoder remain fixed.

For a dimensionless level `q`, draw one locked standard-normal field per split
and example, shared across both arms and all model seeds. Scale that same field
at every level so the curve is nested. At `q = 1`, use:

| Calibration field | Corruption |
| --- | --- |
| orientation | additive angular error, standard deviation `0.15` radians |
| speed magnitude | multiplicative log-normal error, log standard deviation `log(1.25)` |
| amplitude | multiplicative log-normal error, log standard deviation `log(1.25)` |
| planar offset | additive per-axis error, standard deviation `0.10` sensor units |
| planar drift | additive per-axis error, standard deviation `0.16` sensor units |

Speed sign and positive amplitude are preserved. The orientation pair is
reconstructed as a unit vector after angular corruption. Zero corruption must
be bitwise equal to the source calibration tensor.

The fixed levels are:

```text
q = 0, 0.125, 0.25, 0.5, 1, 2, 4.
```

A deterministic whole-example calibration permutation within each split is a
non-ordered negative control. It is not part of the breakpoint curve.

## Data and estimators

Regenerate the exact source evaluation families and seeds:

| Split | Samples | Generator seed | Regime |
| --- | ---: | ---: | --- |
| probe train | `2048` | `184` | interpolation |
| probe validation | `512` | `294` | interpolation |
| in distribution | `1024` | `390` | interpolation |
| composition | `1024` | `1399` | composition |
| extrapolation | `1024` | `2408` | extrapolation |

The source analysis seed is `83`. Corruption streams are independent of all
generator and probe streams and are identified by stored tensor hashes.

At every level, fit fresh nonlinear conditional branch probes on the frozen
front-end and full-depth representations, using the source protocol:

- `2048/512/1024` train/validation/test examples;
- width `128`, at most `240` steps;
- validation-only early stopping;
- an explicit cosine-only nonlinear null;
- deterministic probe seeds matched across corruption levels.

Probe fitting is measurement, not model training. Conditional branch results
mean recoverability by this declared estimator on these generated families.

## Primary endpoint and breakpoint

At `frontend` and `full`, on both composition and extrapolation, a seed-level
cell passes when the same representation satisfies:

```text
cosine Pearson correlation >= 0.90
conditional branch balanced accuracy <= 0.55.
```

Conditional log-loss gain is retained as a secondary diagnostic to preserve
continuity with the source campaign. It cannot rescue or veto the registered
two-dimensional endpoint.

For an arm and level, a seed passes only if all four primary cells pass. The
level passes when at least four of five seeds pass jointly. Define:

- `first_failed_level`: the smallest registered level after zero with fewer
  than four joint seed passes;
- `last_passing_level`: the greatest lower registered level that passes;
- the breakpoint interval as
  `[last_passing_level, first_failed_level]`.

A bounded stable breakpoint requires:

1. zero corruption passes in at least four of five seeds;
2. at least one nonzero level passes;
3. level `4` fails;
4. no higher level passes after the first failed level; and
5. the shuffled-calibration control fails in at least four of five seeds.

The hypothesis is supported only if both structured arms have bounded stable
breakpoints. Checkpoints are the replication units; noise levels and cells are
repeated measurements.

## Validity and replay gates

The campaign is invalid if any of the following fails:

1. source campaign schema, hypothesis, status, implementation, campaign hash,
   completion count, or original structured-arm gate;
2. any requested source result/checkpoint/front-end identity or stored model
   and system state digest;
3. regenerated source task configuration or evaluation data identity;
4. exact zero-level calibration equality;
5. zero-level task and primary endpoint replay against the source record within
   `1e-6` absolute error;
6. finite representations, task probabilities, and probe metrics;
7. strict JSON, scientific fingerprints, per-result hashes, and immutable
   resume; or
8. shuffled-control construction and permutation identity.

A representative systems-only run may expose quality fields but cannot enter
the primary aggregate. No metric, threshold, level, seed, or interpretation may
change after that lifecycle run without a dated amendment and a new root.

## Secondary measurements

Report at every level, cut, regime, arm, and seed:

- cosine correlation;
- conditional branch balanced accuracy and conditional log-loss gain;
- exact-bin accuracy, mean circular error, and target cross-entropy;
- analytic/learned mean and seed range;
- whether full depth improves or worsens the corrupted front-end correlation;
- empirical RMS error for each corrupted calibration field.

Task degradation is expected and is not an alternate success path. In-
distribution behavior is a control, not part of the breakpoint gate.

## Outcome meanings and stop rules

| Outcome | Interpretation | Next action |
| --- | --- | --- |
| both bounded stable | the gauge-repaired quotient has a measurable finite robustness interval in both mechanisms | localize field sensitivity only if engineering calibration budgets require it |
| analytic stable, learned earlier or nonmonotone | the learned invariant family is more calibration-fragile than the structural control | improve pilot/encoder robustness without reopening residual penalties |
| learned stable beyond analytic | learned temporal aggregation supplies additional reference-error tolerance | verify on a fresh corruption family before claiming superiority |
| maximum level still passes | breakpoint is right-censored | preregister a wider physical range; do not call it unbounded |
| zero replay or shuffled control fails | invalid measurement/provenance path | repair digitally under a new root |
| neither tolerates nonzero noise | exact gauge repair is brittle on this scale | prioritize measured pilot quality over model optimization |
| gate re-entry | grid/probe instability prevents a single breakpoint claim | report the full curve and stop threshold interpretation |

No topology scan, link-cobordism scan, residual penalty, writer sidecar, or new
TinyLLM training is justified by this campaign.

## Artifacts and execution

- runner:
  `experiments/structure_net/tinyllm_calibration_degradation.py`
- tests:
  `tests/structure_net/test_tinyllm_calibration_degradation.py`
- primary root:
  `data/experiments/tinyllm_calibration_degradation/20260807_d8_preregistered`
- systems-only root:
  `data/experiments/tinyllm_calibration_degradation/20260807_shakedown_seed7`
- report:
  `docs/08 - Analysis/2026-08-07_tinyllm-calibration-degradation.md`
- meta hypothesis:
  `tinyllm-calibration-noise-breakpoint-v1`

Planned primary command:

```bash
MPLCONFIGDIR=/tmp/matplotlib-calibration-degradation \
pixi run python -m experiments.structure_net.tinyllm_calibration_degradation \
  --device cuda:1 \
  --output \
  data/experiments/tinyllm_calibration_degradation/20260807_d8_preregistered
```

## Method boundaries

This is a sequential robustness study on checkpoints selected because their
exact calibrated quotient already passed. It estimates conditional robustness
of those mechanisms, not population prevalence or a new independent
replication of gauge repair. The curve corrupts metadata at inference after
exact-calibration training; it does not test training with noisy calibration,
biased calibration, temporal drift in calibration quality, adversarial error,
or learned pilot estimation. Gaussian scale choices are tied to the declared
synthetic nuisance ranges and are not a physical instrument specification.
