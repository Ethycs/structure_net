# TinyLLM C3 temporal sensor-only campaign preregistration

**Status:** FROZEN BEFORE TRAINING

**Date:** 2026-08-11

**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED / FIVE-SEED CAUSAL TRAINING`

**Hypothesis:** `tinyllm-c3-temporal-sensor-only-v1`

## Decision question

The registered function-class preflight proved that the existing 184-parameter
exact-`C3` sensor contains the analytic carrier and receives a usable true-task
gradient through the fixed temporal operator and decoder. The remaining
question is optimization and out-of-support acquisition:

> From ordinary random initialization and task loss alone, does the existing
> exact-`C3` sensor learn a task-equivalent invariant carrier that remains useful
> under both composition and outside-speed extrapolation?

This campaign does not instantiate or train TinyLLM. It deliberately removes
the already-falsified learned continuation/readout from the experiment.

## Fixed sources

| Source | SHA-256 |
| --- | --- |
| licensed function-class result | `6a01db25ebc2ed15d202884c39f16db685d5218647b0bb209e2e5a737696a383` |
| function-class runner | `95331fa823f193449af1c46f961f22f4aaf902300695f179912b75d8ed4bcade` |
| existing learned `C3` sensor family | `dbc934190b5f725185ff9a99690409ac63fc4f16bd04686f870e7ef21cba03a6` |
| observable `C3` generator and analytic carrier | `812ac0a574bab812ff9fbbea7a0997c71e8ed0c50f98b270b8344234956ec118` |
| fixed interval likelihood | `b027eb55d971b87f598bd600131b33ead82463e3e73766fcdce38fdc3a5497c6` |

Any source mismatch invalidates the campaign.

## Arms and replication unit

Use seeds `(7, 17, 29, 41, 53)`. One seed is one independently schedulable
replicate containing two sequential matched arms:

| Arm | Sensor initialization | Training target | Purpose |
| --- | --- | --- | --- |
| `learned_true` | seed-matched random state | true fixed interval posterior | primary learned arm |
| `learned_target_shuffled` | byte-identical initial state | pair-preserving target derangement | specificity control |

The analytic carrier is evaluated as the no-training positive control on every
held-out cohort. It is not counted as a learned arm and is never used to
initialize either learned sensor.

## Locked training protocol

- sensor: unmodified `LearnedC3InvariantEncoder(hidden=16,
  character_channels=8)`, exactly 184 parameters;
- trainable parameters: sensor only;
- fixed downstream map:

  ```text
  q_next = q(7) * conjugate(q(6)) * q(7)
  scalar = real(q_next)
  posterior = fixed_interval_posterior(scalar, 16).
  ```

- objective: target soft cross-entropy only;
- examples: 4,096 paired observations per seed from the registered composition
  generator, using seed `seed + 1001`;
- minibatches: 600 fixed paired minibatches of size 64 using seed
  `seed + 6013`;
- optimizer: AdamW, learning rate `3e-4`, weight decay `.01`;
- gradient clipping: global norm `1.0`;
- no warm start, carrier loss, auxiliary head, scheduler, early stopping, loss
  weight, or hyperparameter sweep.

The shuffled arm uses one deterministic permutation of latent pairs and keeps
the two deck sheets of each permuted latent together. Initialization, observed
examples, calibration packets, minibatches, optimizer, and step count remain
matched.

## Held-out cohorts

Training, the function-class preflight, gauge fitting, and primary evaluation
must use disjoint seeds.

| Role | Regime | Seed | Examples |
| --- | --- | ---: | ---: |
| carrier-gauge reference | composition | `431003` | 1,024 |
| primary test | composition | `331003` | 1,024 |
| primary test | extrapolation | `331021` | 1,024 |

All cohorts use observed calibration only. Analytic carriers and targets are
used for evaluation, not for learned-arm optimization.

## Carrier gauge measurement

The task is unchanged under global complex conjugation of the carrier. Fit one
two-dimensional orthogonal Procrustes map from each learned sensor to the
analytic carrier on the disjoint reference cohort. Apply that same fixed map
without refitting to both primary test cohorts.

At each shift report:

- mean aligned unit-vector dot product;
- aligned coordinate RMSE;
- fitted determinant and reference residual;
- direct unaligned carrier error as a descriptive measurement.

Carrier fidelity passes when the same reference-fitted map obtains mean dot
product at least `.90` and coordinate RMSE at most `.35` on both primary
shifts. This permits only one global `O(2)` carrier gauge; it does not permit a
shift-specific repair.

## Primary per-seed gate

For `learned_true`, a seed passes only if all conditions hold:

1. source hashes, paired-data contract, target derangement, parameter count,
   finiteness, checkpoint reload, and exact-resume contracts pass;
2. the sensor state changes from initialization;
3. maximum output change under either nonidentity deck action is at most
   `2e-6` on both primary shifts;
4. the reference-fitted carrier gate passes on both primary shifts;
5. the complete fixed-decoder task gate passes simultaneously:

   | Endpoint | Composition | Extrapolation |
   | --- | ---: | ---: |
   | posterior-mean correlation | `>= .90` | `>= .90` |
   | exact-bin accuracy | `>= .50` | `>= .35` |
   | target cross-entropy | `<= 1.80` | `<= 2.20` |
   | predicted-bin coverage | `>= 14` | `>= 12` |

The analytic positive control must pass all task and carrier gates before any
learned result is interpretable.

## Population decision

Success requires:

```text
learned_true joint passes >= 4/5
and
learned_target_shuffled joint passes <= 1/5.
```

The shuffled joint gate uses the identical action, carrier, task, validity, and
reload criteria against its true held-out targets. The control is expected to
fail the task and carrier portions; structural invariance alone is not a
success.

## Locked classifications

| Outcome | Classification | Decision |
| --- | --- | --- |
| analytic control passes; true `>=4/5`; shuffled `<=1/5` | `task_only_sensor_acquisition_supported` | the quotient can be acquired without a learned transformer continuation; replicate on a new task/group before reintegration |
| analytic control passes; true `<4/5`; shuffled `<=1/5` | `function_class_present_but_task_only_sensor_optimization_unreliable` | stop task-only optimization; do not add losses until the failure geometry is localized artifact-only |
| shuffled `>1/5` | `sensor_acquisition_specificity_failed` | treat any true success as nonspecific |
| analytic/source/lifecycle contract fails | `invalid_campaign` | repair systems only; draw no scientific conclusion |

No threshold or optimizer setting may be changed after primary outcomes are
read.

## Shakedown and stop rules

Before primary launch:

1. focused CPU tests must cover pairing, shuffling, exact `C3` action, gauge
   fitting, joint gates, fingerprints, strict JSON, and aggregation;
2. a two-step CPU exact-resume lifecycle must reproduce sensor state, optimizer
   state, loss history, and held-out posterior exactly;
3. a two-step one-seed CUDA lifecycle must save/reload both learned arms and
   remain explicitly underpowered systems evidence.

If either lifecycle fails, do not launch the primary campaign. The primary
campaign runs all five seeds once; failed infrastructure cells may be retried
only with the identical scientific fingerprint.

Expected artifact root:

```text
data/experiments/tinyllm_c3_temporal_sensor_only/
  20260811_preregistered/
```
