# TinyLLM C3 temporal fixed-operator ceiling result

**Status:** PROSPECTIVE NO-TRAINING RESULT CONFIRMED

**Date:** 2026-08-11

**Hypothesis:** `tinyllm-c3-temporal-fixed-operator-ceiling-v1`

**Classification:** `fixed_multistep_group_operator_dominates_last_step`

**Preregistration:** [C3 temporal fixed-operator ceiling](../07%20-%20Status%20Reports/2026-08-11_tinyllm-c3-temporal-fixed-operator-ceiling-preregistration.md)

**Primary artifact:** `data/experiments/tinyllm_c3_temporal_fixed_operator_ceiling/20260811_preregistered/result.json`

## Verdict

The noiseless `C3` sequence contains useful temporal redundancy beyond the
registered last-two-frame operator, but exploiting it does not require
TinyLLM. One fixed all-frame circular mean of the seven exact group increments
materially improves every composition and extrapolation cell in all five new
replicates.

```text
all-increment material dominance: 5/5 seeds
last-increment declared ceiling:   5/5 seeds
required material population:    >=4/5 seeds

classification: fixed_multistep_group_operator_dominates_last_step
```

The all-frame operator roughly halves scalar error and adds about two exact-bin
accuracy points under both shifts. It has no fitted coefficient, target access,
checkpoint, trainable parameter, or model. The current same-task TinyLLM
continuation therefore has a known cheaper replacement and is not licensed for
retraining.

## Operators

The inherited baseline estimates the next group state from only the last
increment:

```text
d_7 = q_7 * conjugate(q_6)
u_last = Re(q_7 * d_7).
```

The sole new arm averages all seven adjacent increments on the unit circle:

```text
d_t = q_t * conjugate(q_(t-1))
d_mean = sum_t d_t / |sum_t d_t|
u_mean = Re(q_7 * d_mean).
```

For the exact constant-speed carrier, both formulas reproduce the target to
maximum error `1.050e-14`. With quantized observations, the circular mean
reduces independent frame-level phase error while preserving the exact `C3`
action.

## Population result

Means over five independently generated `4,096`-example cohorts per shift:

| Shift | Operator | Scalar RMSE | Scalar corr | Exact-bin acc | Posterior corr | Cross-entropy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| composition | last increment | `.007675` | `.999941` | `.95874` | `.999583` | `1.28032` |
| composition | all increments | `.003946` | `.999985` | `.97783` | `.999625` | `1.27914` |
| extrapolation | last increment | `.008238` | `.999932` | `.95981` | `.999570` | `1.28119` |
| extrapolation | all increments | `.004254` | `.999982` | `.98037` | `.999621` | `1.27984` |

The mean scalar-RMSE reductions are `48.59%` on composition and `48.36%` on
extrapolation. Mean exact-bin accuracy rises by `1.91` and `2.06` percentage
points respectively. Cross-entropy improves in every seed and shift; the mean
changes are `-.00118` and `-.00135`.

## Seedwise registered comparison

| Seed | Comp RMSE ratio | Comp acc delta | Extrap RMSE ratio | Extrap acc delta | Both-shift material gate |
| ---: | ---: | ---: | ---: | ---: | --- |
| 7 | `.5120` | `+.0188` | `.5173` | `+.0217` | pass |
| 17 | `.5205` | `+.0166` | `.5068` | `+.0198` | pass |
| 29 | `.5148` | `+.0227` | `.5194` | `+.0200` | pass |
| 41 | `.5084` | `+.0168` | `.5145` | `+.0210` | pass |
| 53 | `.5152` | `+.0205` | `.5246` | `+.0203` | pass |

Every ratio is far below the registered `.75` limit. No cell trades away
accuracy or cross-entropy to obtain its scalar improvement.

## What TinyLLM failed to do

The earlier analytic-sensor TinyLLM population had exact invariant input and
causal closure in `5/5` seeds but natural utility in only `2/5`. The fixed
last-step bypass then reached about `.959` extrapolation accuracy, and the
sensor-only learned system reproduced that path in `5/5` seeds.

This experiment shows that even that positive bypass was leaving an elementary
calculation unused. The full eight-frame observation contains seven noisy
copies of the same constant group increment. Averaging them is the natural
symmetry-typed estimator, yet the unrestricted trained continuation did not
reliably recover a comparable support-stable temporal computation.

The localization is now sharper:

```text
exact C3 sensor: sufficient and learnable
all-frame temporal group statistic: available and analytically useful
fixed metric decoder: sufficient
unrestricted TinyLLM continuation: failed to exploit the redundancy reliably.
```

This is not evidence that a transformer is mathematically incapable of circular
averaging. It is evidence that paying for one here has no demonstrated value
when the required operator is known, exact, cheaper, and more accurate.

## Controls and integrity

| Contract | Result | Limit |
| --- | ---: | ---: |
| requested/completed/invalid cells | `10 / 10 / 0` | exact |
| new evaluation examples | `40,960` | registered |
| quantizer saturation | `0` | `0` |
| continuous algebra error | `1.050e-14` maximum | `<=1e-12` |
| deck-action prediction error | `2.776e-15` maximum | `<=2e-12` |
| shuffled-target absolute correlation | `.0360` maximum | `<=.10` |
| shuffled-target RMSE | above `.80` in every cell | `>=.80` |
| deterministic dataset hashes | `10/10` | exact |
| optimizer steps / changed parameters | `0 / 0` | `0 / 0` |
| checkpoints / TinyLLM instances / target fits | `0 / 0 / 0` | `0 / 0 / 0` |

Both operators also pass the inherited complete task floor in every cell. The
baseline ceiling count is `5/5` because its scalar RMSE is already below `.01`;
the preregistered classification gives precedence to material all-frame
dominance, which also passes `5/5`.

## Scope boundary

This result establishes a better fixed operator for the declared constant-speed,
calibrated, noiseless generator. It does not establish:

- optimality among all possible analytic temporal estimators;
- robustness to changing speed, missing frames, outliers, sensor noise, or
  approximate group actions;
- TinyLLM usefulness on a task where the temporal law is unknown;
- language-model behavior or transfer to a different group;
- that the old TinyLLM checkpoints internally represent the circular mean.

No alternative weighting, phase unwrapping, robust mean, subset, or learned
coefficient was inspected. The single preregistered all-increment operator is
the complete candidate family for this result.

## Program decision

Promote the all-increment circular mean to the positive-control temporal
baseline and close same-task TinyLLM continuation training. The registered
typed-residual function-class branch is not licensed because the fixed operator
already dominates in `5/5` seeds.

Learning becomes scientifically justified only when the temporal estimation
problem changes—for example nonconstant dynamics, missing/corrupted frames, or
an unknown group law—and only after the corresponding analytic and
identifiability preflight. Another optimizer, continuation width, seed, loss,
or fixed-operator sweep on the current generator would not test a new claim.

## Reproduction and provenance

```bash
MPLCONFIGDIR=/tmp/mpl-c3-fixed-operator \
pixi run python -m \
  experiments.structure_net.tinyllm_c3_temporal_fixed_operator_ceiling
```

| Artifact | SHA-256 |
| --- | --- |
| primary result | `9b03a0cf19ef820586e2a031d80a694d498655de151f70f245c3c82b0abb853a` |
| runner | `9471ffe7f319d3d53234fd795fc48ce39a7717970d0e191ae2882343ad6d3b37` |
| preregistration | `07d54cb5b4d65b080fe59a5e06d0853ee59a7d0132fc04faa68f303775a2f8bf` |
| generator | `812ac0a574bab812ff9fbbea7a0997c71e8ed0c50f98b270b8344234956ec118` |
| interval likelihood | `b027eb55d971b87f598bd600131b33ead82463e3e73766fcdce38fdc3a5497c6` |
| source sensor-only campaign | `4d023d9f9d77f64b1fe75970acd63461cfd5a64aa1de60c7954a2c671c427012` |
| source affine mechanism result | `c3dbfecd7a6381c2129e4d99f135557f003ad8a225fa2b4d3f4fa0cb429f669b` |

The focused runner suite passes `7/7` tests against the sealed artifact.
