# TinyLLM C3 constant-acceleration fixed-operator result

**Status:** PROSPECTIVE NO-TRAINING RESULT CONFIRMED

**Date:** 2026-08-11

**Hypothesis:** `tinyllm-c3-constant-acceleration-fixed-operator-v1`

**Classification:** `fixed_all_frame_degree2_closes_constant_acceleration`

**Preregistration:** [C3 constant-acceleration fixed-operator preflight](../07%20-%20Status%20Reports/2026-08-11_tinyllm-c3-constant-acceleration-fixed-operator-preregistration.md)

**Primary artifact:** `data/experiments/tinyllm_c3_constant_acceleration_fixed_operator/20260811_preregistered/result.json`

## Verdict

Changing the calibrated `C3` task from constant speed to constant acceleration
does not create a learned-temporal-model opportunity. The inherited
constant-speed operator fails the fixed ceiling in every seed, confirming that
the dynamics change is material. But the constant-acceleration law closes
under exact degree-2 group differences:

```text
constant-speed fixed ceiling:  0/5 seeds
recent degree-2 fixed ceiling: 5/5 seeds
all-frame degree-2 ceiling:    5/5 seeds
all-frame material dominance: 5/5 seeds
required population:         >=4/5 seeds
```

The all-frame operator roughly halves the remaining quantization error relative
to the last-three-frame degree-2 formula and adds about four exact-bin accuracy
points under both shifts. It uses no model, checkpoint, fit, phase unwrap, or
target information. Constant-acceleration TinyLLM training is rejected before
model construction.

## Group-polynomial operators

For carrier states `q_t`, define group velocity and acceleration:

```text
d_t = q_t conjugate(q_(t-1))
a_t = d_t conjugate(d_(t-1)).
```

Under constant angular acceleration, every `a_t` is the same group element and
the exact recurrence is

```text
q_8 = q_7 d_7 a_7.
```

The recent degree-2 arm uses this formula directly. The all-frame arm first
averages all six acceleration estimates, transports every observed increment
to an estimate of `d_7`, circularly averages those seven estimates, and then
applies the same recurrence. This is the multiplicative-group analogue of a
degree-2 finite-difference extrapolator.

The inherited constant-speed arm averages the seven increments without an
acceleration term. Its failure is the registered misspecification control.

## Population results

Means over five independently generated `4,096`-example cohorts per shift:

| Shift | Operator | Scalar RMSE | Scalar corr | Exact-bin acc | Posterior corr | Cross-entropy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| composition | constant-speed mean | `.152774` | `.976643` | `.229834` | `.976235` | `1.913706` |
| composition | recent degree-2 | `.015278` | `.999766` | `.917432` | `.999403` | `1.285653` |
| composition | all-frame degree-2 | `.007435` | `.999945` | `.959082` | `.999582` | `1.280810` |
| extrapolation | constant-speed mean | `.323810` | `.895260` | `.088721` | `.894899` | `4.130264` |
| extrapolation | recent degree-2 | `.016402` | `.999731` | `.913965` | `.999373` | `1.287257` |
| extrapolation | all-frame degree-2 | `.008006` | `.999936` | `.957764` | `.999576` | `1.281680` |

The all-frame degree-2 estimator reduces scalar RMSE by `51.3%` on composition
and `51.2%` on extrapolation relative to the recent degree-2 estimator. Accuracy
rises by `4.17` and `4.38` percentage points respectively. The
constant-speed operator fails every complete task gate; both degree-2 arms pass
every gate.

## Seedwise registered comparison

| Seed | Comp RMSE ratio | Comp acc delta | Extrap RMSE ratio | Extrap acc delta | Both-shift material gate |
| ---: | ---: | ---: | ---: | ---: | --- |
| 107 | `.4897` | `+.0461` | `.4893` | `+.0430` | pass |
| 127 | `.4814` | `+.0400` | `.4936` | `+.0483` | pass |
| 149 | `.4840` | `+.0457` | `.4828` | `+.0413` | pass |
| 173 | `.4911` | `+.0376` | `.4869` | `+.0430` | pass |
| 197 | `.4873` | `+.0388` | `.4880` | `+.0435` | pass |

Every RMSE ratio is below `.50`, well inside the preregistered `.75` limit.
Every accuracy and cross-entropy comparison also improves.

## Controls and integrity

| Contract | Result | Limit |
| --- | ---: | ---: |
| requested/completed/invalid cells | `10 / 10 / 0` | exact |
| new evaluation examples | `40,960` | registered |
| dataset regeneration | `10/10` exact | exact |
| quantizer saturation | `0` | `0` |
| continuous degree-2 algebra error | `2.502e-14` maximum | `<=1e-12` |
| deck-action prediction error | `5.770e-15` maximum | `<=2e-12` |
| target deck-invariance error | `6.606e-15` maximum | `<=2e-12` |
| group/token action errors | `0` | `0` |
| shuffled-target fixed points | `0` | `0` |
| shuffled absolute correlation | `.03551` maximum | `<=.10` |
| shuffled scalar RMSE | `.98510` minimum | `>=.80` |
| shuffled complete task passes | `0/30` | `0` |
| optimizer steps / changed parameters | `0 / 0` | `0 / 0` |
| models / checkpoints / target fits | `0 / 0 / 0` | `0 / 0 / 0` |

Composition and extrapolation are disjoint in both velocity and acceleration
magnitude. The full campaign regenerated each dataset independently before
accepting its hash.

## What the result establishes

- Nonconstant speed alone is not sufficient reason to introduce a learned
  temporal continuation.
- The relevant structure is the order of the group-valued temporal law:
  constant velocity closes at first differences, while constant acceleration
  closes at second differences.
- Repeated group differences provide useful quantization-noise reduction when
  transported to a common time before circular averaging.
- The constant-speed negative comparator confirms that success is not caused
  by an unchanged easy target.

## Scope boundary

This result covers calibrated, noiseless, constant angular acceleration with
observable exact `C3` action. It does not cover stochastic or piecewise
dynamics, an unknown law order, missing frames, outliers, sensor corruption,
or an approximate group action. It does not prove optimality among all fixed
degree-2 estimators.

No TinyLLM was evaluated. The result is an analytic-ceiling decision about
whether TinyLLM should be trained, not evidence about a trained transformer's
behavior on accelerated sequences.

## Program decision

Promote the all-frame degree-2 group operator as the constant-acceleration
positive control. Do not train TinyLLM on this fixed law, and do not climb a
polynomial-degree ladder: any known finite-order group polynomial admits the
corresponding fixed finite-difference construction.

The next learning-relevant scope must make the law or observations genuinely
uncertain—for example unknown/piecewise dynamics, missing or corrupted frames,
or approximate group actions. It must still begin with identifiability and a
robust fixed-estimator ceiling.

## Reproduction and provenance

```bash
MPLCONFIGDIR=/tmp/mpl-c3-acceleration \
pixi run python -m \
  experiments.structure_net.tinyllm_c3_constant_acceleration_fixed_operator
```

| Artifact | SHA-256 |
| --- | --- |
| primary result | `b04a5574efc658ec1ed73f70fa494041ad16c0ae1342423cdde32925c1c7bc53` |
| runner | `6ea952f386b82b12355c3aa2e9552af6bf73e03e7cd47310fec764ce49d0d5e2` |
| preregistration | `ae4e15e88fc16ec3cd1cc3a52724e042402e80348af6057df5b65b4719c4ee7b` |
| retained generator | `812ac0a574bab812ff9fbbea7a0997c71e8ed0c50f98b270b8344234956ec118` |
| fixed interval decoder | `b027eb55d971b87f598bd600131b33ead82463e3e73766fcdce38fdc3a5497c6` |
| predecessor fixed-operator result | `9b03a0cf19ef820586e2a031d80a694d498655de151f70f245c3c82b0abb853a` |
| frozen trajectory result | `a0ac3315b03aa65df273539a24d8c08f51f12e8ceb702859cc16a282886ddf27` |

The focused runner suite passes `9/9` tests against the authoritative artifact.
