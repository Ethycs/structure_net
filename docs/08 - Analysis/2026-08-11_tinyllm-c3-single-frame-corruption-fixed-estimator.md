# TinyLLM C3 single-frame corruption fixed-estimator result

**Status:** PROSPECTIVE NO-TRAINING RESULT CONFIRMED

**Date:** 2026-08-11

**Hypothesis:** `tinyllm-c3-single-frame-corruption-fixed-estimator-v1`

**Classification:** `fixed_robust_estimator_closes_single_frame_corruption`

**Preregistration:** [C3 single-frame corruption fixed-estimator preflight](../07%20-%20Status%20Reports/2026-08-11_tinyllm-c3-single-frame-corruption-fixed-estimator-preregistration.md)

**Primary artifact:** `data/experiments/tinyllm_c3_single_frame_corruption_fixed_estimator/20260811_preregistered/result.json`

## Verdict

One unmarked gross frame substitution is a real observation shift, but it does
not create a TinyLLM opportunity under the known constant-acceleration law. The
unprotected all-frame group operator collapses in every seed. An exhaustive,
target-free leave-one-out quadratic estimator identifies the inconsistent frame
and recovers the future state at essentially the oracle ceiling:

```text
clean fixed ceiling:             5/5 seeds
corrupted naive fixed ceiling:   0/5 seeds
oracle deletion fixed ceiling:   5/5 seeds
robust deletion fixed ceiling:   5/5 seeds
corruption material:             5/5 seeds
robust material repair:          5/5 seeds
robust/oracle fidelity:          5/5 seeds
required population:           >=4/5 seeds
```

No model, checkpoint, optimizer step, reusable fitted parameter, or target-using
fit was involved. TinyLLM training and a learned robust-sensor comparison are
both rejected for this declared corruption law.

## Intervention and estimator

Each of the `40,960` matched constant-acceleration sequences receives exactly
one unmarked corruption. All three quantized channels at a uniformly sampled
time are replaced by the same-time frame from a deranged donor in the same
shift. This preserves framewise token marginals while breaking the receiving
trajectory.

For every possible deleted frame, the robust estimator:

1. unwraps the other seven invariant-carrier phases in time order;
2. applies the fixed pseudoinverse for `1`, `t`, and `t(t-1)/2`;
3. scores retained phase residual;
4. chooses the minimum-residual deletion and evaluates the quadratic at time
   `8`.

The oracle uses the identical estimator with the true corruption index. Both
arms are target-free per-example state estimators, not dataset-trained models.

## Population results

Means over five independently corrupted `4,096`-example cohorts per shift:

| Shift | Arm | Scalar RMSE | Scalar corr | Exact-bin acc | Posterior corr | Cross-entropy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| composition | clean all-frame degree 2 | `.007435` | `.999945` | `.959082` | `.999582` | `1.280810` |
| composition | corrupted all-frame degree 2 | `.753275` | `.431537` | `.183887` | `.431295` | `9.922470` |
| composition | oracle delete-one quadratic | `.005723` | `.999967` | `.968701` | `.999603` | `1.280197` |
| composition | robust delete-one quadratic | `.005789` | `.999966` | `.968359` | `.999602` | `1.280218` |
| extrapolation | clean all-frame degree 2 | `.008006` | `.999936` | `.957764` | `.999576` | `1.281680` |
| extrapolation | corrupted all-frame degree 2 | `.748548` | `.440132` | `.184766` | `.440023` | `9.758924` |
| extrapolation | oracle delete-one quadratic | `.006044` | `.999964` | `.969922` | `.999602` | `1.280928` |
| extrapolation | robust delete-one quadratic | `.006072` | `.999963` | `.969775` | `.999602` | `1.280937` |

The naive corrupted operator loses about `77.5` exact-bin accuracy points.
Robust deletion recovers those points and reduces corrupted-naive RMSE by more
than `99.1%` in both shifts. Its mean RMSE excess over the oracle is only
`6.65e-5` on composition and `2.79e-5` on extrapolation.

The clean positive control is slightly worse than the seven-frame quadratic
fit because it is a different quantization-noise estimator; that comparison is
descriptive. The causal conclusion depends on the preregistered corrupted-naive,
oracle, and robust arms.

## Seedwise registered endpoint

| Seed | Comp robust RMSE / acc | Extrap robust RMSE / acc | Comp index recovery | Extrap index recovery | Joint repair/fidelity |
| ---: | ---: | ---: | ---: | ---: | --- |
| 107 | `.005891 / .9683` | `.006054 / .9658` | `.9980` | `.9944` | pass / pass |
| 127 | `.005825 / .9678` | `.006143 / .9722` | `.9944` | `.9966` | pass / pass |
| 149 | `.005875 / .9722` | `.006042 / .9670` | `.9968` | `.9958` | pass / pass |
| 173 | `.005693 / .9690` | `.005971 / .9717` | `.9946` | `.9932` | pass / pass |
| 197 | `.005662 / .9646` | `.006148 / .9722` | `.9956` | `.9963` | pass / pass |

Quantized corruption-index recovery averages `.9959` on composition and `.9953`
on extrapolation. The few index mismatches are benign: prediction remains
oracle-faithful because an alternative deletion can fit an effectively
trajectory-compatible donor frame.

## Controls and integrity

| Contract | Result | Limit |
| --- | ---: | ---: |
| requested/completed/invalid cells | `10 / 10 / 0` | exact |
| matched base hash replay | `10/10` | exact |
| new corrupted evaluations | `40,960` | registered |
| donor fixed points | `0` | `0` |
| minimum examples at any frame index | `473` | `>=400` |
| corruption regeneration | `10/10` exact | exact |
| corruption/deck equivariance token errors | `0` | `0` |
| maximum operator deck-action error | `6.125e-14` | `<=2e-12` |
| minimum clean phase-chart margin | `.333877` | `>=.20` |
| continuous oracle/robust prediction error | `2.736e-14` maximum | `<=1e-10` |
| continuous hidden-index recovery | `40,960 / 40,960` | exact |
| shuffled-target fixed points | `0` | `0` |
| shuffled absolute scalar correlation | `.02821` maximum | `<=.10` |
| shuffled scalar RMSE | `.98450` minimum | `>=.80` |
| shuffled complete task passes | `0/40` | `0` |
| models / checkpoints / optimizer steps | `0 / 0 / 0` | `0 / 0 / 0` |
| changed parameters / target-using fits | `0 / 0` | `0 / 0` |

All five frame-position histograms passed prospectively in both shifts. The
base datasets reproduce the exact predecessor hashes, so the result isolates
the new corruption intervention rather than a generator change.

## What the result establishes

- A single unmarked marginally matched frame substitution is severe enough to
  invalidate the ordinary all-frame finite-difference operator.
- Known law order plus temporal redundancy makes that corruption identifiable
  without labels or learning.
- Exhaustive robust state estimation can restore the exact group-polynomial
  solution under outside-range extrapolation.
- A failed naive analytic operator is not, by itself, evidence for a learned
  model; the robust fixed ceiling must be exhausted first.

## Scope boundary

The result assumes exactly one substituted frame, eight observations, known
constant acceleration, calibrated sensors, exact observable `C3` action, and a
phase-chart margin that prevents clean gap aliasing. It does not cover:

- an unknown number of corruptions;
- corruption of calibration or the corruption process itself;
- missingness coupled to the latent state;
- unknown, switching, or stochastic temporal laws;
- approximate group actions;
- learned-model behavior.

The estimator also uses the declared quadratic law. It should not be described
as a general outlier-robust sequence model.

## Program decision

Promote robust leave-one-out quadratic estimation as the single-frame
corruption positive control. Do not train TinyLLM on this task and do not climb
a corruption-count ladder under the same known law merely to manufacture a
failure.

The next learning-relevant preflight should combine uncertainty in the law with
uncertainty in the observations—for example a within-sequence mixture of
constant speed and constant acceleration with an unmarked corruption. First
compare a fixed adaptive model-selection estimator against an oracle-law arm.
Only an oracle-positive, adaptive-fixed-negative result would license a compact
learned sensor or dynamics selector.

## Reproduction and provenance

```bash
MPLCONFIGDIR=/tmp/mpl-c3-corruption \
pixi run python -m \
  experiments.structure_net.tinyllm_c3_single_frame_corruption_fixed_estimator
```

| Artifact | SHA-256 |
| --- | --- |
| primary result | `59681f2764b988f05b0916965898b87d5b233b2165151fe19e0c97391fe467b9` |
| runner | `8600081fc813ae6da5a49c39222bf3a94286127d7238899d61ec9acd1c6d31f8` |
| preregistration | `29b47e522fc46d5993cfba811eeba2d6bd2cc8eca189a1f6205c422f159b572e` |
| predecessor result | `b04a5574efc658ec1ed73f70fa494041ad16c0ae1342423cdde32925c1c7bc53` |
| predecessor runner | `6ea952f386b82b12355c3aa2e9552af6bf73e03e7cd47310fec764ce49d0d5e2` |
| retained generator | `812ac0a574bab812ff9fbbea7a0997c71e8ed0c50f98b270b8344234956ec118` |
| physical interval decoder | `b027eb55d971b87f598bd600131b33ead82463e3e73766fcdce38fdc3a5497c6` |

The focused runner suite passes against the authoritative artifact. The report
hash is recorded by the meta-hypothesis evidence module after this file is
sealed.
