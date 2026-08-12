# TinyLLM C3 hidden-order corruption fixed-estimator result

**Status:** PROSPECTIVE NO-TRAINING RESULT INCONCLUSIVE

**Date:** 2026-08-11

**Hypothesis:** `tinyllm-c3-hidden-order-corruption-fixed-estimator-v1`

**Classification:** `inconclusive_hidden_order_corruption_preflight`

**Preregistration:** [C3 hidden-order corruption fixed-estimator preflight](../07%20-%20Status%20Reports/2026-08-11_tinyllm-c3-hidden-order-corruption-fixed-estimator-preregistration.md)

**Primary artifact:** `data/experiments/tinyllm_c3_hidden_order_corruption_fixed_estimator/20260811_preregistered/result.json`

## Verdict

The common robust quadratic estimator clears the strong absolute endpoint on
constant speed, constant acceleration, and their actual 50:50 mixture under
both shifts in every paired replicate. It uses no law label or adaptive
selector and remains oracle-faithful in `5/5` replicates.

The full preregistered closure claim is nevertheless **not confirmed**. The
joint material-repair gate required at least a `.20` accuracy improvement in
every population. Constant-speed corruption left naive accuracy near `.806`,
so restoring it to about `.972` improved accuracy by only `.156--.191` even
though RMSE fell by more than `98.3%`, cross-entropy improved materially, and
the strong absolute ceiling passed. That one effect-size clause makes the
all-population repair count `0/5`.

```text
common robust absolute endpoint:       5/5
oracle absolute endpoint:              5/5
corruption materiality:                5/5
oracle fidelity:                       5/5
full all-population repair gate:       0/5
corrupted naive mixture ceiling:       0/5
```

The registered verdict is therefore inconclusive, not positive. Neither a
typed selector comparison nor TinyLLM training is licensed.

## Why the law selector was unnecessary in the observed computation

For both law families,

```text
arg(q_t) = beta_0 + beta_1 t + beta_2 t(t-1)/2.
```

Constant speed is the exact `beta_2=0` subfamily. The same frozen exhaustive
delete-one quadratic estimator was applied to every sequence without reading
the law identity. Continuous inclusion error was at most `1.277e-15`, and the
common estimator recovered all `81,920` continuous corruption indices and
future states to numerical precision.

Thus the proposed hidden degree-1/degree-2 label did not create a new function
class. This is strong mechanistic evidence, but the report keeps it separate
from the failed preregistered joint classification.

## Population measurements

Means over five paired cohorts per shift. `Clean` uses the registered
law-specific positive control; `naive` applies that operator to the corrupted
trajectory; `robust` is the single unlabelled quadratic estimator.

| Shift | Population | Clean RMSE / acc | Naive RMSE / acc | Robust RMSE / acc | Robust corr | Robust CE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| composition | constant speed | `.003946 / .9778` | `.365590 / .8062` | `.005701 / .9715` | `.999968` | `1.279600` |
| composition | constant acceleration | `.007435 / .9591` | `.753275 / .1839` | `.005789 / .9684` | `.999966` | `1.280218` |
| composition | 50:50 mixture | `.005952 / .9685` | `.592126 / .4950` | `.005745 / .9699` | `.999967` | `1.279909` |
| extrapolation | constant speed | `.004254 / .9804` | `.378417 / .8054` | `.006220 / .9716` | `.999961` | `1.280398` |
| extrapolation | constant acceleration | `.008006 / .9578` | `.748548 / .1848` | `.006072 / .9698` | `.999963` | `1.280937` |
| extrapolation | 50:50 mixture | `.006411 / .9691` | `.593104 / .4951` | `.006147 / .9707` | `.999962` | `1.280668` |

Every robust row clears `RMSE <=.020`, accuracy `>=.90`, and the complete
physical task gate. Every naive row fails the strong ceiling.

## The registered gate mismatch

The speed-family repair statistics explain the inconclusive classification:

| Shift | Robust/naive RMSE | Accuracy gain | Cross-entropy gain | Registered repair |
| --- | ---: | ---: | ---: | --- |
| composition mean | `.01561` | `+.16533` | `-1.82517` | fail: accuracy `<.20` |
| extrapolation mean | `.01644` | `+.16626` | `-1.94139` | fail: accuracy `<.20` |

Across the ten speed cells, accuracy gain ranges from `.1560` to `.1914`.
Acceleration and mixture repair pass all `10/10` cells. Speed passes every
other repair component and every absolute endpoint, but preregistration makes
the conjunction indivisible.

The `.20` clause was inherited from the much more destructive acceleration
baseline and was not calibrated to the constant-speed control. This diagnosis
is post-outcome and cannot be used to reclassify the result.

## Controls and integrity

| Contract | Result | Limit |
| --- | ---: | ---: |
| requested/completed/invalid paired cells | `10 / 10 / 0` | exact |
| matched predecessor examples replayed | `81,920` | registered |
| new corrupted evaluations | `81,920` | registered |
| base cohort hash replay | `20/20` | exact |
| minimum examples at any frame index | `466` | `>=400` |
| continuous law-inclusion error | `1.277e-15` maximum | `<=1e-10` |
| continuous future-state error | `2.736e-14` maximum | `<=1e-10` |
| continuous hidden-index recovery | `81,920 / 81,920` | exact |
| minimum quantized hidden-index recovery | `.993164` | descriptive |
| minimum clean phase-chart margin | `.333877` | `>=.20` |
| corruption/deck equivariance token errors | `0` | `0` |
| maximum prediction deck-action error | `6.125e-14` | `<=2e-12` |
| shuffled absolute scalar correlation | `.02783` maximum | `<=.10` |
| shuffled scalar RMSE | `.98205` minimum | `>=.80` |
| shuffled complete task passes | `0/120` | `0` |
| law-label reads / selector decisions | `0 / 0` | `0 / 0` |
| models / checkpoints / optimizer steps | `0 / 0 / 0` | `0 / 0 / 0` |
| changed parameters / target-using fits | `0 / 0` | `0 / 0` |

The actual mixture metrics were computed from concatenated predictions and
targets, not inferred by averaging aggregate reports.

## What is established and what is not

Established:

- degree-one and degree-two carrier laws are nested in the tested quadratic
  chart;
- one frozen, law-blind robust estimator clears every absolute endpoint under
  both shifts;
- the corruption is material and the common estimator is oracle-faithful;
- the tested evidence does not license a selector or TinyLLM.

Not established:

- the full preregistered closure hypothesis, because one joint effect-size gate
  failed in every replicate;
- robustness to non-nested, switching, nonlinear, or stochastic laws;
- usefulness of a learned continuation;
- a generally optimal robust estimator.

## Program decision

Do not reinterpret the absolute endpoint as a preregistered positive result,
and do not train a selector or TinyLLM. If formal confirmation of this nested
family matters, use fresh paired cohorts and preregister a nonredundant repair
gate whose role is only to prevent a tradeoff—for example absolute robust
ceiling plus RMSE/cross-entropy dominance and nondecreasing accuracy. Do not
reuse these outcomes in that aggregate.

For genuinely new learned scope, move to a non-nested law family or a within-
sequence switching law. Such a study must still compare a fixed enumerative or
change-point estimator with an oracle-law arm before optimization.

## Reproduction and provenance

```bash
MPLCONFIGDIR=/tmp/mpl-c3-hidden-order \
pixi run python -m \
  experiments.structure_net.tinyllm_c3_hidden_order_corruption_fixed_estimator
```

| Artifact | SHA-256 |
| --- | --- |
| primary result | `88e44e76c44d654331ea647ccedd8dad505030a65ab7c4ac241d8b98d98bd02e` |
| runner | `318cf81497960d37f86b1be58af3d819076e7dc9b620fe2ba73ae215ae0adea7` |
| preregistration | `5530806f4f71ff72eba9abdcc3001e5ebe072bfb0217891662fc4e748fe95da5` |
| constant-speed predecessor result | `9b03a0cf19ef820586e2a031d80a694d498655de151f70f245c3c82b0abb853a` |
| constant-speed predecessor runner | `9471ffe7f319d3d53234fd795fc48ce39a7717970d0e191ae2882343ad6d3b37` |
| corruption predecessor result | `59681f2764b988f05b0916965898b87d5b233b2165151fe19e0c97391fe467b9` |
| corruption predecessor runner | `8600081fc813ae6da5a49c39222bf3a94286127d7238899d61ec9acd1c6d31f8` |

The focused runner suite passes against the authoritative artifact. The report
hash is pinned by the meta-hypothesis evidence module after sealing.
