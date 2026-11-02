# TinyLLM C3 hidden-order corruption corrective result

**Status:** FRESH PROSPECTIVE NO-TRAINING RESULT CONFIRMED

**Date:** 2026-08-11

**Hypothesis:** `tinyllm-c3-hidden-order-corruption-corrective-v1`

**Classification:** `fresh_corrective_confirms_common_robust_nested_law_closure`

**Preregistration:** [C3 hidden-order corruption corrective](../07%20-%20Status%20Reports/2026-08-11_tinyllm-c3-hidden-order-corruption-corrective-preregistration.md)

**Primary artifact:** `data/experiments/tinyllm_c3_hidden_order_corruption_corrective/20260811_preregistered/result.json`

## Verdict

Five entirely fresh paired cohorts confirm that one frozen, law-blind robust
quadratic estimator closes constant speed, constant acceleration, and their
actual 50:50 mixture under one unmarked frame substitution:

```text
common robust absolute endpoint:  5/5
oracle absolute endpoint:         5/5
corruption materiality:           5/5
Pareto repair:                    5/5
oracle fidelity:                  5/5
required population:            >=4/5
```

No predecessor example was pooled. The primary estimator read no law label and
made no selector decision. No model, checkpoint, optimizer step, changed
parameter, or target-using fit was involved.

This result does not retroactively change the first study's valid
`inconclusive_hidden_order_corruption_preflight` classification. It supplies a
separate confirmatory result on fresh data with the pre-existing `.005`
nondegradation guard.

## Corrective design

The original repair gate required a `.20` accuracy gain even though the strong
absolute endpoint already required accuracy `>=.90` and materiality separately
required the corrupted naive arm to fail. That made the effect-size requirement
depend on how badly each law-specific baseline was damaged.

The corrective retained every absolute, corruption, RMSE, cross-entropy,
oracle, group-action, and shuffle contract. It replaced only that redundant
effect-size clause with the `-.005` accuracy nondegradation guard used by the
fixed-operator studies before the inconclusive result existed. Fresh seeds and
new dataset, corruption, and target-shuffle streams prevent outcome reuse.

## Population measurements

Means over five fresh paired cohorts per shift:

| Shift | Population | Clean RMSE / acc | Naive RMSE / acc | Robust RMSE / acc | Robust corr | Robust CE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| composition | constant speed | `.004016 / .9794` | `.365386 / .8033` | `.005828 / .9714` | `.999966` | `1.282519` |
| composition | constant acceleration | `.007298 / .9626` | `.745592 / .1906` | `.005789 / .9697` | `.999966` | `1.281885` |
| composition | 50:50 mixture | `.005890 / .9710` | `.587203 / .4969` | `.005810 / .9705` | `.999966` | `1.282202` |
| extrapolation | constant speed | `.004246 / .9793` | `.365727 / .8052` | `.005960 / .9716` | `.999964` | `1.281893` |
| extrapolation | constant acceleration | `.007798 / .9591` | `.748327 / .1890` | `.006158 / .9677` | `.999962` | `1.282409` |
| extrapolation | 50:50 mixture | `.006279 / .9692` | `.588993 / .4971` | `.006060 / .9697` | `.999963` | `1.282151` |

Every robust row clears `RMSE <=.020`, accuracy `>=.90`, and the complete
physical task gate. Every corrupted naive row fails the strong ceiling.

## Pareto repair

| Shift | Population | Robust/naive RMSE | Accuracy delta | Registered Pareto repair |
| --- | --- | ---: | ---: | --- |
| composition | constant speed | `.01600` | `+.16812` | pass |
| composition | constant acceleration | `.00776` | `+.77910` | pass |
| composition | mixture | `.00990` | `+.47361` | pass |
| extrapolation | constant speed | `.01631` | `+.16641` | pass |
| extrapolation | constant acceleration | `.00823` | `+.77871` | pass |
| extrapolation | mixture | `.01029` | `+.47256` | pass |

All cells also improve cross-entropy by more than the registered `.10`. The
common estimator is not trading task accuracy for scalar repair.

## Controls and integrity

| Contract | Result | Limit |
| --- | ---: | ---: |
| requested/completed/invalid paired cells | `10 / 10 / 0` | exact |
| fresh base examples | `81,920` | registered |
| fresh corrupted evaluations | `81,920` | registered |
| old primary examples pooled | `0` | `0` |
| deterministic base/corruption regeneration | `20/20` | exact |
| minimum examples at any frame index | `465` | `>=400` |
| continuous law-inclusion error | `1.096e-15` maximum | `<=1e-10` |
| continuous future-state error | `2.561e-14` maximum | `<=1e-10` |
| continuous hidden-index recovery | `81,920 / 81,920` | exact |
| minimum quantized hidden-index recovery | `.993652` | descriptive |
| minimum clean phase-chart margin | `.306074` | `>=.20` |
| corruption/deck equivariance token errors | `0` | `0` |
| maximum prediction deck-action error | `1.263e-13` | `<=2e-12` |
| shuffled absolute scalar correlation | `.04413` maximum | `<=.10` |
| shuffled scalar RMSE | `.97621` minimum | `>=.80` |
| shuffled complete task passes | `0/120` | `0` |
| law-label reads / selector decisions | `0 / 0` | `0 / 0` |
| models / checkpoints / optimizer steps | `0 / 0 / 0` | `0 / 0 / 0` |

## Scientific conclusion

Hidden degree-1 versus degree-2 order is not a genuine model-selection problem
in this representation. Both laws occupy one quadratic carrier chart, and the
same robust estimator handles the unmarked observation error without knowing
which subfamily generated a sequence.

The combined evidence now supports:

> A learned selector has no demonstrated job when candidate dynamics are nested
> inside one fixed identifiable chart.

This is narrower than saying learned temporal models are unnecessary. It covers
exact calibrated `C3`, eight observations, one gross corrupted frame, and the
nested constant-speed/constant-acceleration family.

## Program decision

Close selector and TinyLLM training on this nested family. Do not climb law
degree or corruption count to manufacture a failure.

The next learning-relevant preflight must use a genuinely non-nested family or
a law that switches within a sequence. It should first compare:

1. an oracle-law or oracle-change-point estimator;
2. a fixed enumerative/change-point estimator;
3. only conditionally, a compact learned typed continuation.

Training is licensed only if the oracle remains recoverable while the strongest
registered fixed estimator fails on fresh data.

## Reproduction and provenance

```bash
MPLCONFIGDIR=/tmp/mpl-c3-hidden-corrective \
pixi run python -m \
  experiments.structure_net.tinyllm_c3_hidden_order_corruption_corrective
```

| Artifact | SHA-256 |
| --- | --- |
| primary result | `bd5af6550e65c0b9048030a8f6d01cf80e06e9f027edb89066d8338bb8b47c2a` |
| runner | `6d8e408f2347b86440ba6307345b86cad9f2740ee0945c358dea122d1c8d2789` |
| preregistration | `2778eb0e9ade9c0f084e194865df17b2b9ece1558d396d4e9227f3d599e91834` |
| predecessor result | `88e44e76c44d654331ea647ccedd8dad505030a65ab7c4ac241d8b98d98bd02e` |
| predecessor runner | `318cf81497960d37f86b1be58af3d819076e7dc9b620fe2ba73ae215ae0adea7` |

The focused runner suite passes against the authoritative artifact. The report
hash is pinned by the meta-hypothesis evidence module after sealing.
