# TinyLLM C3 relational connection acquisition result

**Status:** VALID NEGATIVE PRIMARY — EXACT CLASS, UNRELIABLE JOINT ACQUISITION

**Date:** 2026-08-11

**Hypothesis:** `tinyllm-c3-relational-connection-acquisition-v1`

**Primary classification:**
`exact_function_class_but_population_acquisition_unreliable`

**Corrective readout classification:**
`posthoc_public_scale_readout_reaches_four_of_five_one_wrong_winding_remains`

**Preregistration:** [matched acquisition protocol](../07%20-%20Status%20Reports/2026-08-11_tinyllm-c3-relational-connection-acquisition-preregistration.md)

**Primary artifact:**
`data/experiments/tinyllm_c3_relational_connection_acquisition/20260811_preregistered/campaign_results.json`

## Verdict

The exact 187-parameter connection-invariant function class does not acquire
its known six-weight solution reliably under the frozen population protocol.
The analytic connection solution passes both shifts in all five fresh seeds,
and all three learned controls pass zero seeds. The learned true arm passes
only one of five.

```text
analytic connection ceiling:          5/5
learned true:                          1/5
learned no connection:                 0/5
learned connection shuffled:           0/5
learned target shuffled:               0/5
CPU/CUDA lifecycle-valid learned arms: 20/20
TinyLLM models instantiated:           0
```

This separates three propositions that the predecessor could not separate:

```text
the connection identifies the target                 true
the restricted class contains the exact computation  true
ordinary joint training acquires it population-wide  false
```

The failure is not a broken symmetry implementation or missing information.
It is an optimization and public-coordinate acquisition failure inside an
otherwise valid function class.

## Protocol validity

The primary seeds `(1453, 1471, 1483, 1531, 1543)` and every data,
minibatch, permutation, and action stream were disjoint from the three pilot
seeds. Scalar MSE, AdamW `1e-3`, zero weight decay, batch 64, and 2,400 steps
were frozen before any primary cohort was inspected.

The analytic positive-control stop gate passed `5/5` before learned scheduling:

| Metric across ten analytic cells | Worst value | Gate |
| --- | ---: | ---: |
| scalar correlation | `.9999966` | `>=.999` |
| scalar RMSE | `.001828` | `<=.01` |
| exact-bin accuracy | `.98535` | `>=.98` |
| target cross-entropy | `1.29009` | `<=1.35` |
| predicted-bin coverage | `16` | `16` |

All five training protocols have zero saturation, sixteen-bin coverage,
fixed-point-free target and connection derangements, and changed-connection
fractions above `.999`. Every learned arm starts from the same state within a
seed.

## Primary learned results

The registered endpoint required correlation `>=.999`, RMSE `<=.01`, exact
accuracy `>=.95`, cross-entropy `<=1.35`, and all sixteen predicted bins on
both shifts.

| Seed | Winding init/mid/final | Composition corr / RMSE / acc | Extrapolation corr / RMSE / acc | Joint |
| ---: | --- | --- | --- | --- |
| 1453 | `1 / -2 / -2` | `.01069 / .71405 / .0264` | `-.04555 / .70322 / .0508` | fail |
| 1471 | `1 / 1 / 1` | `.9999969 / .001757 / .9922` | `.9999970 / .001741 / .9922` | **pass** |
| 1483 | `1 / -2 / 1` | `.9999436 / .32531 / .0938` | `.9999472 / .32626 / .0889` | fail |
| 1531 | `1 / -2 / 1` | `.9999969 / .03386 / .7637` | `.9999967 / .03403 / .7656` | fail |
| 1543 | `1 / -2 / 1` | `.9999942 / .13263 / .2422` | `.9999927 / .13306 / .2109` | fail |

Seed 1453 ends in the wrong degree-`-2` charged map and never recovers the
target. Three other failures recover degree `1` and nearly perfect ordering by
the final checkpoint, but their output scale is still compressed. Only seed
1471 remains in the correct winding class throughout and completes the public
scale convention within the frozen budget.

This winding diagnostic is exploratory rather than a primary gate. It is
nevertheless a useful mechanistic warning: architectural equivariance permits
multiple globally different equivariant maps, and normalization makes their
winding transitions part of the optimization problem.

## Controls

The information-removal controls remain clean:

| Arm, ten held-out cells | Mean RMSE | Maximum absolute corr | Maximum accuracy | Maximum coverage |
| --- | ---: | ---: | ---: | ---: |
| no connection | `.70970` | `.05482` | `.05371` | `2` |
| shuffled connection | `.71000` | `.05046` | `.04980` | `2` |

The target-shuffled arm also passes no seed, with mean RMSE `.71416`, maximum
accuracy `.05078`, and at most two predicted bins. Some of its nearly constant
outputs have large signed correlation after normalization; their absolute
error, coverage, and task likelihood remain null. Correlation alone is
therefore not treated as success.

## Lifecycle

Every one of the twenty learned cells passes:

- finite optimization and nonzero state change;
- local observed-pair action invariance at initialization, midpoint, and final;
- exact final checkpoint state, optimizer, and prediction replay;
- exact continuation from the 1,200-step midpoint, including second-half
  history, final state, optimizer, and predictions.

The maximum action error over all sixty audited states is `8.941e-7`, more
than twenty times inside the `2e-5` gate. The campaign used 48,000 primary and
24,000 resume-verification optimizer steps. Maximum allocated CUDA memory was
`.0689 GiB`; the aggregate five-cell wall time was 1,124 seconds while the
five seeds ran concurrently across three GPUs.

## Corrective frozen-readout audit

The primary result remains negative. After observing the compressed high-
correlation failures, a separately registered post-outcome audit reloaded the
sealed final checkpoints and used zero optimizer steps. On each training
cohort it solved exactly one unregularized scalar affine fit and one
three-parameter linear readout of the frozen neutral carrier, then applied the
fit unchanged to both held-out shifts.

Both fits reach the same population result:

```text
learned true:                    4/5
learned no connection:           0/5
learned connection shuffled:     0/5
learned target shuffled:         0/5
newly repaired true seeds:       3
persistent failure:              seed 1453
optimizer steps:                 0
```

The scalar affine slopes are:

| Seed | Slope | Intercept | Post-hoc joint |
| ---: | ---: | ---: | --- |
| 1453 | `.93748` | `-.00106` | fail |
| 1471 | `1.00004` | `-.00001` | pass |
| 1483 | `1.85189` | `-.01235` | pass |
| 1531 | `1.04941` | `-.00038` | pass |
| 1543 | `1.23254` | `.01001` | pass |

Across the eight passing affine cells, minimum correlation is `.9999436`,
maximum RMSE `.007513`, minimum accuracy `.95508`, and maximum cross-entropy
`1.29020`. The neutral linear solution gives the same seed count. Seed 1453
remains null under both fits, as expected from its final degree-`-2` carrier.

This is diagnostic evidence, not a rescued primary outcome. It shows that
three primary failures contain the correct one-dimensional relation but have
not finished coordinating its public scale with the learned head. One failure
is genuinely representational under the declared linear interface.

## What the result means

Exact architectural symmetry removes invalid gauge-dependent functions from
the class, but it does not make the desired representative easy to acquire.
The class still contains distinct equivariant winding sectors, and joint
optimization can discover the right relation before fixing the public scalar
scale that the interval task requires.

This sharpens the program's engineering rule:

> Identifiability and exact equivariance are prerequisites for the desired
> computation, not guarantees of population-stable acquisition or calibrated
> use.

The analytic connection remains the economic baseline. It has six nonzero
parameters, passes every fresh cell without training, and avoids the winding
and scale acquisition problem entirely.

## Program decision

Close learning-rate, loss, weight-decay, step-count, warm-start, seed, and
threshold tuning for this function class. Do not promote the result to
unrestricted TinyLLM training.

A successor is scientifically justified only if it changes the observation
scope so that learning could provide value unavailable to the fixed analytic
transport—for example missing, noisy, partial, or unknown connection data.
Such a study must retain the fixed analytic solution as its ceiling and pass a
new identifiability contract before optimization. No same-law extension is
currently licensed.

## Reproduction and provenance

Primary campaign:

```bash
MPLCONFIGDIR=/tmp/mpl-c3-connection-acquisition-primary \
pixi run python -m \
  experiments.structure_net.tinyllm_c3_relational_connection_acquisition \
  --mode primary
```

Corrective audit:

```bash
MPLCONFIGDIR=/tmp/mpl-c3-connection-readout-audit \
pixi run python -m \
  experiments.structure_net.tinyllm_c3_relational_connection_readout_audit
```

| Artifact | SHA-256 |
| --- | --- |
| primary preregistration | `d1ca34b1d251dc06a69aa293bfbfac2c3ee63fb80de3d4b7ed822e01ad10c015` |
| primary runner | `cf425970a3424a32e410492ea79d7d17fd579a83cea78bacea9b8a58674116f0` |
| primary campaign result | `b4b6e94392a866f7a94188d18935db6602fbc83b936e14324b1060e56ce61a4a` |
| corrective audit registration | `17d76393374a7898943fe9044ea8019281b2bceab693a372c2af6f14c458cf9d` |
| corrective audit runner | `0c08b9ccd1062fa173e45769ebaa0396614ff81eee327486f3936b657fffa75a` |
| corrective audit result | `1fb139ebd13b1ac78d77fa0b82d206f237b3d14ef86173cd7e8d6825dd1731a5` |

The corrective runner and result hashes above identify the audited bytes. The
primary result remains the sole preregistered acquisition outcome.
