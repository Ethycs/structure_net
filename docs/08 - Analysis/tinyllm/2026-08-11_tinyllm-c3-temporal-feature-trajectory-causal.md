# TinyLLM C3 temporal feature-trajectory causal result

**Status:** VALID REGISTERED NEGATIVE; DOWNSTREAM FAILURE LOCALIZED

**Date:** 2026-08-11

**Hypothesis:** `tinyllm-c3-temporal-feature-trajectory-causal-v1`

**Classification:** `fixed_operator_available_but_frozen_continuation_cannot_use_projected_trajectory`

**Evidence role:** `registered_post_outcome_artifact_only_feature_trajectory_causal`

**Preregistration:** [C3 temporal feature-trajectory causal diagnostic](../07%20-%20Status%20Reports/2026-08-11_tinyllm-c3-temporal-feature-trajectory-causal-preregistration.md)

**Primary artifact:** `data/experiments/tinyllm_c3_temporal_feature_trajectory_causal/20260811_d6_preregistered/result.json`

## Verdict

Supplying the five frozen analytic-sensor TinyLLM continuations with a
denoised, constant-speed carrier trajectory does not repair their population
task failure. The fixed all-increment scalar and physical decoder pass both
shifts in `5/5` seeds, but the frozen TinyLLM continuation passes after the
corresponding all-frame trajectory patch in only the same `2/5` seeds that pass
without the intervention.

```text
natural source continuation:        2/5 seeds
last-increment trajectory patch:     2/5 seeds
all-increment trajectory patch:      2/5 seeds
fixed all-increment bypass:          5/5 seeds
shuffled-target all-frame controls:  0/5 seeds
required repair population:         >=4/5 seeds
```

The correct temporal statistic is available, but these frozen continuations do
not reliably turn its projected feature trajectory into the task posterior.
The failure is therefore downstream of analytic carrier construction and
all-frame temporal denoising in the tested systems. Same-task TinyLLM repair is
closed.

## Intervention

Let `q_0,...,q_7` be the exact analytic `C3` carrier sequence. Four feature
sequences were patched immediately before the frozen learned sequence
embedding:

```text
source:
  q_t

last_consistent:
  d_last = q_7 conjugate(q_6)
  q_t' = q_7 d_last^(t-7)

mean_consistent:
  d_mean = normalize(sum_(t=1..7) q_t conjugate(q_(t-1)))
  q_t' = q_7 d_mean^(t-7)

early_deranged:
  q_0,...,q_5 from another example; retain the example's q_6,q_7
```

The `mean_consistent` arm is the causal repair candidate. It contains the same
all-frame group statistic that succeeds with the fixed physical decoder. No
arm uses the target, latent phase, speed, fitted coefficient, or parameter
update.

## Registered population decision

An arm passes a seed only when correlation, exact-bin accuracy,
cross-entropy, and bin-coverage thresholds pass jointly on composition and
extrapolation.

| Seed | Source | Last-consistent | Mean-consistent | Fixed mean bypass |
| ---: | --- | --- | --- | --- |
| 7 | pass | pass | pass | pass |
| 17 | fail | fail | fail | pass |
| 29 | fail | fail | fail | pass |
| 41 | fail | fail | fail | pass |
| 53 | pass | pass | pass | pass |

The all-frame patch fails the registered `4/5` repair threshold and does not
change which seeds pass. The locked negative classification applies because
the fixed bypass passes `5/5` and all target-deranged controls fail.

## Aggregate task behavior

Means over the five frozen checkpoints on the unchanged `1,024`-example
cohort per shift:

| Shift | Feature arm | Gate count | Correlation | RMSE | Exact-bin acc | Cross-entropy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| composition | source | 5/5 | `.998890` | `.041832` | `.817383` | `1.295251` |
| composition | last-consistent | 5/5 | `.998842` | `.042347` | `.815625` | `1.296435` |
| composition | mean-consistent | 5/5 | `.998893` | `.041778` | `.818359` | `1.295173` |
| extrapolation | source | 2/5 | `.943764` | `.240845` | `.380664` | `2.401342` |
| extrapolation | last-consistent | 2/5 | `.943046` | `.242342` | `.381641` | `2.408889` |
| extrapolation | mean-consistent | 2/5 | `.943844` | `.240616` | `.382812` | `2.400611` |

The mean-consistent patch changes extrapolation accuracy by only `+.00215`
and cross-entropy by `-.00073` on average. It is a valid intervention but not a
material population repair.

The failing all-frame seeds miss different conjuncts:

| Seed | Mean extrap corr | Mean extrap acc | Mean extrap CE | Failed conjuncts |
| ---: | ---: | ---: | ---: | --- |
| 17 | `.871214` | `.331055` | `3.360150` | correlation, accuracy, CE |
| 29 | `.951054` | `.411133` | `2.325336` | CE |
| 41 | `.961312` | `.343750` | `2.069755` | accuracy |

This variation is consistent with a support-dependent continuation/readout
defect rather than one missing temporal statistic shared by all checkpoints.

## Causal sensitivity of the early trajectory

The negative repair is not evidence that the continuation ignores the first
six carrier states. Replacing only those states with a deterministic
other-example derangement changes the predicted bin on average for `72.89%`
of composition examples and `76.66%` of extrapolation examples. It reduces
mean exact-bin accuracy to `.2539` and `.1449`, and passes no task gate.

By contrast, the all-frame projection changes the predicted bin on only
`.68%` of composition examples and `.78%` of extrapolation examples; mean
Jensen-Shannon divergence from the source posterior is `1.18e-5` and
`3.70e-5`. The continuations are causally sensitive to early-frame detail, but
the symmetry-typed denoising that helps the fixed decoder barely moves their
posteriors. They learned to use the trajectory in a way that does not extract
the registered all-frame improvement.

## Validity and controls

| Contract | Result | Limit |
| --- | ---: | ---: |
| completed/valid checkpoint cells | `5/5` | exact |
| source posterior replay error | `0` maximum | `<=2e-6` |
| stored source metric replay error | `2.384e-7` maximum | `<=2e-6` |
| feature algebra error | `1.480e-7` maximum | `<=2e-6` |
| source state unchanged | `5/5` | exact |
| early-feature derangement fixed points | `0` in every cell | `0` |
| target derangement fixed points | `0` in every cell | `0` |
| mean-consistent shuffled seed passes | `0/5` | `<=1/5` |
| fixed mean bypass passes | `5/5` | `5/5` |
| optimizer steps / changed parameters | `0 / 0` | `0 / 0` |
| checkpoints loaded / target-using fits | `5 / 0` | `5 / 0` |

All five systems were loaded sequentially on `cuda:0`. The runner, source
campaign, checkpoints, datasets, preregistration, and predecessor fixed-operator
result were content-validated before interpretation.

## What the result establishes

- The registered all-frame feature patch does not repair the analytic d6
  TinyLLM population.
- The fixed physical computation remains sufficient on exactly the same data.
- The population failure is not caused solely by failure to denoise the
  analytic carrier increments.
- The frozen continuations use early-frame information, but do not reliably
  convert the improved symmetry-typed trajectory into improved task output.

## Scope boundary

The result does not establish that a transformer cannot learn circular means,
that every alternative typed continuation must fail, or that TinyLLM is
useless under nonconstant or unknown dynamics. It tests five existing analytic
d6 checkpoints, one declared all-frame projection, one last-step comparator,
and the existing physical task gates.

It also does not distinguish sequence embedding, attention, MLP, final
normalization, and tied answer rows inside the downstream failure. That finer
localization has no current engineering decision value because the fixed
operator already solves the declared task more accurately and cheaply.

## Program decision

Use the fixed all-increment group operator and physical decoder for the current
calibrated, noiseless, constant-speed task. Do not train another same-task
TinyLLM continuation, widen the continuation, or add a residual loss.

A learned temporal model becomes licensed only after the problem changes so
that the fixed operator is no longer sufficient—for example, declared
nonconstant dynamics, missing or corrupted frames, or an unknown group law.
Such a study must begin with identifiability and analytic-ceiling preflight.

## Reproduction and provenance

```bash
MPLCONFIGDIR=/tmp/mpl-c3-trajectory \
pixi run python -m \
  experiments.structure_net.tinyllm_c3_temporal_feature_trajectory_causal \
  --device cuda:0
```

| Artifact | SHA-256 |
| --- | --- |
| primary result | `a0ac3315b03aa65df273539a24d8c08f51f12e8ceb702859cc16a282886ddf27` |
| runner | `bcb10634a4928848d55608f3ef3fbe054fb3ed84c145f8311af465fb8a1fd17f` |
| preregistration | `73bf3b792ec44872b8362b69fb51e88826be44073739615a38a28fbdfe2a97ca` |
| source campaign | `e46710de358a3c9c5d30ddd4a19e36182b94a50cabc699014f79d58d3d3de7cc` |
| fixed-operator result | `9b03a0cf19ef820586e2a031d80a694d498655de151f70f245c3c82b0abb853a` |

The focused runner suite passes `8/8` tests against the authoritative artifact.
