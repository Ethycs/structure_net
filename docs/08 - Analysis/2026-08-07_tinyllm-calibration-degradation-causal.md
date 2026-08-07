# TinyLLM calibration-degradation causal test

**Status:** FAILED PREREGISTERED JOINT ROBUSTNESS GATE — representation quotient
persists beyond frozen task utility

**Date:** 2026-08-07

**Hypothesis:** `tinyllm-calibrated-reference-robustness-curve-v1`

**Conformance:** `PREREGISTERED`; frozen-checkpoint inference-only diagnostic.
The intervention, noise grid, observers, thresholds, seed rule, and outcome
labels were fixed in the
[preregistration](../07%20-%20Status%20Reports/2026-08-07_tinyllm-calibration-degradation-causal-preregistration.md).

## Verdict

The retained calibrated TinyLLM systems require substantially more accurate
calibration than the preregistered target. The analytic positive control has
joint robust radius `0.00`; the learned calibrated-equivariant arm reaches only
`0.05`. Both were required to reach `0.20`. The fixed campaign classification
is therefore:

```text
exact_calibration_required
```

The failure is more specific than that label alone suggests. At noise levels
`0.05`, `0.10`, and `0.20`, **every one of the 20 representation cells per
arm** still passes the joint cosine-retention and conditional-branch gates.
What fails first is the frozen task output: exact-bin accuracy crosses the
three-point utility ceiling. Branch balanced accuracy and conditional log-loss
gain never fail at any point on the primary noise curve; cosine retention does
not begin failing until `0.40`.

Thus the tested systems retain a probe-defined internal quotient under modest
reference error, but the frozen continuation does not use that perturbed
coordinate robustly enough for the declared task endpoint. This rejects the
claim that the full calibrated architecture is robust to `sigma=0.20`; it does
not show that the quotient geometry itself disappears at the same threshold.

## Primary robustness gate

A seed passes a level only when both cuts and both shifts pass the
representation gate and both shifts lose at most `0.03` exact-bin accuracy.
An arm passes at least `4/5` seeds. Robust radius is the largest consecutive
passing prefix from zero.

| Arm | `0.00` | `0.05` | `0.10` | `0.20` | `0.40` | `0.80` | Radius |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Analytic calibrated | 5/5 | 1/5 | 0/5 | 0/5 | 0/5 | 0/5 | `0.00` |
| Learned calibrated-equivariant | 5/5 | 4/5 | 0/5 | 0/5 | 0/5 | 0/5 | `0.05` |

The learned arm is slightly more tolerant at the first nonzero level, but it
does not approach the locked target. The analytic control also fails, so this
is not primarily a learned-encoder optimization failure.

The `all_default` dynamic-range control passes `0/5` seeds in both arms, as
required. Its failure prevents the clean result from being explained by a
system that ignores the calibration packet.

## Failure localization

The table below counts failing shift cells among five seeds times two shifts.
Representation counts include both cuts, so their denominator is 20 per arm;
task counts have denominator 10.

| Arm and level | Cosine failures | Branch-accuracy failures | Log-loss failures | Task-utility failures |
| --- | ---: | ---: | ---: | ---: |
| Analytic `0.05` | 0/20 | 0/20 | 0/20 | 4/10 |
| Analytic `0.10` | 0/20 | 0/20 | 0/20 | 10/10 |
| Analytic `0.20` | 0/20 | 0/20 | 0/20 | 10/10 |
| Analytic `0.40` | 15/20 | 0/20 | 0/20 | 10/10 |
| Learned `0.05` | 0/20 | 0/20 | 0/20 | 1/10 |
| Learned `0.10` | 0/20 | 0/20 | 0/20 | 8/10 |
| Learned `0.20` | 0/20 | 0/20 | 0/20 | 9/10 |
| Learned `0.40` | 4/20 | 0/20 | 0/20 | 10/10 |

Across both arms and all 240 primary representation cells, conditional branch
accuracy stays below `0.55` and conditional log-loss gain stays below `0.02`.
The first representation failure is semantic-base loss, not renewed branch
leakage.

At `sigma=0.20`, mean cosine correlation over cuts, shifts, and seeds remains
`0.963` in both arms. Mean conditional branch accuracy is `0.502` analytic and
`0.504` learned, with mean conditional log-loss gain approximately zero. This
is a clean separation between a still-decodable invariant coordinate and a
brittle frozen task readout.

## Task degradation

Mean exact-bin accuracy and absolute drop from the same checkpoint's clean
value are:

| Arm | Level | Composition accuracy / drop | Extrapolation accuracy / drop |
| --- | ---: | ---: | ---: |
| Analytic | `0.00` | 0.745 / 0.000 | 0.616 / 0.000 |
| Analytic | `0.05` | 0.716 / 0.029 | 0.595 / 0.022 |
| Analytic | `0.10` | 0.650 / 0.095 | 0.537 / 0.080 |
| Analytic | `0.20` | 0.510 / 0.235 | 0.413 / 0.203 |
| Learned | `0.00` | 0.717 / 0.000 | 0.492 / 0.000 |
| Learned | `0.05` | 0.697 / 0.020 | 0.486 / 0.006 |
| Learned | `0.10` | 0.637 / 0.080 | 0.456 / 0.036 |
| Learned | `0.20` | 0.486 / 0.231 | 0.388 / 0.104 |

Means do not determine the gate. For example, the analytic `0.05` composition
mean drop is below `0.03`, but three individual composition seeds exceed the
ceiling; a fourth seed fails extrapolation. Conversely, the learned arm passes
`0.05` because only seed 17 composition misses, at a `0.03125` drop.

The strict exact-bin endpoint is intentionally sensitive to boundary motion.
This result establishes frozen task-output fragility; it does not by itself
identify whether the error is a smooth coordinate bias, altered confidence, or
a changed nonlinear continuation. That distinction is the next low-cost
activation diagnostic and should be resolved before retraining.

## Activation geometry

The learned encoder moves slightly less than the analytic control under the
same reference errors. At `sigma=0.20`, mean clean-to-perturbed centered linear
CKA is:

| Arm | Front end | Full depth |
| --- | ---: | ---: |
| Analytic | 0.942 | 0.916 |
| Learned | 0.967 | 0.945 |

Mean full-depth row cosine is `0.960` analytic and `0.962` learned. Despite
this substantial geometric agreement and passing probe endpoints, mean task
accuracy has already fallen by 20.3–23.5 points in the analytic arm and
10.4–23.1 points in the learned arm, depending on shift. Aggregate activation
similarity therefore cannot substitute for frozen-continuation sufficiency.

At `sigma=0.40`, cosine-base failures begin: all analytic front-end cells and
all analytic extrapolation full-depth cells fail, while only four learned
front-end cells fail. Transformer depth can still recover the probe-defined
base in some cases after task utility is already gone.

## Calibration-component ablations

| Defaulted calibration group | Analytic passes | Learned passes |
| --- | ---: | ---: |
| Orientation | 0/5 | 0/5 |
| Signed speed | 0/5 | 2/5 |
| Amplitude | 5/5 | 0/5 |
| Offset | 0/5 | 0/5 |
| Drift | 5/5 | 5/5 |
| Entire packet | 0/5 | 0/5 |

The fixed canonicalizer is exactly insensitive to the amplitude and drift
defaults in this generator, but depends strongly on orientation, speed, and
offset. The learned encoder shares orientation/offset dependence, is partially
more tolerant of speed defaulting, and has acquired amplitude dependence that
the analytic construction does not need. Neither arm depends materially on
the declared drift field. These are secondary causal localizations, not a new
primary success criterion.

## Integrity and provenance

The campaign completed `10/10` checkpoint evaluations with no failures or
exclusions. It trained `0` models, `0` front ends, and `0` task heads. It fit
ten clean conditional nulls and twenty clean diagnostic suites, then froze
them across all interventions.

Every cell reproduced its predecessor clean metrics with maximum absolute
error `0`, preserved identical sensor/input/target tensors across calibration
interventions, and revalidated source model and full-system state hashes. The
locked source manifest covers 30 result/model/front-end files.

```text
campaign SHA-256
170d99553058ab544d95d6abf4d26ea1062bfe1b79fda2deefc531dfaebd0f6e

implementation SHA-256
f273abf83231e4fc37583154c24158c81bba15a29e11ca3f96daf0b579be0f70

source campaign SHA-256
80623a7283fed3d902a1cbf9fb58afe2a2fef82ab86406212f907de40a96c501

source manifest SHA-256
23598aaef5a0d16825ff9a928de57857e7bd58b10feab8853739448acfd983fc

common-noise SHA-256
1896c852644f724f5a2d214d5e69380eef3e46abb2e7b295eb0d4bace51a0c82
```

An exact rerun reported that the aggregate was already complete and left its
bytes unchanged. The analysis took 199.7 seconds on an RTX 2060 SUPER with
peak allocated CUDA memory of 335,768,576 bytes.

The complete data tree, including the campaign, shakedown, activation audits,
JSON meta record, and ChromaDB evidence, is tracked by DVC as:

```text
DVC root: ca2abd03b528233760a5f1cb23686dca.dir
Files: 2,098
Logical bytes: 39,892,063,839
lakeFS commit: 4fca75c78b6ec1f9e83c1548bb15aca79d076de909b9e0dc7547c55223bb816f
```

The exact directory object is addressable at
`lakefs://artifacts/4fca75c78b6ec1f9e83c1548bb15aca79d076de909b9e0dc7547c55223bb816f/structure-net/files/md5/ca/2abd03b528233760a5f1cb23686dca.dir`.
After commit, the lakeFS branch had no uncommitted object diff and DVC reported
the local cache and `lakefs` remote in sync.

## Interpretation and next decision

The predecessor established that an exact phase-independent reference makes
the absolute-cosine quotient identifiable and constructible. This experiment
adds a necessary qualification:

> The structured front ends preserve the probe-defined quotient under modest
> reference error, but the trained frozen continuation is calibrated too
> tightly to that coordinate for stable exact-bin behavior.

The preregistered stop rule is active: do not tune residual penalties, add a
writer sidecar, or retrain TinyLLM yet. The shortest decisive next step is an
inference-only activation/readout decomposition using these same checkpoints
and perturbations:

1. measure continuous circular-moment error, not only exact-bin boundaries;
2. project perturbed full-depth activations onto the frozen clean task
   covector/readout geometry;
3. compare a no-fit clean-coordinate transport against an oracle scalar
   recalibration; and
4. determine whether reference error changes the available semantic coordinate
   or only its relation to the fixed task decoder.

Only if a no-fit or one-scalar positive control recovers utility should a
prospective calibration-aware readout be trained. If the oracle cannot recover
it, improve the measurement/reference process itself.

## Artifacts and reproduction

- Aggregate: `data/experiments/tinyllm_calibration_degradation_causal/20260807_d8_existing_checkpoints/campaign_results.json`
- Per-checkpoint results and activation audits: `data/experiments/tinyllm_calibration_degradation_causal/20260807_d8_existing_checkpoints/runs/`
- Runner: `experiments/structure_net/tinyllm_calibration_degradation_causal.py`
- Tests: `tests/structure_net/test_tinyllm_calibration_degradation_causal.py`
- Typed evidence: `data/meta_hypotheses/tinyllm-calibrated-reference-robustness-curve-v1.json`
- Preregistration: `docs/07 - Status Reports/2026-08-07_tinyllm-calibration-degradation-causal-preregistration.md`

The typed evidence JSON has SHA-256
`eea84ff0902a8e428cf9f211ddf26200a39767c0d0cedb7155b9de8b864dda11`;
its hypothesis and all ten direct checkpoint records passed ChromaDB
read-back.

```bash
MPLCONFIGDIR=/tmp/matplotlib-calibration-degradation \
pixi run python -m \
  experiments.structure_net.tinyllm_calibration_degradation_causal \
  --device cuda:1 \
  --output \
  data/experiments/tinyllm_calibration_degradation_causal/20260807_d8_existing_checkpoints
```

The synthetic `sigma` coordinate is not a real-instrument uncertainty model,
and these five retained systems do not establish population prevalence.
