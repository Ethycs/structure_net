# TinyLLM task-activation scalar sensor v2

**Status:** NOT CONFIRMED — NO PORTABLE OBSERVABLE SCALAR SENSOR, 0/3  
**Date:** 2026-08-07  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, `UNDERPOWERED`, fresh-cohort corrective mechanistic diagnostic  
**Hypothesis:** `tinyllm-c2-task-activation-scalar-sensor-v2`  
**Schema:** `nal.tinyllm-c2-task-activation-scalar-sensor.v2`  
**Preregistration:** [task-activation scalar sensor v2 preregistration](../07%20-%20Status%20Reports/2026-08-07_tinyllm-task-activation-scalar-sensor-v2-preregistration.md)

## Verdict

The observable task-activation scalar-sensor hypothesis is **not confirmed**.
The complete preregistered gate passed `0/3` checkpoints, with the fixed
classification `observable_scalar_not_identified` in seeds `7`, `29`, and
`53`.

This is a valid negative rather than a failed mechanism check. Provenance,
numerics, source and fresh local linearization, target controls, PCA, and both
oracle correction paths passed in all three checkpoints. The local oracle and
the frozen source covector supplied with the exact fresh signed error each
passed all six fresh composition/extrapolation cells. What failed was the
remaining scalar prediction problem.

No declared source-fitted scalar observer passed its joint magnitude, sign,
and causal endpoint gates on fresh cohort E. The primary pre-write
`causal_combined` observer had fresh zero-referenced R2 from `-2.528` to
`0.130`, sign agreement from `58.5%` to `69.4%`, and relative L2 from `0.933`
to `1.878`. The locked requirements were R2 at least `0.50`, sign agreement at
least `75%`, and relative L2 at most `0.707`. Later post-MLP and output
lookahead features did not rescue any checkpoint.

The only repeated partial effect is support-relative: the primary patch passes
composition in `3/3` checkpoints but extrapolation in `0/3`. It does not rescue
the hypothesis because scalar prediction remains poor and the locked endpoint
requires both shifts.

The shortest retrospective sidecar path is therefore closed for the declared
observable summaries. The stable mechanistic result remains a portable
phase-conditioned task covector whose required signed amplitude is
example-local. The next justified branch is prospective architectural
training with a typed invariant/equivariant sensor, not another source-only
ridge observer, wider covector, topology scan, or representation penalty.

## Lifecycle correction

The initially preferred combined observable-phase protocol was stopped before
cohort E. Its source-only CUDA shakedown failed the observed-carrier and frozen
covector-replay contracts, so its fresh-E runner is execution-locked. Cohort D
from an obsolete one-checkpoint lifecycle is quarantined and contributes no
quality evidence.

Version 2 isolates the unresolved scalar question by retaining the already
validated oracle quotient-phase chart and frozen source covector. A disposable
one-checkpoint CUDA lifecycle used separate `systems_f` seeds
`730007/730008`; all validity gates passed, but none of its scalar outcomes is
scientific evidence. Fresh cohort E (`630007/630008`) was first evaluated by
the locked three-checkpoint campaign.

## Campaign integrity

| Item | Measured value |
| --- | --- |
| checkpoints requested/completed | `3/3` (`7`, `29`, `53`) |
| failures, exclusions, retries | `0`, `0`, `0` |
| TinyLLM models or writers trained | `0` |
| PCA summaries fit | `9` |
| scalar observers fit | `24` |
| source-fit orbit examples | `768` |
| fresh primary cells | `6` |
| fresh seeds | composition `630007`; extrapolation `630008` |
| device | NVIDIA GeForce RTX 2060 SUPER (`cuda:1`) |
| PyTorch / CUDA build | `2.5.1+cu121` |
| peak allocated CUDA memory | `289,287,168` bytes |
| aggregate analysis time | `16.68` seconds |
| implementation SHA-256 | `698744c6e1791cec7f9e1cdd0504346ad84211a1313509c82dbdc1eddc29d03c` |
| DVC root | `3842da6ff75cac8efa7d2a01f89b898a.dir` (`1,930` files; `39,817,712,232` bytes) |
| lakeFS snapshot | `0d5cbc99bf88295b1189f6706066cfab8ccd58d63fe9c8e7468b242770632345` |

The checkpoints are the replication units. The six cells are repeated
composition/extrapolation measurements, not six independent models. All
results use one producing implementation. An immutable resume reused the
completed aggregate without rewriting it.

The exact DVC directory object exists at
`lakefs://artifacts/main/structure-net/files/md5/38/42da6ff75cac8efa7d2a01f89b898a.dir`
in the cited lakeFS snapshot. The branch reports zero uncommitted objects.

## Primary scalar endpoint

The primary scalar observer combines the source-fitted oracle phase features,
calibration packet, pre-write activation PCA coordinates, and predicted
activation PCA coordinates. It never receives the fresh target, exact carrier
coordinate, exact continuation, or fresh derivative.

| Seed | source R2 | fresh R2 | fresh sign | fresh relative L2 | predictive gate |
| ---: | ---: | ---: | ---: | ---: | --- |
| 7 | 0.216 | -2.528 | 59.2% | 1.878 | fail |
| 29 | 0.717 | -0.387 | 69.4% | 1.178 | fail |
| 53 | 0.338 | 0.130 | 58.5% | 0.933 | fail |

The failure is not a borderline threshold choice. Every checkpoint misses all
three predictive requirements. Even on the source cohorts, only seed 29
exceeds the R2 and relative-L2 thresholds, and its sign agreement is only
`70.2%`.

The best fresh R2 among *all* declared arms was only `0.172` for seed 7
(`phase_only`), `0.239` for seed 29 (`predicted_activation`), and `0.130` for
seed 53 (`causal_combined`). Their corresponding best relative L2 values were
still `0.910`, `0.873`, and `0.933`, all above the `0.707` ceiling. No
activation or lookahead rung reveals a robust linearly portable amplitude.

## Causal endpoint

Values below are aggregate mean circular-moment shift from the exact
continuation across composition and extrapolation; lower is better. Passing
also requires the complete endpoint in each cell.

| Seed | order-4 baseline | local oracle | frozen covector + exact error | primary observer | all fresh cells | specific controls |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| 7 | 0.1282 | 0.0243 | 0.0284 | 0.1481 | fail | fail |
| 29 | 0.1891 | 0.0420 | 0.0477 | 0.1712 | fail | fail |
| 53 | 0.1400 | 0.0168 | 0.0224 | 0.1390 | fail | fail |

The oracle interventions pass `6/6` cells and reduce mean error by roughly
four- to eightfold. The learned primary scalar worsens seed 7, modestly helps
seed 29 without reaching the endpoint, and is almost neutral in seed 53. Its
shuffled, sign-flipped, and norm-matched random controls are not separated by
the locked `0.05`-bin margin in all checkpoints. This nonspecificity is
expected when the candidate correction itself is ineffective; it does not
weaken the already-negative primary endpoint.

There is nevertheless a useful support-relative detail: the primary patch
passes the composition cell in `3/3` checkpoints and the extrapolation cell in
`0/3`. Activations therefore recover a small in-support causal correction, but
not a portable quotient coordinate.

| Seed | composition shift / endpoint | extrapolation shift / endpoint |
| ---: | --- | --- |
| 7 | `0.1105` / pass | `0.1857` / fail |
| 29 | `0.1167` / pass | `0.2257` / fail |
| 53 | `0.1105` / pass | `0.1676` / fail |

Even the composition pass is not accurate scalar recovery: composition R2 is
only `-0.044--0.198`, relative L2 is `0.895--1.022`, and sign agreement is
`56.7--60.7%`. The causal endpoint is tolerant enough for the correction to be
adequate in support, while its example-level amplitude remains unidentified.

All post-MLP, output, and full-lookahead arms fail their joint predictive and
causal gates in every checkpoint. Thus the missing amplitude is not recovered
merely by observing a later state with the same small source-fitted linear
sensor.

## Preregistered gates

| Gate | Required | Result | Verdict |
| --- | --- | --- | --- |
| provenance, numerics, and PCA | `3/3` | `3/3` | pass |
| source and fresh local linearization | `3/3` | `3/3` | pass |
| zero/exact/direct target controls | `3/3` | `3/3` | pass |
| local oracle passes both fresh cells | `3/3` | `3/3` | pass |
| frozen source covector + exact fresh error passes both cells | `3/3` | `3/3` | pass |
| primary scalar predictive gate | `3/3` | `0/3` | fail |
| primary scalar causal gate | `3/3` | `0/3` | fail |
| every negative control specific | `3/3` | `0/3` | fail |
| any declared lookahead joint pass | diagnostic | `0/3` | no rescue |
| complete scalar-sensor gate | `3/3` | `0/3` | **fail** |

## Mechanistic checks

The fresh finite-difference calculation is well conditioned. Fine/coarse
derivative cosines exceed `0.999999`, and the exact local signed-error model has
fresh zero-referenced R2 `0.981--0.997` with residual-MAE fractions
`0.033--0.066`. This rules out an invalid local linearization as the reason the
observers fail.

The phase-conditioned source covector also continues to replay accurately on
cohort E: cell-wise zero-referenced R2 is `0.9745--0.9972`. Supplying it with
the exact fresh scalar closes all six causal endpoints. The experiment
therefore preserves the earlier component split:

```text
portable phase-conditioned task covector
  + observable source-portable scalar: not found
  + exact example-local scalar: causally sufficient.
```

This result is stronger than another observational decoding failure because
the candidate scalars were also applied through the frozen covector to the
actual continuation. Neither prediction nor intervention supports the
declared scalar sensors.

## Interpretation and stopping rule

The negative result rules out the tested retrospective construction, not all
possible nonlinear sensors. The PCA and ridge observers are deliberately
small; a high-capacity probe could memorize source structure without proving a
usable mechanism. Escalating probe capacity after seeing cohort E would also
invalidate the fresh test.

Accordingly:

- do not tune these observers on cohort E;
- do not add another residual penalty to the frozen transformer;
- do not widen the phase covector, which already transports;
- do not invoke Morse or link-cobordism machinery, because no canonical defect
  locus is needed to explain this failure; and
- move to the preregistered prospective architecture branch only if further
  work is warranted: expose a typed scalar/error channel during training in an
  invariant or equivariant sensor encoder, with matched raw and analytic
  canonicalizer controls and the same composition/extrapolation split.

Important boundaries remain. The frozen covector uses an oracle quotient-phase
chart; source scalar labels use exact diagnostic residuals; all patches are
local and off manifold; cohort E changes seeds within the established shift
families; and three selected checkpoints do not establish population
prevalence.

## Artifacts and reproduction

- campaign:
  `data/experiments/tinyllm_task_activation_scalar_sensor/20260807_d6_fresh_e/campaign_results.json`
- campaign SHA-256:
  `a9a1b910b1d5f86a834fb3f03c2b9bb050e365bb5a65b8c35b50cbbda6ea37a8`
- result SHA-256 values:
  seed 7 `2b006951c59198d8c85c0c1c6cdcf5cd0ed918b57a0e60c69611d9a75dfdd58a`;
  seed 29 `680d4ec063d88f59cfe9fb5dab8ed7088999b493b427aa0d5a8d7e53ad5743ea`;
  seed 53 `6f66840004d6a7be96ea43b339a79085568ae45dcbd5add6ffd5604a50e7a47f`
- systems-only campaign:
  `data/experiments/tinyllm_task_activation_scalar_sensor/20260807_v2_shakedown_cuda/campaign_results.json`
- systems-only SHA-256:
  `622b7457707a32056d014675a0b29a2b7e739ce73f96acdf87e4b975e33de49f`
- invalid combined-protocol source check:
  `data/experiments/tinyllm_observable_scalar_residual/20260807_source_only_shakedown_cuda/campaign_results.json`

```bash
MPLCONFIGDIR=/tmp/mpl-task-scalar-v2-primary pixi run python -m \
  experiments.structure_net.tinyllm_task_activation_scalar_sensor_v2 \
  --device cuda:1 \
  --output \
  data/experiments/tinyllm_task_activation_scalar_sensor/20260807_d6_fresh_e
```

The conservative meta-hypothesis record is stored at
`data/meta_hypotheses/tinyllm-c2-task-activation-scalar-sensor-v2.json`.
