# TinyLLM acquisition-draw stability preregistration

**Status:** PREREGISTERED — NO PRIMARY DRAW GENERATED OR INSPECTED  
**Date:** 2026-08-10  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, frozen-system,
no-fit acquisition replication  
**Hypothesis:** `tinyllm-acquisition-draw-stability-v1`  
**Schema:** `nal.tinyllm-acquisition-draw-stability.v1`

## Decision question

Two stored orientation-acquisition arrays support inverse-square recovery of
the frozen TinyLLM task, but they disagree at one checkpoint at `m=64`. The
first array passed all ten systems at `m=64`; under the second array, the
learned-front-end seed-53 system required `m=256`. Those outcomes were known
before this study and are not replication units here.

This prospective study asks:

> Across fresh independent acquisition draws, how often does `m=64` recover
> both retained five-checkpoint populations, and is `m=256` a stable
> population ceiling?

No representation, denoiser, readout, observer, or model is fit. The only
estimator is the analytic circular mean.

## Locked sources

Use the orientation-noise source campaign and its frozen calibrated systems:

```text
orientation campaign
data/experiments/tinyllm_calibration_orientation_noise/
    20260807_d8_preregistered/campaign_results.json

campaign SHA-256
876af062f9d0bdd365e2f1ffbb959d9ff2e5e4e277321b510bbe6a956456877f

orientation implementation SHA-256
990975365c7f0c468c3895029c1a87a98fa14d5a28853a1fc445070769218c70

source noise-array SHA-256
b8d959169f2f249d635037a362c70522e514ace13d3bf656a91bdaea99ef43e7

calibrated-system campaign SHA-256
80623a7283fed3d902a1cbf9fb58afe2a2fef82ab86406212f907de40a96c501
```

Hard-validate the campaign, result, checkpoint, model-state, system-state,
configuration, and dataset hashes before inspecting any fresh task outcome.
The source campaign's exact-reference endpoint is the positive replay control;
its single-observation `sigma=0.175` endpoint is the inherited negative
baseline.

The two outcome-exposed predecessor acquisition seeds are `23711` and
`42700019`. The new seed tree is rooted at `81027026`; none of its sixteen
spawned draw streams may alias either predecessor array.

## Frozen systems and replication units

Retain both structured conditions:

- `analytic_calibrated`;
- `learned_calibrated_equivariant`.

Retain checkpoint seeds `7`, `17`, `29`, `41`, and `53`. TinyLLM, front ends,
scalar embeddings, layer norms, answer rows, task heads, probes, and observers
remain frozen.

The primary replication unit is one complete acquisition draw: a pair of
fresh composition/extrapolation error tensors reused identically across all
ten systems. There are exactly `16` primary draws. Checkpoint seeds are a
fixed population within each draw, not independent acquisition replicates.

## Data and intervention

Regenerate only the two locked held-out cohorts:

| Split | Seed | Regime | Examples |
| --- | ---: | --- | ---: |
| composition | `20399` | composition | `1024` |
| extrapolation | `21408` | extrapolation | `1024` |

For every unique exact-cosine fiber, draw `256` independent standard-normal
orientation errors. Both `C2` sheets in a fiber receive bit-identical errors.
The two retained counts are nested prefixes of the same draw:

```text
m in {64, 256}
sigma = 0.175 radians per observation
theta_i = theta_true + sigma * epsilon_i
q_m = normalize(sum_i (cos(theta_i), sin(theta_i)))
```

Only the two orientation fields of the eight-field observed calibration
packet change. The intervention never receives phase, cosine, target bins,
labels, activations, logits, or checkpoint identity.

Store every unique-fiber error tensor and mapping in one NPZ. Hard-audit:

- finite arrays and exact shape/dtype;
- distinct spawned draw streams;
- nested `m=64`/`m=256` prefixes;
- paired-sheet identity;
- cross-arm/checkpoint reuse;
- unit-norm aggregate orientations; and
- maximum absolute inter-draw error correlation at most `0.05`.

## Measurements

For each acquisition draw, count, checkpoint, and held-out regime, record:

- angular MAE, RMSE, signed bias, and 95th-percentile absolute error;
- exact-bin task accuracy;
- mean circular task error;
- target cross-entropy; and
- exact-bin accuracy loss from the unchanged exact-reference baseline.

For each checkpoint, the task gate passes only when accuracy loss is at most
`0.03` on **both** composition and extrapolation. Within one arm and draw, the
population gate passes at `4/5` checkpoints. A complete draw passes only when
both arms pass simultaneously.

Report checkpoint-by-draw matrices, per-arm population pass counts, complete
draw pass counts, per-checkpoint pass frequencies, and two-sided 95% Wilson
intervals. Do not pool composition and extrapolation or count the ten systems
as sixteen independent acquisition draws.

## Controls and validity

Before the primary aggregate is interpreted:

1. exact-reference task metrics must replay the source within `2e-6` for all
   ten systems;
2. every model and system state hash must remain unchanged;
3. all results and acquisition arrays must be finite and pass the declared
   acquisition contracts;
4. the locked source single-observation endpoint must remain below the
   four-of-five population gate in each arm; and
5. on fresh draw zero, a deterministic fiber-shuffled `m=256` aggregate must
   pass at most one checkpoint per arm.

Any failure classifies the campaign as `invalid`. The shuffled control is
evaluated on one predeclared draw because specificity has already replicated
on both predecessor arrays; it is not a third count or a fitted method.

## Locked endpoints and classification

The primary endpoint is the number of complete `m=256` draw passes out of 16.
The stability hypothesis passes at `>=15/16`. This means stable in the sampled
synthetic acquisition process; it does not mean a mathematical guarantee.

The `m=64` complete-draw frequency is the locked secondary endpoint. Use this
classification table in order after validity:

| Condition | Classification |
| --- | --- |
| `m256 < 15/16` | `m256_not_stable` |
| `m256 >= 15/16` and `m64 >= 15/16` | `m64_stable_population_ceiling` |
| `m256 >= 15/16` and `12 <= m64 < 15` | `m64_broadly_stable_checkpoint_variable` |
| `m256 >= 15/16` and `4 <= m64 < 12` | `m64_draw_sensitive_m256_stable` |
| `m256 >= 15/16` and `m64 < 4` | `m64_unreliable_m256_stable` |

The report must preserve the observed integer frequencies even when adjacent
classification cutoffs would support a more attractive narrative.

## Lifecycle and stopping rule

Run one lifecycle-only shakedown with one arm, one checkpoint, and one draw
rooted at the disjoint seed `81026999`. It must not use or inspect any stream
spawned from the primary root `81027026`. Do not inspect primary draw task
outcomes until the shakedown's source, array, state, control, serialization,
and exact-resume contracts pass. The shakedown is never pooled.

Then run the exact `2 arms x 5 checkpoints x 16 draws x 2 counts` primary
design. Resume may reuse only implementation- and artifact-matched completed
cells. No outcome-conditioned seed addition, threshold change, count change,
noise change, system exclusion, or model change is allowed.

After the campaign, store the report, per-system records, complete arrays,
hash manifest, and meta-hypothesis evidence in DVC and commit the DVC objects
to lakeFS. This study licenses no new training. Its purpose is to close the
sample-cost stability question before any broader acquisition model is built.

## Scope boundary

The acquisition errors are synthetic, independent, unbiased, homoscedastic,
and Gaussian. The systems are five retained checkpoints in two structured
front-end arms, not a sampled architecture population. The experiment does
not address correlated or biased sensing, real acquisition cost, other noise
scales, natural-language tasks, or arbitrary TinyLLM checkpoints.
