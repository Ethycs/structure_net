# TinyLLM gauge-repaired I/O correspondence test

**Status:** PARTIAL SUPPORT — observation-side gauge repair succeeds; the
declared target-side repair is identifiable but is not stably realized

**Date:** 2026-08-06

**Hypothesis:** `tinyllm-io-correspondence-gauge-descent-v1`

**Conformance:** The corrected result follows the
[preregistration and disclosed Amendment A](../07%20-%20Status%20Reports/2026-08-06_tinyllm-io-correspondence-preregistration.md).
The amendment was necessary after affected outcomes had been observed, so the
corrected learned-uncalibrated result is not presented as pristine
preregistered confirmation.

## Verdict

The full prediction was **not confirmed**. Both valid task relations passed the
data-level descent contract, but only observation-side gauge repair produced a
stable internal quotient.

- Calibrated analytic canonicalizer: joint gate **5/5 seeds**.
- Calibrated learned-equivariant encoder: joint gate **5/5 seeds**.
- Uncalibrated relative target with corrected 2-D equivariant encoder: **0/5**.
- Uncalibrated absolute-cosine negative control: **0/5**, as predicted.

At full depth, the calibrated learned arm achieved mean `(base correlation,
conditional branch accuracy) = (0.999, 0.503)` on composition and
`(0.987, 0.505)` on extrapolation. The corrected relative learned arm achieved
only `(0.776, 0.910)` and `(0.376, 0.749)`, respectively. It therefore failed
both horizontal base preservation and vertical fiber descent.

The causal conclusion is narrower and stronger than “identifiability solves
generalization”:

> Identifiability is necessary, but a target that descends to observations is
> not sufficient for a particular equivariant architecture to realize its
> global base-and-fiber geometry.

## Data-level I/O relation contract

The experiment treated the fibered relation

```text
X <- Gamma_T -> Q_T
```

as the reference object. An exhaustive 4,032-pair gauge grid, backed by the
symbolic generator identity, established:

| Relation | Descent result |
| --- | --- |
| uncalibrated absolute `cos(phi)` | deliberate failure: 4,032 identical-observation target-changing counterexamples |
| calibrated absolute `cos(phi)` | passed: zero violations; minimum target-changing calibration distance 0.04999 |
| uncalibrated relative `cos(phi + theta)` | passed: zero violations |

The maximum numerical error between gauge-related uncalibrated observations
was `1.02e-15`. The relative target is valid because
`(phi + alpha) + (theta - alpha) = phi + theta`; it uses no latent phase or
target side channel.

This contract separates mathematical well-definedness from learned
realization. The relative failure is not another observation-gauge
impossibility.

## Campaign integrity and correction

The original schema-v1 campaign completed 35/35 cells but exposed only the
x-component of an internally equivariant vector in both uncalibrated learned
arms. A scalar coordinate is not an SO(2)-equivariant representation. That
artifact remains preserved and its affected learned-arm verdict was rejected.

Schema v1.1 exposes the complete two-dimensional vector and passes the exact
contract `E(R_alpha x) = R_alpha E(x)`. It reran every cell in a new append-only
root rather than pooling implementations:

- 35/35 corrected-schema cells completed; zero failures;
- 20 newly trained uncalibrated cells;
- 15 frozen calibrated source checkpoints, all read back by checkpoint,
  architecture, initialization, data, minibatch, and implementation digest;
- two CUDA workers on PyTorch logical GPU 1;
- seven-arm single-worker and two-worker shakedowns passed first;
- implementation digest
  `f3c20526437f2e9e41329307b61f974a226048ff6d6d0fc65ee7be13220c6354`.

The calibrated primary endpoints had been observed in the preceding campaign
and are retained positive-control evidence. Their new post-attention,
post-MLP, Mapper, and paired-distortion measurements were not previously
available.

## Fixed design

- d8 TinyLLM, 50,965,504 transformer parameters and a common 27-position
  capacity;
- seeds 7, 17, 29, 41, and 53;
- N3, 4,096 shared examples, batch 64, 600 task-only updates;
- AdamW `3e-4`, weight decay `0.01`, gradient clip `1.0`;
- matched transformer initialization, latent/sensor base, and minibatch schedule
  within each seed;
- raw, analytic-calibrated, learned calibrated-invariant, and corrected learned
  uncalibrated-equivariant front ends;
- fresh nonlinear conditional probes on disjoint train/validation/test cohorts;
- front-end, post-attention block 1, post-MLP block 1, and full-depth cuts;
- composition and outside-range extrapolation as the primary shifts.

Success required every primary cut and shift in the same seed to satisfy
`corr >= 0.90`, conditional branch balanced accuracy `<= 0.55`, and conditional
log-loss gain `<= 0.02`, in at least four of five seeds. A repaired learned arm
also could not trail matched raw task accuracy by more than three points.

## Primary endpoints

The table gives five-seed mean `(cosine correlation, conditional branch
balanced accuracy)` and the all-cuts/all-shifts joint seed count.

| Arm | Front-end composition | Front-end extrapolation | Full composition | Full extrapolation | Joint |
| --- | --- | --- | --- | --- | ---: |
| Uncalibrated absolute raw | `(0.961, 0.998)` | `(0.571, 0.886)` | `(0.974, 0.617)` | `(0.497, 0.502)` | 0/5 |
| Uncalibrated absolute equivariant | `(0.580, 0.844)` | `(0.215, 0.707)` | `(0.602, 0.854)` | `(0.253, 0.691)` | 0/5 |
| Calibrated absolute raw | `(0.959, 0.998)` | `(0.001, 0.498)` | `(0.972, 0.616)` | `(0.481, 0.502)` | 0/5 |
| Calibrated absolute analytic | `(0.972, 0.501)` | `(0.964, 0.499)` | `(0.998, 0.500)` | `(0.992, 0.497)` | 5/5 |
| Calibrated absolute equivariant | `(0.972, 0.498)` | `(0.960, 0.501)` | `(0.999, 0.503)` | `(0.987, 0.505)` | 5/5 |
| Uncalibrated relative raw | `(0.967, 0.999)` | `(0.625, 0.909)` | `(0.993, 0.523)` | `(0.515, 0.518)` | 0/5 |
| Uncalibrated relative equivariant | `(0.756, 0.922)` | `(0.365, 0.781)` | `(0.776, 0.910)` | `(0.376, 0.749)` | 0/5 |

The relative raw model shows the predicted support-local pattern: full-depth
composition passed the base/fiber endpoint in 4/5 seeds, while extrapolation
passed in 0/5 because base correlation fell to 0.515. The corrected learned
front end did worse: it retained a strongly decodable branch at every cut and
never reached the 0.90 base threshold.

Conditional log-loss gains were approximately zero for the successful
calibrated arms. They were large under relative composition for the learned
arm (`0.499` at full depth), confirming that its branch accuracy was not merely
an imbalanced-label artifact.

## Task control

| Arm | ID accuracy | Composition | Extrapolation |
| --- | ---: | ---: | ---: |
| Calibrated raw | 0.490 | 0.389 | 0.130 |
| Calibrated analytic | 0.751 | 0.745 | 0.616 |
| Calibrated learned | 0.738 | 0.717 | 0.492 |
| Relative raw | 0.619 | 0.625 | 0.130 |
| Relative learned | 0.322 | 0.315 | 0.090 |

The calibrated learned arm beat its raw control by 32.8 composition points and
36.2 extrapolation points. The relative learned arm trailed raw by 31.1 points
on composition and 4.0 points on extrapolation, failing the preregistered
three-point task-control floor as well as the representation gate.

## Target-lens Mapper result

The calibrated structured arms produced the desired interval-like Mapper:

- analytic: 5/5 interval-like maps at every cut and primary shift;
- learned calibrated: 5/5 except full composition, where 4/5 passed;
- structured front-end interior single-sheet fraction: 1.00 in almost every
  primary cell.

However, the finite Mapper was not a sufficient fiber test. Raw post-attention,
post-MLP, and full-depth representations frequently produced 5/5 interval-like
maps even when the conditional branch probe remained materially above chance.
For example, calibrated raw full composition was Mapper interval-like in 5/5
seeds while branch accuracy was 0.616. Relative learned full composition was
also Mapper interval-like in 5/5 while branch accuracy was 0.910.

This is a useful negative methodological result: the declared cover/kNN Mapper
captures coarse target-ordered connectivity, but can join two thin, decodable
branch sheets. It must be reported alongside conditional fiber measurements and
cannot certify descent by itself.

## Paired-map distortion

At the front-end output, fiber-averaged whitened distortion cleanly separated
the calibrated structured relation from raw observation geometry:

| Arm | Composition `D` | Extrapolation `D` | Extrapolation distance-order rho |
| --- | ---: | ---: | ---: |
| Calibrated raw | 0.331 | 0.337 | 0.175 |
| Calibrated analytic | 0.015 | 0.029 | 0.985 |
| Calibrated learned | 0.018 | 0.038 | 0.975 |
| Relative raw | 0.327 | 0.347 | 0.128 |
| Relative learned | 0.104 | 0.242 | 0.219 |

The corrected relative encoder improved local target ordering relative to raw
but did not make it stable: extrapolation distortion more than doubled and
distance-order correlation fell to 0.219. At full depth its extrapolation
distortion was 0.338 with rho 0.146, consistent with the failed base probe.

Depth did not monotonically reduce the distortion of successful calibrated
representations. The analytic extrapolation `D` rose from 0.029 at the front end
to 0.267 at full depth even while cosine correlation rose from 0.964 to 0.992.
Thus tested decodability, target-order distortion, and coarse Mapper topology
are distinct geometric measurements.

## Throughput observation

Effective training input throughput on the shared GPU was approximately
10.5k–12.7k tokens/s for raw 27-token contexts and 2.4k–2.8k tokens/s for
structured 3-token contexts. The corrected relative-equivariant arm averaged
2,734 tokens/s (2,705–2,770 across seeds). These are bookkeeping rates from
shared-run training time, not isolated model-throughput benchmarks; front-end
and optimizer work do not scale with transformer token count.

## Interpretation and next experiment

The study resolves four claims:

1. The absolute target still fails to descend without calibration.
2. Observation-side calibration plus analytic or architectural use of the
   reference realizes the same base-and-fiber geometry through every cut.
3. The relative target is mathematically identifiable, but this corrected
   equivariant encoder does not learn its global map and actively retains the
   branch cover.
4. Finite Mapper connectivity can look quotient-like despite substantial
   conditional branch information.

The next experiment should not alter representation penalties. It should test
the target-side repair with an analytic sensor-frame positive control and a
vector-valued equivariant token preserved into the task head, separating:

- whether the relative coordinate is recoverable analytically under the full
  nuisance generator;
- whether the learned temporal vector estimates the correct future sensor-frame
  vector;
- whether scalar interval supervision causes the model to keep an unnecessary
  branch cover.

If the analytic relative control fails, the target or generator specification
is inadequate. If it passes while learned vector regression fails, the problem
has moved to approximation/optimization rather than identifiability.

## Artifacts and reproduction

- Corrected aggregate:
  `data/experiments/tinyllm_io_correspondence/20260806_d8_corrected_equivariant/campaign_results.json`
- Corrected per-seed results and new weights:
  `data/experiments/tinyllm_io_correspondence/20260806_d8_corrected_equivariant/runs/`
- Preserved nonconformant original:
  `data/experiments/tinyllm_io_correspondence/20260806_d8_preregistered/`
- Preregistration and amendment:
  `docs/07 - Status Reports/2026-08-06_tinyllm-io-correspondence-preregistration.md`
- Base runner: `experiments/structure_net/tinyllm_io_correspondence.py`
- Correction runner: `experiments/structure_net/tinyllm_io_correspondence_v2.py`
- Tests: `tests/structure_net/test_tinyllm_io_correspondence.py` and
  `tests/structure_net/test_tinyllm_io_correspondence_v2.py`

```bash
MPLCONFIGDIR=/tmp/matplotlib-io pixi run python -m \
  experiments.structure_net.tinyllm_io_correspondence_v2 \
  --gpus 1 --slots-per-gpu 2 --max-parallel 2 \
  --output data/experiments/tinyllm_io_correspondence/20260806_d8_corrected_equivariant
```
