# TinyLLM sensor-noise law dose localization

**Status:** MEASURED — PRIMARY EVALUABLE, HYPOTHESIS NOT CONFIRMED  
**Date:** 2026-08-10  
**Hypothesis:** `tinyllm-noise-law-dose-localization-v1`  
**Schema:** `nal.tinyllm-noise-law-dose-localization.v1`  
**Classification:** `asymmetric_law_breaks_within_isotropic_window`  
**Preregistration:** [sensor-noise law dose localization](../07%20-%20Status%20Reports/2026-08-10_tinyllm-noise-law-dose-localization-preregistration.md)

## Verdict

The corrective study found a common nonzero utility-valid isotropic dose, then
rejected complete additive-noise closure at that dose. The failure is specific
to the law with a nonzero lab-frame mean; zero-mean anisotropy still passes the
registered population gate.

The deterministic selection rule chose multiplier `0.625`, corresponding to
`sigma=0.03125`. At that dose:

- isotropic noise passes `5/5` analytic and `5/5` learned checkpoints;
- lab-anisotropic noise passes `4/5` analytic and `5/5` learned checkpoints;
- lab-biased noise passes only `1/5` analytic and `3/5` learned checkpoints.

The biased-law failures are failures of natural utility relative to clean
input. They are not failures of the observed `C2` action or Reynolds twirl.
Correct action and twirl pass every arm/law/seed cell at both cuts and both
shifts, while all matched target-changing controls fail.

Therefore:

> The frozen quotient computation remains functionally `C2`-closed conditional
> on a noisy identity, but a persistent lab-frame sensor mean can move that
> identity outside the complete system's utility tolerance even at a dose where
> matched-energy isotropic and zero-mean anisotropic error are tolerated.

This is a valid corrective negative result, not an independent replication of
the known `sigma=0.05` study.

## What was held fixed

The study reused the exact ten retained d8/N3 calibrated systems, seeds
`7, 17, 29, 41, 53`, 512-example composition and extrapolation cohorts,
calibration packets, token decoder, task, batch size 256, gates, observed action,
twirl, and target-changing controls.

The only changed quantity was a scalar multiplier applied elementwise to the
stored `sigma=0.05` error arrays. No random draw was regenerated. No model,
front end, head, action, observer, probe, or noise process was trained or
fitted.

## Stage 1: isotropic dose localization

Dose selection inspected only isotropic natural utility. A seed passed a dose
only when accuracy, circular error, and target cross-entropy all passed on both
composition and extrapolation. An arm required at least four passing seeds. A
dose was selectable only if it and every smaller registered positive dose
passed in both arms.

| Multiplier | Sigma | Analytic | Learned equivariant | Joint prefix-valid |
| ---: | ---: | ---: | ---: | :---: |
| `0.000` | `0.00000` | 5/5 | 5/5 | zero control |
| `0.125` | `0.00625` | 5/5 | 5/5 | yes |
| `0.250` | `0.01250` | 5/5 | 5/5 | yes |
| `0.375` | `0.01875` | 5/5 | 5/5 | yes |
| `0.500` | `0.02500` | 5/5 | 5/5 | yes |
| **`0.625`** | **`0.03125`** | **5/5** | **5/5** | **selected** |
| `0.750` | `0.03750` | 2/5 | 5/5 | no |
| `1.000` | `0.05000` | 0/5 | 4/5 | no; known source result |

The zero-dose posterior replay error is exactly zero in all ten systems. The
ladder is monotone at the population level and reproduces the source
`sigma=0.05` pass counts exactly, so the selected dose is neither an inserted
point nor a favorable asymmetric-law outcome.

## Stage 2: locked law-shape comparison

### Joint population gates

| Arm | Law | Natural utility | Action, both cuts | Twirl, both cuts | Target-changing control |
| --- | --- | ---: | ---: | ---: | ---: |
| analytic | isotropic | **5/5** | 5/5 | 5/5 | 0/5 |
| analytic | lab-anisotropic | **4/5** | 5/5 | 5/5 | 0/5 |
| analytic | lab-biased | **1/5** | 5/5 | 5/5 | 0/5 |
| learned equivariant | isotropic | **5/5** | 5/5 | 5/5 | 0/5 |
| learned equivariant | lab-anisotropic | **5/5** | 5/5 | 5/5 | 0/5 |
| learned equivariant | lab-biased | **3/5** | 5/5 | 5/5 | 0/5 |

For every law, the joint seed count equals the natural-utility seed count.
No correct action or twirl gate removes another seed. The registered primary
fails only because the biased law misses `4/5` in both arms.

### Natural accuracy loss

The exact-bin accuracy ceiling is 5 percentage points. Median losses are:

| Arm | Law | Composition | Extrapolation | Accuracy-pass seeds by shift |
| --- | --- | ---: | ---: | ---: |
| analytic | isotropic | 3.71 pp | 2.44 pp | 5/5, 5/5 |
| analytic | lab-anisotropic | 3.91 pp | 3.52 pp | 4/5, 4/5 |
| analytic | lab-biased | **7.23 pp** | 2.44 pp | **1/5**, 5/5 |
| learned equivariant | isotropic | 1.86 pp | 0.68 pp | 5/5, 5/5 |
| learned equivariant | lab-anisotropic | 1.66 pp | 1.17 pp | 5/5, 5/5 |
| learned equivariant | lab-biased | **4.98 pp** | 0.88 pp | **3/5**, 5/5 |

The biased-law population failure occurs on composition only. Median circular
and cross-entropy increases remain inside their gates; the discontinuous
exact-bin endpoint is decisive. The learned temporal encoder is more naturally
robust than the endpoint-based analytic front, but it still misses the locked
four-seed biased-law threshold.

### Conditional functional closure

Across all 120 correct action cells and 120 correct twirl cells formed by ten
systems, three laws, two shifts, and two cuts:

- maximum action accuracy loss: `0.008789`;
- maximum twirl accuracy loss: `0.008789`;
- maximum action posterior JS: `0.0009770`;
- maximum twirl posterior JS: `0.0002443`.

All population action/twirl counts are `5/5`; all orthogonal action/twirl
control counts are `0/5`. Analytic feature action error is at most
`2.086e-7`.

This establishes a clean separation:

`measurement robustness relative to clean input != functional group closure relative to noisy identity`.

The biased observation can be harmful before the action is applied while the
frozen computation remains equivariant/invariant to the declared action around
that observation.

## Integrity and lifecycle

| Contract | Result |
| --- | ---: |
| requested/completed Stage 1 systems | 10/10 |
| requested/completed Stage 2 systems | 10/10 |
| failed / excluded / retried | 0 / 0 / 0 |
| trained or fitted objects | 0 |
| zero-dose maximum posterior replay error | `0` |
| maximum source clean replay error | `1.4603e-6` |
| maximum cut replay error | `0` |
| maximum analytic feature action error | `2.0862e-7` |
| finite outputs | 20/20 stage results |
| frozen states unchanged | 20/20 stage results |
| exact completed-tree resume | byte-identical |

The source campaign, runner, stored error arrays, prerequisite campaigns,
checkpoint digests, cohort hashes, and preregistration are all pinned and
verified before evaluation.

## Interpretation

### Supported

1. A common isotropic utility-valid window exists for the two frozen structured
   systems under these draws; its largest registered prefix-valid point is
   `sigma=0.03125`.
2. Matched-energy, zero-mean lab anisotropy is tolerated at the registered
   population threshold in both arms.
3. A nonzero lab-frame mean is not tolerated at that same dose, especially on
   the composition exact-bin endpoint.
4. The observed `C2` action and twirl remain functionally closed under every
   tested law, including the biased law. The utility failure precedes and is
   independent of that closure measurement.

### Not supported

1. The primary claim that all three additive laws close at the selected dose.
2. A general claim that reflection asymmetry is harmful: the reflection-
   asymmetric zero-mean anisotropic law passes. The sharper implicated property
   is the persistent lab-frame mean.
3. A population noise-radius estimate. One deterministic draw pair is scaled.
4. A claim that the learned front is fully bias-robust. It improves the counts
   but remains below `4/5`.
5. Any repair by retraining, denoising, or new objective pressure.

## Decision

Close the generic additive-noise-shape question at this dose. The result has
already separated zero-mean covariance shape from nonzero mean, so another
law sweep or representation penalty would add little.

The shortest next causal diagnostic is a frozen **bias-component
decomposition** using the same selected arrays:

1. deterministic mean-only error;
2. centered stochastic-only error;
3. their registered full sum;
4. the sign-reversed mean with the same stochastic component.

Evaluate natural utility only, then use the already validated action/twirl as a
mechanistic control rather than the primary endpoint. If mean-only reproduces
the composition failures and sign reversal moves them across phase bins, the
effect is a directional sensor/readout calibration defect. If only the full
sum fails, the mechanism is a nonlinear mean-noise interaction. If centered
noise fails too, the current attribution to persistent bias is wrong.

Do not train a denoiser before this decomposition. The generator exposes the
components exactly, and the intervention can falsify the simpler explanation
with the same checkpoints and no fitted parameters.

## Artifacts and reproduction

| Item | SHA-256 / value |
| --- | --- |
| campaign | `9b05823ebdb88bd828f27699da596dc5e7dcf0c4af5e13f1664fa70e5111f9bd` |
| exact primary tree | `cd908e386404a87d3dc0e47335f8d9b6ac5d7f49a1d74c6b29a2e911ed8d1387` |
| composite implementation | `bab495e0f3985c8358d90344fc3cf02986b6e138adaeb9fa01c1d38c482187c2` |
| runner | `39a72dd535f96f13bae644c74096b298b85fb8587d980211dc489ed463aeb725` |
| preregistration | `79913c913c7f6f41714400fd4337224f039be0466541ec0a7f26736c599b7a4a` |
| result manifest | `976545c812e428ea4b020ca46a88643cb741a6ad5c7797389e9a5e6ca81f7562` |
| selected arrays | `740c5c30f01c482fa799db1865a11c069ad3b59f474879a59f1906b94f4130f3` |
| selected law contract | `89ddaa6d726d89767e34bc2efb4ef75af8cbe098528da0f214473d01a51ac1f5` |
| source campaign | `868ad0ffee546f157e701790c34a83f20bfb3116e78b2f8c5bc34dd7bfe660d7` |
| systems-only shakedown | `1fdc77a787223d0808047115fc79e76b1cb9b62973e8a5edba19b48ba6e3862b` |
| meta-hypothesis record | `531fd058c987c4cfa52becbfe2f2a6fa4b814efb699873bc2a28a89d1da54e5b` |
| DVC data root | `c07286d2b9710cd68228cd21f487e425.dir` |
| lakeFS commit | `d4fb92ef41e39d0cc672d672e55c9192ea0e9dcf01597b1a549efcf973577061` |
| device recorded by PyTorch | NVIDIA GeForce RTX 2060 (`cuda:2`) |
| peak CUDA allocation | `293,665,792` bytes |
| campaign analysis time | `313.76` seconds |

- primary:
  `data/experiments/tinyllm_noise_law_dose_localization/20260810_d10_preregistered/`
- systems-only shakedown:
  `data/experiments/tinyllm_noise_law_dose_localization/20260810_shakedown_analytic_cuda/`

```bash
MPLCONFIGDIR=/tmp/matplotlib-noise-dose-primary \
pixi run python -m \
  experiments.structure_net.tinyllm_noise_law_dose_localization \
  --device cuda:2 \
  --output \
  data/experiments/tinyllm_noise_law_dose_localization/20260810_d10_preregistered
```

## Boundaries

The study covers one synthetic calibrated task, two structured front ends,
five seeds, two fixed cohorts, three lab-frame planar error laws, eight nested
multipliers of one stored draw pair, and one selected-dose law comparison. It
does not test fresh noise draws, temporal correlation, calibration-packet
error, other bias directions, learned correction, other groups, natural
language, or real sensors. Scaling a fixed draw isolates dose cleanly but does
not estimate deployment frequency or a universal robustness radius.
