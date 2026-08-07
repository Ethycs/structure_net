# TinyLLM relative-carrier factorization

**Status:** NOT CONFIRMED — COMPOSITIONAL CARRIER ONLY  
**Date:** 2026-08-06  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`  
**Hypothesis:** `tinyllm-relative-carrier-fixed-quotient-v1`  
**Preregistration:** `../07 - Status Reports/2026-08-06_tinyllm-relative-carrier-factorization-preregistration.md`

## Verdict

Separating carrier recovery from a fixed quotient did not create a stable global
factorization. The learned vector carrier passed its composition gate in four of
five seeds, but passed extrapolation in zero. Mean carrier alignment fell from
`0.9819` to `0.6252`, MSE rose from `0.0181` to `0.3748`, and mean measured
winding changed from `1.0` to `3.6`. The fixed quotient therefore received a
good carrier on composition and an invalid carrier outside range.

The downstream intervention was still useful: learned-vector plus fixed quotient
improved composition task accuracy by 6.11 percentage points over the scalar-only
baseline while maintaining tested branch contraction. Extrapolation improved by
only 0.76 points. This supports a support-relative carrier, not the preregistered
global factorization.

## Campaign integrity

All 20 requested d8 cells completed: four matched arms across seeds `7, 17, 29,
41, 53`. No cell failed or was reused. Training examples, minibatch schedules,
optimizer, TinyLLM architecture, evaluation families, and probe protocol were
matched. A one-worker shakedown and a two-worker CUDA concurrency pilot preceded
the full run. The campaign ran two logical workers on one NVIDIA GeForce RTX 2060
SUPER.

## Primary endpoints

| Arm | Composition alignment / MSE / degree | Extrapolation alignment / MSE / degree | Joint carrier seeds | Task accuracy comp. / extra. |
| --- | --- | --- | ---: | --- |
| analytic carrier + fixed quotient | 0.8393 / 0.1607 / 0.8 | 0.5780 / 0.4220 / -0.2 | 0/5 | 0.1238 / 0.0967 |
| learned vector carrier | 0.9819 / 0.0181 / 1.0 | 0.6252 / 0.3748 / 3.6 | 0/5 | — |
| learned vector + fixed quotient | 0.9819 / 0.0181 / 1.0 | 0.6252 / 0.3748 / 3.6 | 0/5 | 0.3758 / 0.0973 |
| scalar-only baseline | -0.3875 / 1.3875 / 1.0 | -0.1869 / 1.1869 / 3.6 | 0/5 | 0.3146 / 0.0896 |

The learned carrier’s maximum equivariance error stayed small (`1.00e-5` on
composition and `2.77e-5` on extrapolation). Thus approximate algebraic
equivariance on sampled transforms was not sufficient for correct global carrier
coordinates or degree.

## Preregistered gates

| Gate | Result |
| --- | --- |
| learned carrier passes composition and extrapolation in at least 4/5 seeds | **fail, 0/5 joint** |
| fixed quotient passes front-end and full-depth quotient endpoints in at least 4/5 seeds on both shifts | **fail, 0/5 joint** |
| fixed quotient does not materially harm task performance | **pass** |
| complete factorization | **not confirmed** |

## Interpretation and boundaries

The result localizes the failure upstream of the fixed quotient. On observed
support, a learned two-dimensional carrier is accurate, degree one, and useful to
the quotient. Outside range, its nominal equivariance relation remains numerically
tight while semantic alignment and degree fail. Future carrier claims need a
global coordinate/degree test, not equivariance error alone.

The analytic estimator was not a successful positive control under the finite,
quantized, noisy observation model. That does not show analytic relative carriers
are impossible; it shows this observation-only estimator does not meet the stated
gate.

## Artifacts and reproduction

| Artifact | Path |
| --- | --- |
| aggregate | `data/experiments/tinyllm_relative_carrier_factorization/20260806_d8_preregistered/campaign_results.json` |
| retained cells and weights | `data/experiments/tinyllm_relative_carrier_factorization/20260806_d8_preregistered/runs/` |
| aggregate SHA-256 | `2170bd93f65ee971f71c8190ef6b9d6e01487ce71e25158b805d303c1bc6e367` |

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
pixi run python -m experiments.structure_net.tinyllm_relative_carrier_factorization \
  --gpus 0 --slots-per-gpu 2 --max-parallel 2 \
  --output data/experiments/tinyllm_relative_carrier_factorization/20260806_d8_preregistered
```
