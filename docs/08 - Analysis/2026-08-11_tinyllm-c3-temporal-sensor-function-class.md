# TinyLLM C3 temporal sensor function-class preflight

**Status:** PREREGISTERED NO-TRAINING PREFLIGHT PASSED

**Date:** 2026-08-11

**Hypothesis:** `tinyllm-c3-temporal-sensor-function-class-v1`

**Classification:** `existing_c3_sensor_contains_analytic_carrier_and_task_gradient`

**Preregistration:** [C3 temporal sensor function class](../07%20-%20Status%20Reports/2026-08-11_tinyllm-c3-temporal-sensor-function-class-preregistration.md)

**Primary artifact:** `data/experiments/tinyllm_c3_temporal_sensor_function_class/20260811_preregistered/result.json`

## Verdict

The existing 184-parameter exact-`C3` sensor family contains the analytic
carrier required by the frozen temporal operator and fixed interval decoder.
A closed-form state with only five nonzero parameters reproduces the analytic
complex carrier to less than `7e-7` on both registered shifts. The complete
fixed task passes composition and extrapolation, while the fixed target-roll
control is null.

At a separate deterministic random initialization, the true task reaches the
sensor through the frozen operator and decoder. The total gradient norm is
`49.6855`, `183/184` scalar parameters have nonzero numerical gradients, and a
normalized `1e-3` negative-gradient perturbation lowers the same-batch loss by
`.04554`. The state is restored exactly afterward.

The preregistered result therefore licenses one sensor-only five-seed campaign.
It does not show that gradient descent will find the witness.

## Constructive capacity result

The witness uses the exact identity

```text
GELU(x) - GELU(-x) = x.
```

Two hidden units reconstruct each corrected observed scalar. One output
channel takes their difference; one real mixer coefficient selects that
channel. The existing character projection then computes the first deck
character, and cubing removes the `C3` sheet. Every other parameter is zero.

| Measurement | Composition | Extrapolation | Gate |
| --- | ---: | ---: | ---: |
| carrier maximum absolute error | `6.346e-7` | `6.832e-7` | `<= 2e-6` |
| maximum nonidentity action error | `6.977e-7` | `6.796e-7` | `<= 2e-6` |
| temporal target correlation | `.999940` | `.999931` | `>= .99` |
| temporal target RMSE | `.007755` | `.008258` | `<= .08` |
| fixed-decoder exact accuracy | `.96216` | `.95972` | `>= .50/.35` |
| fixed-decoder cross-entropy | `1.28106` | `1.28308` | `<= 1.80/2.20` |
| fixed-decoder coverage | `16` | `16` | `>= 14/12` |
| target-roll correlation | `-.00722` | `.00715` | absolute `<= .10` |
| target-roll RMSE | `1.00579` | `.99078` | `>= .80` |

The shared scalar reconstruction error on 10,001 points over `[-4,4]` is
`2.384e-7`. Neither 4,096-example cohort has a quantizer saturation.

## Gradient-route result

The derivative audit uses 512 paired examples and a fresh random encoder,
without TinyLLM or a checkpoint:

```text
sensor
  -> q7 * conjugate(q6) * q7
  -> fixed interval posterior
  -> true target cross-entropy.
```

| Measurement | Result | Gate |
| --- | ---: | ---: |
| initial target loss | `8.888625` | finite |
| total parameter-gradient norm | `49.68547` | `>= 1e-6` |
| nonzero numerical gradients | `183/184` (`.99457`) | `>= .90` fraction |
| deck-rolled loss error | `3.020e-9` | `<= 1e-6` |
| deck-rolled maximum gradient error | `7.629e-6` | `<= 2e-5` |
| local diagnostic loss decrease | `.045542` | `>= 1e-4` |
| state before/after restoration | identical SHA-256 | exact |

The final linear bias is nearly null because a channel-shared constant cancels
under the nontrivial character projection. Its tiny float32 gradient does not
block the active witness path and was not removed post hoc.

## What this settles

### Supported

- The predecessor's learned sensor family is expressive enough; a sensor
  redesign is not required to represent the analytic quotient carrier.
- The fixed temporal and metric interface is differentiable back to the
  sensor under the true task.
- Exact `C3` invariance holds for the function class and for the parameter
  gradient route under a deck-transformed batch.
- The previous trained-continuation failure cannot be attributed to a missing
  analytic carrier in the 184-parameter sensor family.

### Not supported

- The preflight does not establish that the sensor can be learned from random
  initialization in five seeds.
- It does not establish sample efficiency, optimizer stability, or recovery of
  the analytic coordinate up to the declared metric tolerance.
- It does not rehabilitate the stopped raw or learned TinyLLM arms.
- The five-parameter witness is a capacity certificate, not a proposed manual
  initialization for the confirmatory campaign.

## Program decision

Proceed to the smallest prospective learned test:

```text
learned 184-parameter exact-C3 sensor
  -> frozen analytic temporal operator
  -> frozen physical interval decoder.
```

Run true and pair-preserving target-shuffled arms for seeds
`(7, 17, 29, 41, 53)`. Retain the simultaneous composition/extrapolation task
gate and require true success in at least four seeds with at most one shuffled
success. Do not instantiate or train TinyLLM. Do not warm-start from the
closed-form witness; it remains only the analytic positive control.

## Integrity and reproduction

The primary preflight uses CPU only, performs zero optimizer steps, instantiates
zero TinyLLM models, loads zero checkpoints, and restores the sole local
derivative perturbation byte-for-byte.

```bash
MPLCONFIGDIR=/tmp/mpl-c3-sensor-preflight pixi run python -m \
  experiments.structure_net.tinyllm_c3_temporal_sensor_function_class \
  --output \
  data/experiments/tinyllm_c3_temporal_sensor_function_class/20260811_preregistered/result.json

MPLCONFIGDIR=/tmp/mpl-c3-sensor-preflight pixi run pytest -q \
  tests/structure_net/test_tinyllm_c3_temporal_sensor_function_class.py
```

| Artifact | SHA-256 |
| --- | --- |
| result | `6a01db25ebc2ed15d202884c39f16db685d5218647b0bb209e2e5a737696a383` |
| runner | `95331fa823f193449af1c46f961f22f4aaf902300695f179912b75d8ed4bcade` |
| preregistration | `83f20896853cb7612618c93d3302a5bfa0b2091b6c957625eecbe3451401701f` |

Five focused tests pass. The result JSON records and verifies the three pinned
source hashes, all numeric gates, the complete gradient audit, and zero-training
accounting.
