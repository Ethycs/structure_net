# TinyLLM observable C3 temporal-quotient Stage 0

**Status:** SYSTEMS LIFECYCLE PASSED — PRIMARY TRAINING STILL UNAUTHORIZED

**Date:** 2026-08-11

**Evidence role:** `systems_lifecycle_only_not_scientific_evidence`

**Parent hypothesis:** `tinyllm-c3-temporal-quotient-training-v1`

**Frozen registration:** [C3 temporal Stage 0](../07%20-%20Status%20Reports/2026-08-11_tinyllm-c3-temporal-quotient-stage0-registration.md)

## Verdict

The matched raw, analytic, and exactly invariant learned-`C3` implementations
pass the registered CPU lifecycle. The analytic d6 implementation also passes
the registered CUDA shakedown on an NVIDIA GeForce RTX 3060. Uninterrupted and
checkpoint-resumed training are tensor-exact for the system state, optimizer
state, loss history, and evaluation posteriors.

The locked classification is:

```text
c3_temporal_quotient_stage0_passed
```

No primary cell ran. The result validates the execution path and licenses a
numeric d6 preregistration; it does not establish that raw or learned TinyLLM
discovers a quotient.

## Prospective data contract

The preflight had fixed composition and extrapolation but not the training
support. Stage 0 froze that missing boundary before model outcomes:

- phase, signed speed, and deck element vary throughout training;
- amplitude, offset, and drift appear one family at a time, with the inactive
  values fixed at `A=1.2`, `o=0`, and `d=0`;
- composition combines all three calibration nuisances independently inside
  the same marginal ranges;
- extrapolation uses the preflight's wider speed and calibration ranges.

Every training history appears under two distinct deck elements with one shared
target and calibration packet. Every minibatch contains complete pairs. The
protocol validator found zero action mismatches, zero saturation, zero paired
target/calibration error, and zero exact fixed points in the target-changing
derangement. Its mean absolute target change was `.80343`.

## CPU lifecycle

All arms ran two uninterrupted optimizer steps and an independently initialized
one-step-plus-reload continuation on the `tiny` systems preset.

| Arm | TinyLLM params | Injection | Learned encoder | Largest nonidentity deck change | Exact resume |
| --- | ---: | ---: | ---: | ---: | --- |
| raw | `33,984` | `128` | `0` | `1.7385` | pass |
| analytic | `33,984` | `96` | `0` | `1.223e-15` | pass |
| learned `C3` | `33,984` | `96` | `184` | `9.984e-7` | pass |

The raw feature therefore retains sheet identity, while both structured
features satisfy the declared invariance tolerance. The learned encoder also
passed at a deterministic perturbed parameter state and after optimization,
so invariance is a function-class property rather than an initialization
accident.

All three arms had the same initial TinyLLM digest. Analytic and learned arms
also had the same sequence-injection digest. Data and minibatch hashes were
identical across arms.

For every arm, the resumed and uninterrupted paths had:

- identical system-state digests;
- identical optimizer-state digests;
- identical loss and gradient-norm histories;
- maximum posterior difference `0.0` on composition and extrapolation; and
- finite gradients plus nonzero parameter change.

The two-step task values are intentionally omitted from interpretation: the
tiny preset is a lifecycle vehicle, not a quality experiment.

## Analytic d6 CUDA shakedown

The only d6 cell was the registered analytic positive control. It used 512
training examples, 64 optimizer steps, batch size 64, and an exact resume at
step 32. The model contained `29,950,080` TinyLLM parameters plus a `1,152`
parameter sequence injection.

| Contract | Result |
| --- | ---: |
| system-state equality after resume | exact |
| optimizer-state equality after resume | exact |
| loss-history equality after resume | exact |
| composition posterior difference | `0.0` |
| extrapolation posterior difference | `0.0` |
| analytic invariance before training | `1.223e-15` |
| analytic invariance after training | `1.223e-15` |
| first-step loss | `2.87458` |
| step-64 loss | `1.73059` |

The underpowered diagnostic task metrics were:

| Regime | Exact-bin accuracy | Mean correlation | RMSE | Cross-entropy | Bin coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| composition | `.28125` | `.96997` | `.20610` | `1.79623` | `15/16` |
| extrapolation | `.11328` | `.71176` | `.54616` | `3.04063` | `13/16` |

These values are not model evidence. They show that the analytic route learns
the supported temporal map quickly while 64 steps do not solve the wider speed
shift. The 600-step primary gate must therefore require high extrapolation
correlation and materially better exact-bin utility rather than grandfathering
the shakedown values.

## What is validated

- The observable training generator implements a real held-out nuisance
  composition and produces exact paired deck sheets.
- Raw, analytic, and learned arms consume matched scientific material.
- The learned front end is exactly `C3` invariant for arbitrary tested weights,
  not merely encouraged by a loss.
- All arms perform real forward, backward, optimizer, checkpoint, reload, and
  continuation operations.
- The d6 analytic CUDA path is deterministic under the registered runtime.

## What remains untested

- Five-seed natural task adequacy.
- Whether task loss discovers a useful learned invariant carrier.
- Conditional deck leakage under fresh probes.
- Frozen causal orbit-barycenter closure and specificity.
- Raw-model Reynolds/Jensen synthesis.
- Any d10 architecture claim.

## Decision

Freeze the numeric d6 preregistration and implement it in a separate campaign
runner that imports this unchanged Stage-0 source by hash. Do not edit the
Stage-0 runner in place, do not run d10, and do not tune the learned encoder
against these shakedown values.

## Reproduction

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
pixi run pytest -q \
  tests/structure_net/test_tinyllm_c3_temporal_quotient_training.py \
  tests/structure_net/test_tinyllm_c3_temporal_quotient_preflight.py

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
pixi run python -m \
  experiments.structure_net.tinyllm_c3_temporal_quotient_training \
  --stage0 cpu \
  --output /tmp/tinyllm-c3-temporal-quotient-stage0-cpu.json

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/matplotlib-c3-stage0 \
pixi run python -m \
  experiments.structure_net.tinyllm_c3_temporal_quotient_training \
  --stage0 cuda \
  --cpu-result /tmp/tinyllm-c3-temporal-quotient-stage0-cpu.json \
  --output /tmp/tinyllm-c3-temporal-quotient-stage0.json
```

| Artifact | SHA-256 |
| --- | --- |
| Stage-0 registration | `c50c7fd3c437f3e4e14ddae672d5a5b251ea0b9589c73d717867fd824987e892` |
| Stage-0 runner | `dbc934190b5f725185ff9a99690409ac63fc4f16bd04686f870e7ef21cba03a6` |
| CPU lifecycle JSON | `d0b59903ae1fb4a099d4b632e18f6366b36d46737cd594caf51d19dac42b314f` |
| CUDA lifecycle JSON | `54dbfa3432eb57d2ea39645922249b26b3d4f37a0faedfad55f7081fb1167c5d` |

The JSON and checkpoints are disposable `/tmp` systems artifacts. They are not
placed in the DVC scientific evidence root.
