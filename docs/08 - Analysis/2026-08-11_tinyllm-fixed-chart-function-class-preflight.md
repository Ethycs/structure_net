# TinyLLM fixed-chart function-class preflight

**Status:** VALID NO-TRAINING ARCHITECTURE PREFLIGHT — EXACT CHART REDUCES TO A FROZEN SCALAR SPINE

**Date:** 2026-08-11

**Conformance:** NAL-STD-EXPERIMENT `DESIGN / NO-TRAINING ARCHITECTURE PREFLIGHT`

**Hypothesis:** `tinyllm-fixed-chart-function-class-v1`

**Frozen design:** [fixed-chart function-class preflight](../07%20-%20Status%20Reports/2026-08-11_tinyllm-fixed-chart-function-class-preflight.md)

## Verdict

Sign, order, range, and endpoint constraints do not type the exact physical
cosine chart. Two explicit smooth monotone maps satisfy all four constraints
but change more than 71% of the fixed sixteen-bin assignments on a resolved
interval grid. The only tested function class that makes the public physical
scalar non-regaugeable is a typed split whose scalar is the immutable analytic
canonicalizer and whose learned carrier is a separate field.

The locked classification is:

```text
exact_physical_chart_requires_frozen_scalar_spine
```

No optimizer step ran and no TinyLLM checkpoint was loaded. This is an
architecture-contract result, not evidence that the auxiliary carrier is
useful or that a new model performs better.

## Validity

| Check | Result |
| --- | ---: |
| calibrated source hash | pass |
| fixed interval-interface hash | pass |
| observation/target identifiability | pass |
| analytic composition fidelity | corr `.998108`, RMSE `.033732`, pass |
| analytic extrapolation fidelity | corr `.991184`, RMSE `.072643`, pass |
| monotone counterexample gate | `2/2` |
| typed scalar/auxiliary split | pass |
| optimizer steps / checkpoint loads | `0 / 0` |
| focused tests | `2 passed` |

The analytic checks used 4,096 fresh composition and 4,096 fresh
extrapolation observations. Their gates were fixed at correlation `.99` and
RMSE `.05/.08` before execution.

## Monotone endpoint counterexample

The registered family is

```text
m_a(u) = u + a u (1-u^2),    a in {-0.4, +0.4}.
```

Each map is odd, fixes `-1`, `0`, and `1`, stays in `[-1,1]`, and is strictly
increasing. The measurements on 65,537 evenly spaced points are:

| `a` | analytic derivative floor | measured slope floor | max displacement | bin disagreement | structural violations |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `-0.4` | `.600000` | `.600000` | `.153960` | `.710438` | `0` |
| `+0.4` | `.200000` | `.200037` | `.153960` | `.722371` | `0` |

The disagreement is large because the exact physical interface is a metric
chart, not merely an ordered interval. A monotone warp preserves topology and
orientation while moving many examples across the fixed decoder's metric bin
boundaries.

Consequently, a monotone head with fixed endpoints would prevent sign reversal
and unbounded affine scaling, but it would not solve the support-relative
nonlinear calibration defect exposed by the preceding affine audit.

## Typed split positive construction

The positive construction returns two named fields:

```text
physical_scalar   = analytic_calibrated_cosine(observation)
auxiliary_carrier = learned_equivariant_vector(observation).
```

Two independently randomized auxiliary encoders were evaluated on both fresh
cohorts. The auxiliary carrier contains 10,097 parameters and changes by RMS
`.772190` between states. At the same time:

| Contract | Result |
| --- | ---: |
| physical scalar parameter count | `0` |
| auxiliary-to-scalar gradient leaks | `0` |
| maximum scalar change across auxiliary states | `0.0` |
| maximum deviation from analytic scalar | `0.0` |

This is non-regaugeability by type separation. Learned parameters may carry
additional information, but they cannot rename or deform the public physical
coordinate.

## Scientific accounting

### What this settles

- A fixed orientation and endpoint convention is weaker than a fixed physical
  chart.
- Monotonicity does not prevent the exact-bin failures observed under
  extrapolation.
- Requiring every allowed parameter state to preserve exact cosine forces the
  learned degrees of freedom out of the scalar path on the current task.
- The analytic canonicalizer is not merely a convenient baseline; it is the
  non-regaugeable scalar spine for this observation model.

### What this does not settle

- It does not show that a learned auxiliary carrier improves any task.
- It does not rule out a learned estimator under noisy or incomplete
  observations; exact physical correctness then cannot be guaranteed from the
  function class alone.
- It does not test a richer group, a new task, or a typed multi-channel
  continuation.
- It does not promote a systems preflight into trained-model evidence.

## Program decision

Do not launch another learned-scalar d6/d10 campaign on the present noiseless
future-cosine task. An exact typed scalar would simply reproduce the analytic
solution, while any nontrivial scalar correction reintroduces metric chart
freedom.

There are now two honest paths:

1. For engineering, adopt the frozen analytic scalar as the public interface
   and confine learning to explicitly typed auxiliary channels.
2. For new science, change scope before training. The highest-leverage option
   is a richer identifiable calibrated group or a different identifiable task
   whose sufficient invariant is not already the analytic answer. Begin with
   an observation-level identifiability proof and analytic group projection.

The declared Gaussian acquisition-count and signed-bias laws are already
closed. A noisy-sensor successor is justified only if it states a genuinely new
deployment law rather than retraining under a solved noise model.

## Reproduction

```bash
MPLCONFIGDIR=/tmp/mplconfig pixi run python -m \
  experiments.structure_net.tinyllm_fixed_chart_function_class_preflight \
  --output /tmp/tinyllm-fixed-chart-function-class-preflight.json

MPLCONFIGDIR=/tmp/mplconfig pixi run pytest -q \
  tests/structure_net/test_tinyllm_fixed_chart_function_class_preflight.py
```

| Artifact | SHA-256 |
| --- | --- |
| frozen design | `d300919a678aebb8bc70cb65913d49696815b1fe883f9e7fbf2a63c8e9fb1470` |
| preflight implementation | `a0a498d4f12fdad3020c2c0711725cf790ae6e320c05909ba73da6ca1f06b829` |
| calibrated source | `73e152b9b6c7ab51ad08e6a80786c32a71566f9d4fb410ff2cf3df53761faf77` |
| interval-interface source | `b027eb55d971b87f598bd600131b33ead82463e3e73766fcdce38fdc3a5497c6` |

The output is intentionally a disposable `/tmp` design artifact. The frozen
design, implementation, tests, and this measured report are the reproducible
record; no trained evidence or DVC data object was created.
