# TinyLLM fixed-gauge error decomposition preregistration

**Status:** PREREGISTERED POST-OUTCOME DIAGNOSTIC — ORACLE OUTCOMES NOT INSPECTED  
**Date:** 2026-08-06  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, `UNDERPOWERED`, post-outcome diagnostic  
**Hypothesis:** `tinyllm-c2-fixed-gauge-error-decomposition-v1`  
**Schema:** `nal.tinyllm-fixed-gauge-error-decomposition.v1`

## Known outcome and question

The observation-derived fixed-gauge campaign failed all three checkpoint gates.
Its sensor contract missed narrowly: held-out mean phase error was
`0.106--0.141` output bins, and seed 53's writer missed only one continuous
cell by `0.0009` mean bins. Seeds 7 and 29 failed more broadly despite held-out
coordinate `R2` above `0.97`.

The oracle-carrier interventions below have not been inspected. Is each failure
caused by quantized observation-side phase recovery, or is the fixed
three-channel carrier plus checkpoint-local linear writer itself insufficient?

## Fixed interventions

Reuse the exact checkpoints, bases, fit/held-out cohorts, thresholds, target
controls, and primary observed-carrier writer from
`tinyllm-c2-fixed-semantic-gauge-writer-v1`.

For each orbit define the oracle carrier only for diagnosis:

```text
z_oracle = (cos(2 phi), sin(2 phi), 1),
```

using latent phase. It is not deployable and cannot become the learned result.
Fit a second no-intercept ridge writer from `z_oracle` to target rank-three
coordinates on the same two alignment-fit regimes. Evaluate four states on
each held-out cell:

| Fit carrier | Evaluation carrier | Purpose |
| --- | --- | --- |
| observed | observed | exact replay of the failed primary campaign |
| observed | oracle | substitute only sensor accuracy |
| oracle | observed | measure train/evaluation carrier mismatch |
| oracle | oracle | test ideal fixed-gauge writer capacity |

No model, encoder, decoder, observer, or nonlinear writer is trained.

## Contracts and classification

1. Primary observed-state continuous and coordinate metrics must replay within
   `1e-6`; otherwise the diagnostic is invalid.
2. The oracle carrier must match latent degree-two phase to mean and p95 error
   at most `1e-8` bins.
3. Exact/direct/zero continuous controls must retain their primary outcomes.
4. Each oracle-based writer is judged by the unchanged continuous endpoint:
   alignment loss at most `0.005`, mean shift at most `0.125` bins, p95 at most
   `0.50`, degree within `0.10` of two, and resolved sampling in all four cells.

Checkpoint classification is fixed:

| Result | Classification |
| --- | --- |
| observed-fit/oracle-eval and oracle-fit/oracle-eval both pass | `sensor_limited` |
| only oracle-fit/oracle-eval passes | `sensor_and_fit_mismatch` |
| oracle-fit/oracle-eval fails | `writer_or_carrier_limited` |
| replay/oracle contracts fail | `invalid` |

The campaign makes no confirmation claim; it is a causal error decomposition.

## Decision

- If all checkpoints are sensor-limited, train the typed sensor encoder with
  the three-channel gauge and linear writer fixed.
- If some checkpoints are writer/carrier-limited, the learned sidecar must add
  target-local invariant context or nonlinear neutral synthesis; merely
  improving the sensor is insufficient.

## Planned artifacts

- runner:
  `experiments/structure_net/tinyllm_fixed_gauge_error_decomposition.py`
- tests:
  `tests/structure_net/test_tinyllm_fixed_gauge_error_decomposition.py`
- root:
  `data/experiments/tinyllm_fixed_gauge_error_decomposition/20260806_d6_preregistered_diagnostic`
- report:
  `docs/08 - Analysis/2026-08-06_tinyllm-fixed-gauge-error-decomposition.md`
- meta hypothesis:
  `tinyllm-c2-fixed-gauge-error-decomposition-v1`

## Method boundaries

Latent phase makes the oracle carrier non-deployable. The diagnostic reuses
post-outcome cells and target-local fit access. It distinguishes sensor error
from the declared linear writer/carrier but does not test a learned encoder,
context-conditioned writer, or population prevalence.
