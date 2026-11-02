# TinyLLM fixed-gauge writer-capacity preregistration

**Status:** PREREGISTERED POST-OUTCOME DIAGNOSTIC — CAPACITY OUTCOMES NOT INSPECTED  
**Date:** 2026-08-06  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, `UNDERPOWERED`, post-outcome diagnostic  
**Hypothesis:** `tinyllm-c2-fixed-gauge-writer-capacity-v1`  
**Schema:** `nal.tinyllm-fixed-gauge-writer-capacity.v1`

## Known outcome and question

The observation-derived fixed-gauge writer failed `3/3` checkpoint gates. The
locked oracle decomposition then reproduced the primary metrics exactly,
satisfied its latent-phase contract below `1e-15` output bins, and still
classified all three checkpoints `writer_or_carrier_limited`. Oracle
coordinate fits remained high (`R2 >= 0.9687`), but oracle-fit/oracle-evaluated
writers passed only `1/12` held-out cells.

Because the oracle carrier `(cos(theta), sin(theta), 1)`, with `theta = 2 phi`,
parameterizes the complete declared quotient and direct rank-three target
patches pass, this experiment asks the narrower question:

```text
Is the failed quotient-to-residual write
    a low-order curved function of theta,
    a higher-capacity phase function,
    or a phase-and-calibration-context conditional function?
```

No model is retrained.

## Locked cohort and controls

Reuse the exact checkpoints, rank-three bases, alignment-fit cells, heldout-A
and heldout-B composition/extrapolation cells, readout rotations, and
continuous thresholds from
`tinyllm-c2-fixed-gauge-error-decomposition-v1`.

The following contracts are mandatory:

1. The order-one oracle writer must replay the stored oracle/oracle coordinate
   and continuous metrics within `1e-6`.
2. Oracle carrier mean and p95 phase error remain at most `1e-8` output bins.
3. Zero fails while exact and direct-rank-three controls pass in every cell.
4. Writers see only the two alignment-fit regimes. No held-out fitting,
   threshold selection, or capacity selection is allowed.

## Frozen feature ladder

For quotient angle `theta`, define the fixed order-`M` Fourier vector

```text
q_M(theta) = (cos(theta), sin(theta), ..., cos(M theta), sin(M theta), 1).
```

Fit no-intercept ridge writers from each declared feature matrix into the
checkpoint-local rank-three defect coordinates, with ridge `1e-6`:

| Arm | Feature width | Purpose |
| --- | ---: | --- |
| `quotient_order1` | 3 | exact replay of the failed oracle linear writer |
| `quotient_order2` | 5 | smallest nonlinear neutral curvature |
| `quotient_order4` | 9 | preregistered low-order curvature ceiling |
| `quotient_order40` | 81 | parameter-count control for context arm |
| `quotient_order4_context` | 81 | low-order phase conditioned on observed calibration |

The calibration vector is the orbit-mean eight-field observed pilot packet:
orientation cosine/sine, signed speed, amplitude, planar offset, and planar
drift. Standardize its eight columns using alignment-fit means and standard
deviations only. The conditional feature is the tensor product

```text
q_4(theta) tensor (1, standardized_calibration).
```

This is deliberately a context-conditioned neutral write, not a claim that
the pilot fields are themselves invariant under every acquisition action.
The output is still fitted only to the neutral rank-three defect.

For `order2`, `order4`, `order40`, and the context arm, fit a
regime-preserving target-coordinate-shuffled writer with the same feature
width and ridge.

## Endpoint and specificity

Each candidate must pass the unchanged continuous endpoint in all four
held-out cells:

- alignment loss from exact at most `0.005`;
- mean circular-moment shift at most `0.125` bins;
- p95 shift at most `0.50` bins;
- degree within `0.10` of two;
- resolved sampling.

Its shuffled writer must fail at least one cell, and its aggregate mean shift
must beat shuffled by at least `0.125` bins. Coordinate `R2` is reported but
cannot substitute for the causal endpoint.

## Fixed checkpoint classification

Apply the first matching row:

| Outcome | Classification |
| --- | --- |
| a specific order-2 or order-4 quotient writer passes | `low_order_curvature_limited` |
| neither low-order arm passes, but specific order-40 passes | `high_order_curvature_limited` |
| no quotient-only arm passes, but the specific context arm passes | `calibration_context_limited` |
| no candidate passes | `unresolved_writer_limited` |
| replay, oracle, or target controls fail | `invalid` |

The campaign is a mechanistic decomposition, not a confirmation claim.

## Decision

- `low_order_curvature_limited`: build a small exact neutral nonlinear fusion;
  do not add nuisance context.
- `high_order_curvature_limited`: treat the write as a curved chart and test
  whether the required harmonic order is stable on a fresh cohort before
  architecture work.
- `calibration_context_limited`: build a typed conditional writer whose
  semantic carrier stays neutral while its residual write adapts to observed
  acquisition context.
- `unresolved_writer_limited`: stop fitting portable sidecars and intervene on
  the downstream continuation's nonlinear state dependence.

## Planned artifacts

- runner:
  `experiments/structure_net/tinyllm_fixed_gauge_writer_capacity.py`
- tests:
  `tests/structure_net/test_tinyllm_fixed_gauge_writer_capacity.py`
- root:
  `data/experiments/tinyllm_fixed_gauge_writer_capacity/20260806_d6_preregistered_diagnostic`
- report:
  `docs/08 - Analysis/2026-08-06_tinyllm-fixed-gauge-writer-capacity.md`
- meta hypothesis:
  `tinyllm-c2-fixed-gauge-writer-capacity-v1`

## Method boundaries

Latent phase makes every quotient feature diagnostic rather than deployable.
The context arm uses an observed calibration packet but still has
checkpoint-local target access on alignment-fit orbits. The order-40 control
is intentionally flexible and underdetermined as a general architecture.
Only three selected frozen checkpoints and reused held-out cells are tested;
off-manifold patch success would show sufficiency, not natural use.

