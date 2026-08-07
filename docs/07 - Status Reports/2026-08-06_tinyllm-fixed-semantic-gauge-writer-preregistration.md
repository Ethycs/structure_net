# TinyLLM fixed semantic-gauge writer preregistration

**Status:** PREREGISTERED — WRITER OUTCOMES NOT INSPECTED  
**Date:** 2026-08-06  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, `UNDERPOWERED`, post-outcome positive control  
**Hypothesis:** `tinyllm-c2-fixed-semantic-gauge-writer-v1`  
**Schema:** `nal.tinyllm-fixed-semantic-gauge-writer.v1`

## Evidence boundary and question

The cross-seed Euclidean and task-metric transport outcomes are known, and this
study reuses their alignment-fit and held-out orbit cohorts. The fixed semantic
writer, its shuffled control, and their causal outcomes have not been
inspected. Its evidence role is
`preregistered_post_outcome_underpowered_mechanistic_positive_control`; it can
validate or falsify the proposed sidecar interface but cannot independently
confirm earlier transport results.

Before training a learned sidecar, can one observation-derived, exactly
`C2`-invariant semantic gauge drive all three frozen checkpoint continuations
through checkpoint-local linear writers?

## Fixed universal carrier

The calibrated observation packet already identifies the future phase vector
`v = (cos(phi), sin(phi))` without using latent phase or the target label. Apply
the exact neutral fusion of the nontrivial `C2` character:

```text
z(v) = (v_x^2 - v_y^2, 2 v_x v_y, v_x^2 + v_y^2).
```

Under the deck action `v -> -v`, `z(-v) = z(v)` exactly. The first two channels
are `(cos(2 phi), sin(2 phi))`; the third is the normalized charged-carrier
energy. This fixes channel order, sign, scale, and Euclidean metric before any
checkpoint-specific fit.

The carrier is computed from the observed sensor and calibration packet by the
existing analytic phase carrier. Latent phase is used only after inference to
audit the observation contract; it never enters writer fitting.

## Writers and controls

For each target checkpoint seed 7, 29, and 53, refit the same source-selected
rank-three block-0 post-attention defect basis used by the preceding campaigns.
On the two alignment-fit regimes only, fit a no-intercept ridge writer

```text
W_t = argmin_W ||z W - c_t||^2 + 1e-6 ||W||^2,
```

where `c_t` is the target checkpoint's rank-three defect coordinate. The
constant energy channel supplies the affine offset, so a separate intercept is
not allowed.

A deterministic regime-preserving shuffled control permutes target coordinates
within each fit regime before fitting. No transformer, frontend, carrier,
decoder, probe, calibration, or nonlinear writer is trained.

At held-out evaluation, patch

```text
propagated_t(x) + z(x) W_t B_t
```

into the frozen target continuation. Exact, zero-defect, and direct target
rank-three states are evaluated in the same continuation batch.

## Data and fixed checkpoints

- frozen d6 degree-two checkpoints, seeds 7, 29, and 53;
- block-0 post-attention synthesis front and rank-three target basis;
- 64 exact `C2` orbits per cell;
- alignment fit: seeds 130007/130008, writer fitting only;
- held-out A: seeds 230007/230008;
- held-out B: seeds 330007/330008;
- composition and outside-range extrapolation in every cohort;
- predecessor campaign and every checkpoint/result hash must verify.

## Primary checkpoint-level gates

Each checkpoint must pass all of the following over its four held-out cells:

1. **observation contract:** the observation-derived fixed carrier has circular
   alignment at least `0.99` with latent `(cos(2 phi), sin(2 phi))`, mean
   circular shift at most `0.125` output bins, and p95 shift at most `0.50`
   bins in every cell;
2. **target controls:** zero defect fails while exact and direct rank three pass
   the continuous endpoint in every cell, with decomposition error at most
   `1e-6`;
3. **fixed-gauge causal writer:** alignment loss from exact at most `0.005`,
   mean circular-moment shift at most `0.125` bins, p95 shift at most `0.50`
   bins, winding within `0.10` of degree two, and resolved sampling in every
   cell;
4. **shuffled specificity:** the shuffled writer fails the continuous endpoint
   in at least one cell and the fixed-gauge writer's four-cell mean shift is at
   least `0.125` bins lower than the shuffled writer's.

The campaign supports the fixed portable gauge only if all four gates pass in
all three checkpoints. Hard-bin accuracy with each checkpoint's frozen scalar
calibration and coordinate variance explained are secondary; neither can
rescue a failed continuous gate.

## Interpretation

| Outcome | Meaning | Next action |
| --- | --- | --- |
| all checkpoints pass | the fixed three-channel gauge is a causally sufficient portable interface; learned training only needs to recover it | train typed versus parameter-matched unconstrained encoders with the writer/gauge frozen |
| sensor contract passes but one writer fails | the universal carrier is identifiable, but a checkpoint needs context-dependent or nonlinear synthesis | add target-local invariant context to the writer before training a full model |
| all writers fail | the proposed three-channel semantic gauge is not sufficient for the frozen causal fronts | increase the declared carrier or abandon residual-sidecar synthesis |
| shuffled passes | phase distribution or writer bias, not examplewise semantic correspondence, explains apparent success | reject the control |

## Integrity and artifacts

The runner must record the implementation digest, scientific fingerprints,
predecessor/checkpoint hashes, writer matrices, observation audit, all per-cell
continuous metrics, strict JSON, zero trained models, and exactly six fitted
linear maps. Compatible completed records are byte-immutable. Any CUDA
shakedown is systems-only evidence and cannot be pooled.

- runner:
  `experiments/structure_net/tinyllm_fixed_semantic_gauge_writer.py`
- tests:
  `tests/structure_net/test_tinyllm_fixed_semantic_gauge_writer.py`
- primary root:
  `data/experiments/tinyllm_fixed_semantic_gauge_writer/20260806_d6_preregistered_diagnostic`
- report:
  `docs/08 - Analysis/2026-08-06_tinyllm-fixed-semantic-gauge-writer.md`
- meta hypothesis:
  `tinyllm-c2-fixed-semantic-gauge-writer-v1`

## Method boundaries

This is an analytic positive control, not a learned interpretability result.
It relies on the observed calibration reference and exact declared sensor
decoder. The target-local writer has paired target-checkpoint access on fit
orbits and is not a standalone encoder. Patches are off-manifold, held-out
cells were used by earlier diagnostics, and only three selected stable
checkpoints are tested.
