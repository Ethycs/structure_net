# TinyLLM frozen quotient-writer capacity

**Status:** NEGATIVE DECOMPOSITION — SMALL WRITER INSUFFICIENT 3/3  
**Date:** 2026-08-07  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, `UNDERPOWERED`, post-outcome diagnostic  
**Hypothesis:** `tinyllm-c2-frozen-writer-capacity-v1`  
**Preregistration:** [frozen writer-capacity preregistration](../07%20-%20Status%20Reports/2026-08-07_tinyllm-frozen-writer-capacity-preregistration.md)

## Verdict

A small invariant summary of the recipient state does not rescue the portable
quotient writer. All predecessor-replay, context-numerical, and frozen-target
control contracts passed in all three checkpoints. Nevertheless, none of the
eight phase-only Fourier writers through order 18 or four phase-by-context
writers through order 4 passed all four held-out causal cells. All three
checkpoints are `small_writer_insufficient` under the locked classification.

This closes the preregistered small-writer branch:

> Exact quotient phase plus a source-fitted three-dimensional chart of the
> propagated Reynolds barycenter is not a sufficient shift-stable interface
> for the example-specific rank-three defect.

The context features contain real predictive signal, particularly in seed 29,
but they do not define a complete causal chart. The program should now stop
fitting larger portable writers and inspect the frozen continuation's local
task geometry directly.

## Locked gates

| Gate | Result | Required | Verdict |
| --- | ---: | ---: | --- |
| predecessor order-1 replay | **3/3** | 3/3 | pass |
| propagated-context numerical contract | **3/3** | 3/3 | pass |
| zero/exact/direct-rank-three target controls | **3/3** | 3/3 | pass |
| any low-order phase writer passes | **0/3** | classifier | fail |
| any matched high-order phase writer passes | **0/3** | classifier | fail |
| any phase-by-context writer passes | **0/3** | classifier | fail |
| context improves mean shift by at least 0.125 bins without passing | **0/3** | classifier | fail |

Every writer's regime-preserving shuffled-correspondence control failed and
met the `0.125`-bin specificity margin. No checkpoint was invalid, and no
writer was selected post hoc.

## Intervention

For quotient angle `theta = 2 phi`, the phase-only ladder used

```text
Phi_m(theta) = [1, cos(theta), sin(theta), ..., cos(m theta), sin(m theta)]
```

at orders `1, 2, 3, 4, 6, 10, 14, 18`. The conditional branch flattened the
recipient propagated barycenter `F(mean h)`, fit a rank-three PCA chart on the
two alignment-fit regimes only, and evaluated

```text
Psi_m(theta, c) = Phi_m(theta) tensor [1, c1, c2, c3]
```

for `m = 1, 2, 3, 4`. The corresponding feature widths were `12, 20, 28, 36`;
their matched phase-only controls had widths `13, 21, 29, 37`. All maps used
no-intercept ridge `1e-6`, saw no held-out coordinates, and were patched into
the unchanged frozen continuation.

## Results

| seed | order-1 mean shift | best phase-only arm | mean / passing cells | best context arm | mean / passing cells | classification |
| ---: | ---: | --- | ---: | --- | ---: | --- |
| 7 | 0.2410 | `m=3` | 0.1415 / 1 of 4 | `m=3` | 0.1433 / 1 of 4 | small writer insufficient |
| 29 | 0.2317 | `m=4` | 0.2208 / 0 of 4 | `m=3` | 0.1566 / 1 of 4 | small writer insufficient |
| 53 | 0.1611 | `m=2` | 0.1552 / 1 of 4 | `m=2` | 0.1376 / 1 of 4 | small writer insufficient |

The context arm improves over the order-1 baseline by `0.0977`, `0.0751`, and
`0.0235` bins for seeds 7, 29, and 53. None reaches the preregistered `0.125`
descriptive threshold, and each best context writer passes only one cell.
Increasing phase-only capacity beyond the low-order optimum does not help:
the best order-18 arm passes no cells in seeds 7 and 29 and remains incomplete
in seed 53.

This is a descriptive-versus-causal separation, not an absence of activation
signal. The best context writer's worst held-out defect-coordinate `R2` is
`0.9917`, `0.9868`, and `0.9936` for seeds 7, 29, and 53. Despite explaining
nearly all coordinate variance, each passes only heldout-B composition. The
continuation is therefore sensitive to a small residual direction that
ordinary activation variance treats as negligible.

The invariant context itself is well conditioned:

| seed | flattened context width | rank-three energy | minimum source scale | orthogonality error | maximum replay error |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 1,152 | 0.9499 | 0.2385 | `4.44e-14` | `2.16e-7` |
| 29 | 1,152 | 0.9385 | 0.2512 | `8.59e-14` | `1.29e-7` |
| 53 | 1,152 | 0.9608 | 0.3375 | `9.11e-14` | `3.56e-7` |

The failure is therefore not caused by a degenerate PCA chart or predecessor
drift. The top three barycenter directions capture most source variance, yet a
small tensor-product interaction with phase remains causally incomplete.

## Mechanistic update

The two capacity experiments now rule out, on these three frozen fronts:

1. a linear absolute write from exact quotient phase;
2. low-order phase curvature;
3. high-order phase capacity through order 40;
4. conditioning on the eight observed calibration fields; and
5. conditioning on the top three invariant propagated-barycenter directions.

Actual example-specific rank-three coordinates still pass every target cell.
The remaining mismatch can therefore lie in the task metric of small
coordinate errors, nonlinear continuation around the predicted state, or
recipient context outside the tested low-rank chart. Another flexible writer
would confound those possibilities.

## Shortest next causal test

Run a no-fit local continuation audit at the best order-4 predicted state and
the exact rank-three state in every existing held-out cell:

1. Compute the frozen continuation Jacobian of the circular output moment with
   respect to the three carrier coordinates.
2. Decompose the exact-minus-predicted coordinate residual into local
   task-tangent and task-kernel components.
3. Patch the tangent component alone, the kernel component alone, their
   norm-matched random controls, and the full residual.
4. Retain the same continuous endpoint under composition and extrapolation.

If the tangent correction alone closes the output gap with a stable direction,
the writer used the wrong local task metric and that metric can be fixed
architecturally. If the linearized tangent correction fails while the full
residual passes, the continuation is locally curved or state-conditioned. If
the nominal kernel has causal effect, the first-order chart is insufficient
and higher derivatives—not a larger observational writer—are the next object.

## Campaign integrity

| Item | Value |
| --- | --- |
| requested / completed / failed | 3 / 3 / 0 |
| trained TinyLLMs / fitted predictive observers | 0 / 0 |
| fitted writers, including matched shuffles | 72 |
| checkpoints | d6 seeds 7, 29, 53 |
| held-out cells | two cohorts under composition and extrapolation |
| examples per cell | 64 exact `C2` orbits |
| device | NVIDIA GeForce RTX 2060 SUPER, CUDA |
| peak allocated CUDA memory | 0.262 GiB |
| analysis time | 14.85 seconds |
| implementation SHA-256 | `d53edaedd49ae553af9f8393d92254664239e5100246ac0fd3a06cb420ca80ed` |
| campaign SHA-256 | `7ab6079d56bc8dc78cc5e1f4011c581b8f507bcd8ec20573bbf8b11227df8d1b` |
| final DVC data root | `f29e1f0e920aff74661e2a64d7ec56c1.dir` (`1,796` files, `39,812,097,258` bytes) |
| lakeFS snapshot | `71cda38c5b84bfa364c136a0741dd4ff6e77040395f4e24b5d50d8419c11a648` |

The separate eight-orbit CUDA lifecycle exercised loading, schema, and resume
but cannot replay the 64-orbit scientific metrics; it is explicitly
`systems_lifecycle_only_not_quality_evidence`. Its invalid scientific gates do
not enter this report. Re-running the primary command verified immutable
resume without rewriting any result bytes.

The final DVC root was pushed to the configured
`lakefs://artifacts/main/structure-net/` remote and is contained in the cited
clean lakeFS commit.

The preregistration's Amendment A was written from a stale directory snapshot
while the primary job was still completing and incorrectly described the root
as interrupted. Amendment B preserves and corrects that record: the original
root above is the primary campaign. A post-outcome rerun on an NVIDIA RTX 3060
is retained at the `..._v2` root as a deterministic hardware replication. It
uses the same implementation digest and per-seed scientific fingerprints,
reproduces all classifications and gates, and differs by at most `1.11e-5`
across aggregate writer-summary numerics. Its campaign SHA-256 is
`7387098da1b5852ca9f49207904db42368fd33ed5fca442ca017684e5a153b04`;
it is not pooled with the primary result.

## Artifacts and reproduction

- Aggregate:
  `data/experiments/tinyllm_frozen_writer_capacity/20260807_d6_preregistered_diagnostic/campaign_results.json`
- Per-checkpoint records:
  `data/experiments/tinyllm_frozen_writer_capacity/20260807_d6_preregistered_diagnostic/runs/seed_*/result.json`
- Post-outcome cross-GPU replication:
  `data/experiments/tinyllm_frozen_writer_capacity/20260807_d6_preregistered_diagnostic_v2/`
- Systems-only lifecycle:
  `data/experiments/tinyllm_frozen_writer_capacity/20260807_shakedown_cuda/`
- Runner:
  `experiments/structure_net/tinyllm_frozen_writer_capacity.py`
- Tests:
  `tests/structure_net/test_tinyllm_frozen_writer_capacity.py`
- Meta-hypothesis:
  `data/meta_hypotheses/tinyllm-c2-frozen-writer-capacity-v1.json`

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
MPLCONFIGDIR=/tmp/matplotlib-structure-net \
pixi run python -m \
  experiments.structure_net.tinyllm_frozen_writer_capacity \
  --device cuda:0 \
  --output \
  data/experiments/tinyllm_frozen_writer_capacity/20260807_d6_preregistered_diagnostic
```

## Method boundaries

The exact quotient angle is latent and diagnostic only. PCA and writers use
the locked alignment-fit cells; the propagated-barycenter context is invariant
to sheet permutation but checkpoint-local. The held-out cells have appeared in
earlier post-outcome diagnostics. Off-manifold patch sufficiency does not show
natural use, and three selected checkpoints do not establish population
prevalence.
