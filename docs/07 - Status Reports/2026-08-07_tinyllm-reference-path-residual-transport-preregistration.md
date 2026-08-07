# TinyLLM reference-path residual-transport audit preregistration

**Status:** LOCKED WITH PRE-PRIMARY EXPOSURE AND POST-PRIMARY PRODUCER CORRECTION  
**Date:** 2026-08-07  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED` WITH PRE-PRIMARY SHAKEDOWN OUTCOME EXPOSURE, outcome-directed frozen-checkpoint mechanistic diagnostic  
**Hypothesis:** `tinyllm-reference-path-residual-transport-v1`  
**Schema:** `nal.tinyllm-reference-path-residual-transport.v1.4`

**Post-primary terminal-coordinate producer correction (2026-08-07):** The
schema `v1.3` affected-checkpoint validation again retained the same replay
error, falsifying the norm-producer diagnosis below. The locked source passes
`target_cosine` directly to its one-step update. The audit instead constructed
the final requested coordinate as
`m_0 + 1 * (target_cosine - m_0)`, which is algebraically equal but need not
round to the bit-identical floating value. Schema `v1.4` assigns the exact
stored `target_cosine` at the terminal step and retains the declared linear
formula only at interior points. The residual-update rule, interior schedule,
step grid, endpoints, gates, and thresholds do not change. The failed schema
`v1.3` validation root remains preserved; the affected checkpoint must pass
before any corrected full campaign.

The initially declared corrective root was partially populated by a
superseded schema `v1.2` process while the producer audit was still in
progress. That root is preserved and must not be resumed. The authoritative
schema `v1.4` campaign therefore uses the fresh `..._v4` root declared below.
Artifact routing is excluded from the scientific protocol digest.

**Post-primary norm-producer correction (2026-08-07, superseded):** The schema `v1.2`
single-checkpoint validation retained the same analytic seed 17 replay error,
so endpoint normalization was not the cause of that scalar mismatch. The
endpoint correction remains because it enforces the declared exact `q_1` and
`q_64` boundary values, but the causal diagnosis below is superseded.

The remaining difference was the norm producer itself: the locked source used
`delta.norm(dim=1)`, while the transport audit accumulated step length with
`torch.linalg.vector_norm(delta, dim=1)`. The residual update tensors and every
task output were identical, but the two reduction kernels differed by
`3.814697e-6` on the affected mean. Schema `v1.3` uses the source's exact norm
operator for the reported rollout step length. This changes no residual
intervention, posterior, gate, schedule, threshold, or interpretation. The
failed schema `v1.2` validation root remains preserved, and the affected
checkpoint must pass before a corrected full campaign is launched. The schema
`v1.3` validation showed that this did not remove the replay mismatch; the
terminal-coordinate correction above is the operative hypothesis.

**Post-primary endpoint-replay correction (2026-08-07):** The schema `v1.1`
primary completed all ten cells but was classified `invalid` solely because
analytic seed 17's extrapolation one-step patch norm replay differed from the
source by `3.814697e-6`, above the locked `2e-6` ceiling. Its accuracy and
cross-entropy replayed, all other source replays passed, all ten positive
controls passed, all shuffled controls failed, and every system state remained
unchanged. The invalid `20260807_d8_preregistered` root is preserved.

Audit initially suspected that `shortest_reference_path` normalizing the already normalized
stored `q_1` and `q_64` values a second time. Although the complex endpoint
change was only about `1e-16`, it violated the intended exact source endpoint
replay and changed one float reduction by one ULP-scale increment. Schema
`v1.2` and later schemas construct the same normalized interior curve but overwrite `t=0` and
`t=1` with the exact stored circular-mean endpoints. No target schedule,
gradient update, step grid, task threshold, seed rule, or interpretation gate
changes. A corrected campaign must use a fresh root and remains
corrective/outcome-exposed evidence; it is not a new preregistered confirmation.
The schema `v1.2` validation subsequently showed that this endpoint issue did
not cause the patch-norm replay failure; the norm-producer correction above is
the operative replay fix.

**Pre-primary systems-validity amendment (2026-08-07):** The first one-seed
CUDA shakedown completed before the primary campaign. It exposed the `K=1`
and `K=16` task aggregates for seed 7 in both arms: the true-cosine rollout
passed analytic only at `K=16` and learned at both counts; the path-moment and
shuffled schedules passed neither arm. These outcomes are systems-only and
will not be pooled as separate replicates, but their deterministic primary
replay is now known. The final campaign must therefore be described as
corrective/outcome-exposed evidence rather than fresh confirmation.

The shakedown also showed minimum finite task-gradient norms of approximately
`4e-9--6e-9`, below the originally declared `1e-8` validity threshold. That
threshold was structurally misplaced: a small but finite gradient is a
scientific property of the scalar chart and a possible reason for transport
failure, not evidence that the runner or source replay is invalid. Schema
`v1.1` therefore retains the denominator floor and reports every minimum norm
but removes gradient magnitude from systems validity. No path, rollout,
endpoint, task threshold, population rule, or control gate changes. The
original `20260807_shakedown_cuda` root remains preserved; a new implementation
must use a new root.

**Pre-outcome generator-contract correction (2026-08-07):** No shakedown or
primary transport outcome had been generated. The two sheets of one exact
cosine fiber generally have different absolute calibration orientations. They
share the same acquisition error, so the invariant contract is equality of
their angular path increments relative to their own `q_1`, not equality of
their absolute complex paths. The runner checks the maximum paired increment
difference against `1e-12`. All paths, endpoints, rollout schedules, task
gates, controls, and classifications below are unchanged.

## Provenance boundary

This diagnostic is selected after the repeated-reference acquisition outcome
was known. The source campaign established that coherent input-side averaging
at `m=64` repairs all ten frozen systems, while its one-step true-cosine
residual write passes only `0/5` analytic and `2/5` learned checkpoints. Those
facts are prior evidence and are not outcomes of this study.

No reference-path residual curve, locally relinearized rollout, rollout-step
titration, or path-specificity control from this design has been inspected at
the time this document is locked. The new evidence can diagnose the failed
causal ceiling; it cannot retroactively make the source campaign valid.

## Question and prediction

Does the earlier residual-write ceiling fail primarily because one large
linearized step exceeds the local validity radius of the frozen task chart?

The directional prediction is that repeatedly re-evaluating the same local
task gradient over a fine schedule will recover the frozen task at `K=16` in
at least four of five checkpoints per structured arm, while the inherited
one-step write remains below that population gate. A second rollout follows
the actual task-moment schedule along the observed reference path. Requiring
both schedules to pass distinguishes finite-radius repair from merely changing
the terminal scalar target.

## Locked source evidence

The sole scientific source is:

```text
campaign
data/experiments/tinyllm_reference_acquisition_replicates/
    20260807_d8_preregistered/campaign_results.json

campaign SHA-256
269fd948f0d6fee8916bbe3cb94c1d87f76572e43c103b52fc8775fa9653031e

implementation SHA-256
6c3cc4463b2c515280c778461e4606dca6ffb19f08d11cb5b7a852a942c7df77

result-manifest SHA-256
8181d48f99850d5e487b5209d648373410bea3b46b2eafcb58f57118ed898c1c

repeat-array SHA-256
8886c1f6ad0fd307720748e44b1741edb656fdb9e29e913c7ad059b72397eef4
```

The runner must validate those hashes, all ten source result hashes, the
calibrated-system checkpoint and state hashes, and every dataset hash before
evaluating a new transport endpoint.

## Replication units and fixed systems

Checkpoint seed is the replication unit. Retain the two source arms:

- `analytic_calibrated`;
- `learned_calibrated_equivariant`.

Use seeds `7`, `17`, `29`, `41`, and `53`. Every front end, TinyLLM, scalar
embedding, layer norm, and answer row remains frozen. No observer, probe,
readout, adapter, denoiser, or model parameter is fit.

Evaluate only the source composition and extrapolation datasets, each with
`1024` examples. Their generator configuration, tokens, targets, calibration
packets, and repeat errors must replay the source hashes exactly.

## Reference path

For each example, let `q_1` and `q_64` be the analytic circular-mean orientation
estimates from the stored nested repeat array. Define the shortest circular
path

```text
delta = angle(q_64 * conj(q_1))
q(t) = q_1 * exp(i * t * delta),  t in [0,1].
```

The two sheets of one exact task fiber must receive bit-identical angular path
increments relative to their own `q_1`. Only the two observed orientation
fields change. At every path point, run the
unchanged front end and TinyLLM and retain the final query residual `r(t)`,
answer posterior `p(t)`, and ordered answer moment

```text
m(r) = sum_j center_j * softmax(logits(r))_j.
```

Use the common fine grid `t=s/16`, `s=0,...,16`. Coarser step counts
`K in {1,2,4,8}` are exact nested subsets of that grid.

## Residual interventions

For a residual `z` and requested scalar coordinate `y`, define the local
minimum-norm task-gradient update

```text
N(z; y) = z + (y - m(z)) / max(||grad m(z)||^2, epsilon) * grad m(z),
epsilon = 1e-8.
```

The following arms all begin at the actual `m=1` residual `r(0)`:

| Arm | Target schedule | Role |
| --- | --- | --- |
| actual reference path | run the frozen system with `q(t)` | on-manifold positive mechanism |
| true-cosine rollout | linear schedule from `m(r(0))` to latent `cos(phi)` | exact multi-step version of the failed causal ceiling |
| path-moment rollout | `m(r(s/K))` at every nested path point | isolates local-radius failure from terminal-coordinate mismatch |
| shuffled path-moment rollout | fiber-block permutation of another path's moment schedule | specificity control |
| exact endpoint residual | patch `r(1)` into the answer continuation | endpoint/replay positive control |

The `K=1` true-cosine arm must exactly replay the source one-step oracle. The
primary relinearized endpoints use `K=16`; all five step counts are retained as
a planned dose-response.

The shuffled schedule is permuted at the exact-fiber level, so both sheets
remain paired and the path-increment marginal is preserved. It starts at each
example's own `m(r(0))` and adds another fiber's displacement:

```text
y_shuffled_i(t) = m_i(r(0))
                  + m_pi(i)(r(t)) - m_pi(i)(r(0)).
```

Its permutation seed is a fixed function of condition, checkpoint seed, and
regime.

## Primary endpoints and gates

A transport endpoint passes one checkpoint when its exact-bin accuracy loss
from that checkpoint's unchanged exact-reference clean baseline is at most
`0.03` on both composition and extrapolation.

The finite-radius hypothesis is supported only if all of the following hold:

1. the actual `m=64` reference endpoint and exact-residual endpoint pass `5/5`
   checkpoints in both arms;
2. the `K=16` true-cosine rollout passes at least `4/5` checkpoints in each
   arm;
3. the `K=16` path-moment rollout passes at least `4/5` checkpoints in each
   arm; and
4. the `K=16` shuffled path-moment rollout passes at most `1/5` checkpoint in
   each arm.

The source one-step result is a locked comparison, not a gate newly selected
from this campaign.

## Planned mechanistic measurements

At every step count and on both shifts, retain:

- exact-bin accuracy, target cross-entropy, ordered-moment error, and posterior
  Jensen-Shannon divergence from the actual `m=64` endpoint;
- rollout endpoint distance from `r(1)`, normalized by
  `||r(1)-r(0)||`;
- rollout path length, maximum step norm, and minimum task-gradient norm;
- actual residual-curve arc/chord ratio and maximum chord deviation;
- the fraction of each actual residual increment parallel and orthogonal to
  the local task gradient;
- one-step first-order moment error along the true on-manifold path; and
- the first step count, if any, that clears the task gate.

These are secondary descriptors. They cannot rescue a failed joint task gate.
No topology, link, or cobordism invariant is computed.

## Validity contracts

The campaign is invalid if any of the following fails:

- all locked campaign, implementation, repeat-array, dataset, result, model,
  and system-state digests match;
- the regenerated `m=1`, `m=64`, clean, and one-step source metrics replay to
  maximum absolute error `2e-6`;
- the exact endpoint residual continuation matches the actual reference-path
  endpoint to `2e-6` on every reported task metric;
- all gradients, residuals, posteriors, and metrics are finite;
- every task gradient is finite; its minimum norm and whether it clears
  `1e-8` are reported as scientific diagnostics rather than validity gates;
- exact-fiber path sharing and nested-grid contracts hold; and
- every system state digest remains unchanged.

A one-seed shakedown may waive population-quality gates but must satisfy every
systems and replay contract. Shakedown metrics are never scientific evidence.

## Outcome interpretation

| Outcome | Interpretation | Next action |
| --- | --- | --- |
| both `K=16` rollouts pass | the one-step ceiling exceeded a local validity radius | retain integrated local transport; do not retrain |
| path-moment passes, true-cosine fails | the terminal scalar coordinate, not only step size, is mismatched to the frozen answer geometry | analyze posterior-coordinate calibration |
| true-cosine passes, path-moment fails | semantic targeting helps but the model-output path is not a sufficient transport schedule | inspect path posterior geometry, not representation learning |
| neither passes; on-manifold local steps are accurate | locally valid updates accumulate off-manifold drift | test projection back to the observed residual curve |
| neither passes; on-manifold local steps are inaccurate | a scalar task-moment chart is locally insufficient | stop scalar-gradient writers |
| shuffled paths also pass | transport is nonspecific or the control is defective | classify invalid and repair the control |

No outcome licenses model retraining, another representation loss, a nonlinear
post-hoc readout, or a topology scan. A failed integrated audit falsifies the
shortest scalar-transport explanation and ends this writer branch unless a
new no-fit control is independently justified.

## Artifacts and execution

Primary root:

```text
data/experiments/tinyllm_reference_path_residual_transport/
    20260807_d8_corrected_v4/
```

Required artifacts are `campaign_results.json`, one strict `result.json` per
condition/checkpoint, and compact per-sample diagnostic arrays linked and
hashed from each result.

```bash
MPLCONFIGDIR=/tmp/matplotlib-reference-path-transport \
pixi run python -m \
  experiments.structure_net.tinyllm_reference_path_residual_transport \
  --device cuda:1 \
  --output \
  data/experiments/tinyllm_reference_path_residual_transport/20260807_d8_corrected_v4
```

## Method boundaries

The paths begin and end at two estimators derived from one stored synthetic
Gaussian repeat array. The true-cosine schedule is label-using and not
deployable. The path-moment schedule reads frozen model outputs and is also a
mechanistic oracle, not an estimator. Final-query residual transport says
nothing by itself about earlier block geometry, natural-language behavior,
real sensor noise, or architecture-population prevalence.
