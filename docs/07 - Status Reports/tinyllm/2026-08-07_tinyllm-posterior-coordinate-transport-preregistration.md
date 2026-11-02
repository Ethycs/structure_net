# TinyLLM posterior-coordinate transport rank ladder preregistration

**Status:** LOCKED BEFORE POSTERIOR-TRANSPORT OUTCOMES  
**Date:** 2026-08-07  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, outcome-directed frozen-checkpoint mechanistic diagnostic  
**Hypothesis:** `tinyllm-posterior-coordinate-transport-v1`  
**Planned schema:** `nal.tinyllm-posterior-coordinate-transport.v1`

## Provenance boundary

This experiment is selected after the valid corrective reference-path
transport result. The following facts are prior evidence:

- actual `m=64` and exact-residual endpoints pass `5/5` checkpoints in both
  structured arms;
- at `K=16`, the latent true-cosine schedule passes `5/5` analytic and `4/5`
  learned checkpoints;
- the actual ordered path-moment schedule passes `0/5` in both arms;
- its fiber-block-shuffled control passes `0/5` in both arms; and
- the scalar path rollout nearly matches the endpoint moment while retaining
  a different posterior and residual state.

The sole transport source is:

```text
data/experiments/tinyllm_reference_path_residual_transport/
    20260807_d8_corrected_v4/campaign_results.json

campaign SHA-256
6b232f523cd570f10ebfcc07c47abae6a724b568d4cfc2852e4b91ccff01321f

implementation SHA-256
5cb1944e57db0515dbc7ad5e3956a328be733cdda7f5dda34631d21e8cf3a81c

result-manifest SHA-256
5d765445082346512092dfa0acb00e3e39db369bd71829cbe4d96c8871593b4b
```

This design is therefore outcome-directed. A positive result can identify a
minimal vector chart, but it is not an independent confirmation of the wider
quotient program.

## Question

Does the scalar rollout fail because it discards answer-relevant posterior
shape, or because no answer-output coordinate system is sufficient to
integrate the trained residual path?

The directional hypothesis is that a small fixed vector of ordered answer
coordinates transports the frozen task where one ordered moment fails. The
strong compact-chart prediction is that rank at most four passes the complete
population gate.

## Fixed systems and data

Reuse the two frozen source arms and seeds `7`, `17`, `29`, `41`, and `53`:

- `analytic_calibrated`;
- `learned_calibrated_equivariant`.

Use the exact composition and extrapolation datasets, stored `q_1 -> q_64`
shortest reference paths, model checkpoints, front ends, answer rows, and
system-state hashes from the source campaign. No parameter is trained or fit.

## Fixed posterior coordinates

Let `p(r)` be the frozen answer posterior and let

```text
ell(r) = log(clamp(p(r), 1e-12))
         - mean_j log(clamp(p_j(r), 1e-12)).
```

The answer bins are ordered by their fixed centers. Construct a deterministic
orthonormal DCT-II basis on those bins, omit the constant vector, and retain
the first

```text
r in {1, 2, 4, 8, full}
```

nonconstant coordinates. `full` means the complete centered answer-logit
space and is invariant to the choice of orthonormal basis. Basis construction
uses no examples, labels, residuals, or outcomes.

The inherited ordered-posterior-moment rollout remains a named scalar
comparator; DCT rank one must not be relabeled as an exact replay of that
different scalar.

## Residual transport

At each nested reference-path point retain the actual coordinate schedule

```text
u_r(t) = B_r^T ell(r_actual(t)).
```

Starting from the actual `m=1` residual, recompute the frozen Jacobian

```text
J_r(z) = d u_r(z) / d z
```

at every step and apply the minimum-norm SVD pseudoinverse update

```text
delta = pinv(J_r; rcond=1e-6) @ (u_r(t_next) - u_r(z)).
```

The relative singular-value cutoff is fixed at `1e-6`. Do not tune damping,
clip steps, project to a learned manifold, or select rank by task outcome.
Record effective numerical rank, condition number, coordinate residual, and
step norm. Rank loss or a large update is a scientific observation, not a
systems-validity failure, provided all values remain finite.

Use `K in {4,16}`. `K=16` is primary; `K=4` is retained because the predecessor
showed nonmonotone task behavior across step counts.

## Controls

For every rank retain:

1. the actual frozen reference path and exact endpoint residual as positive
   controls;
2. the inherited ordered-moment rollout as the scalar comparator;
3. a fiber-block-shuffled posterior-coordinate schedule preserving both sheets
   and the coordinate-increment marginal; and
4. exact endpoint-coordinate error, posterior JS, and normalized residual
   distance from the actual `m=64` endpoint.

The shuffled permutation must be fixed from condition, checkpoint seed, and
regime before any task outcome is inspected.

## Primary gates and classification

A checkpoint passes when its exact-bin accuracy loss from its own unchanged
exact-reference clean baseline is at most `0.03` on both composition and
extrapolation.

For each rank, require:

- actual and exact-residual endpoints pass `5/5` checkpoints in both arms;
- the `K=16` coordinate rollout passes at least `4/5` in each arm; and
- the matched shuffled schedule passes at most `1/5` in each arm.

Use the following locked classification:

| Outcome | Classification | Decision |
| --- | --- | --- |
| smallest passing rank `<=4` | `compact_vector_chart` | retain that rank as the conservative answer-coordinate chart |
| first passing rank is `8` or `full` | `high_rank_answer_chart` | posterior shape is sufficient but not compact |
| no rank passes, including `full` | `answer_coordinates_nonintegrable` | stop output-coordinate writers; test direct projection to the observed residual curve only if needed |
| a shuffled schedule exceeds its ceiling | `invalid_control` | repair the control; do not interpret rank |

Ranks are reported as a nested ladder; no lower-rank miss may be hidden by a
higher-rank pass. The full answer-space result is the primary falsifier of the
entire output-coordinate branch.

## Validity contracts

The campaign is invalid unless:

- every source campaign, result, array, dataset, checkpoint, and system-state
  hash validates;
- actual endpoints and inherited scalar metrics replay within `2e-6`;
- DCT bases are orthonormal to `1e-10` and nested exactly;
- actual and shuffled schedules preserve exact fiber pairing;
- coordinate Jacobians, updates, posteriors, and metrics are finite;
- exact-residual endpoint continuation replays the actual endpoint within
  `2e-6`; and
- all system-state hashes remain unchanged.

No outcome licenses model retraining, representation penalties, observer
fitting, topology scans, or link-cobordism analysis. This experiment is the
last no-fit output-coordinate writer test in the current branch.

## Required artifacts

The runner must write one strict result per condition/checkpoint, compact
diagnostic arrays, a ten-result manifest, implementation and source hashes,
exact-resume evidence, a NAL-STD report, and a meta-hypothesis record that keeps
the compact, high-rank, and nonintegrable outcomes distinct.
