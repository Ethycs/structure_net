# TinyLLM cross-seed symmetry-feature swap preregistration

**Status:** PREREGISTERED — PRIMARY SWAP OUTCOMES NOT INSPECTED  
**Date:** 2026-08-06  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, frozen-checkpoint causal diagnostic  
**Hypothesis:** `tinyllm-cross-seed-symmetry-feature-gauge-v1`  
**Schema:** `nal.tinyllm-cross-seed-symmetry-feature-swap.v1`

## Question and prediction

Did the calibrated-equivariant front end learn one portable, symmetry-fixed
three-channel gauge across independently trained checkpoints, or only five
checkpoint-local invariant coordinates?

The existing learned front end constructs an `SO(2)`-equivariant unit vector
and then exposes the invariant feature

```text
z_s(x) = (dot(v_s(x), orientation), cross(orientation, v_s(x)), signed_speed).
```

For source checkpoint `s` and target checkpoint `t`, the intervention replaces
`z_t(x)` by `z_s(x)` immediately before target `t`'s frozen `scalar_map`.
Nothing is fit, trained, calibrated, or selected. If architectural symmetry
fixed a common semantic gauge, the target continuation should preserve its
ordinary task behavior under this swap on both held-out composition and
outside-range extrapolation.

## Evidence boundary

The five d8 learned calibrated-equivariant checkpoints and their successful
within-checkpoint quotient outcomes are already known. Cross-seed feature-swap
outcomes are not. This study is a preregistered mechanistic decomposition of
those retained systems, not an independent training replication.

The unit of replication is the independently trained **target checkpoint**.
The twenty directed source-target pairs are repeated interventions, not twenty
independent training replicates.

## Frozen systems and intervention

- source campaign:
  `data/experiments/tinyllm_calibrated_frontend_causal/20260806_d8_preregistered`;
- condition: `learned_calibrated_equivariant` only;
- seeds: `7, 17, 29, 41, 53`;
- all `5 * 4 = 20` directed pairs with `s != t`;
- frozen artifacts: source/target `model.pt`, `frontend.pt`, `result.json`, and
  predecessor `campaign_results.json`;
- every artifact digest and predecessor scientific fingerprint must be
  recorded and verified before evaluation;
- no optimizer, learned probe, fitted map, label-derived transform, model
  update, or cut selection is allowed.

For every target, evaluate the following continuations:

| condition | feature entering the frozen target continuation | role |
| --- | --- | --- |
| `target_direct` | target `z_t`, through target `scalar_map` | exact replay control |
| `source_feature_swap` | source `z_s`, through target `scalar_map` | primary intervention |
| `source_scalar_swap` | source `scalar_map_s(z_s)`, after target `scalar_map` | secondary localization |
| `shuffled_source_feature` | deterministic within-cell permutation of `z_s` | correspondence control |
| `half_turn_source_feature` | `(-z_s[0], -z_s[1], z_s[2])` | semantic group control |

`source_scalar_swap` asks whether the one-dimensional task bottleneck is
portable even when the preceding invariant chart is not. It is secondary and
cannot rescue a failed primary feature-swap gate.

## Exact acquisition-group contract

Before task evaluation, apply two declared positive-similarity acquisition
actions to each checkpoint and cohort:

```text
sensor_xy' = a R sensor_xy + offset_extra + drift_extra * time
orientation' = R orientation
amplitude' = a amplitude
offset' = a R offset + offset_extra
drift' = a R drift + drift_extra
```

Signed speed is unchanged. The two actions use fixed nonzero rotations and
nonunit positive scales. The maximum absolute discrepancy between original
and transformed `z` must be at most `1e-5`. This is a hard implementation
contract, not evidence for cross-seed portability.

## Held-out data

Generate four deterministic, mutually disjoint paired N3 cohorts with the
unchanged calibrated generator:

| cohort | regime | seed | examples |
| --- | --- | ---: | ---: |
| A | composition | 91,301 | 512 |
| A | extrapolation | 92,301 | 512 |
| B | composition | 91,302 | 512 |
| B | extrapolation | 92,302 | 512 |

These seeds are disjoint from training, predecessor probes, and earlier
calibrated evaluation. Every source and target sees the exact same serialized
examples, observed sensors, calibration packets, targets, and batching.

## Measurements

For every condition and cell, record:

- exact-bin accuracy and loss relative to `target_direct`;
- target cross-entropy and increase relative to `target_direct`;
- correlation between posterior mean on the fixed `[-1,1]` bin centers and
  true future cosine, plus loss relative to `target_direct`;
- posterior-mean RMSE against true future cosine;
- mean Fisher--Rao distance from the `target_direct` posterior;
- mean and p95 absolute difference of the posterior mean from
  `target_direct`.

For the raw invariant features, also record the mean angular displacement,
circular concentration, component correlations, and scalar-map discrepancy
between `z_s` and `z_t`. These are secondary geometry diagnostics.

## Primary cell and pair gates

A `source_feature_swap` cell passes only when all are true:

1. the exact direct replay has maximum posterior error at most `1e-6`;
2. posterior-mean cosine correlation is at least `0.90` and loses at most
   `0.02` relative to `target_direct`;
3. exact-bin accuracy loses at most `0.03` relative to `target_direct`;
4. target cross-entropy increases by at most `0.05` nats;
5. mean Fisher--Rao distance from `target_direct` is at most `0.10`;
6. its target cross-entropy is at least `0.10` nats lower than both the
   shuffled and half-turn controls.

A directed pair passes only if the acquisition-group contract and all six
cell conditions pass in all four held-out cells.

## Campaign gate

The portable symmetry-gauge hypothesis is supported only if:

1. all five checkpoints pass the acquisition-group contract;
2. at least `16/20` directed pairs pass jointly;
3. at least `4/5` target checkpoints receive passing swaps from at least
   `3/4` independent source checkpoints.

No marginal average may substitute for the target-level joint rule. The
secondary scalar-swap analysis cannot promote a failed campaign.

## Outcome interpretation

| Outcome | Meaning | Next action |
| --- | --- | --- |
| primary gate passes | calibrated equivariance produced a portable invariant feature gauge | reuse this interface as the fixed typed front end for group-equivariant TinyLLM |
| feature fails, scalar passes | the one-dimensional semantic bottleneck is shared, but the three-channel invariant chart remains checkpoint-local | freeze or analytically define the pre-scalar gauge; do not train a larger sidecar |
| feature and scalar fail | even the task bottleneck is co-adapted to each transformer continuation | test a fully fixed analytic interface or train matched models with shared typed writers |
| controls reproduce | apparent portability is explained by task marginal or continuation insensitivity | reject the portable-gauge claim and audit the task/control construction |
| group contract fails | the intervention does not implement the declared symmetry | invalidate the campaign and repair before interpretation |

## Integrity and artifacts

Focused algebraic and load/replay tests plus a disposable CUDA lifecycle must
pass before the primary run. The shakedown is
`systems_lifecycle_only_not_quality_evidence` and cannot be pooled.

Completed records are byte-immutable under matching schema, implementation
digest, checkpoint digests, data seeds, and scientific fingerprint. JSON must
be strict (`allow_nan=False`). The runner records zero trained parameters and
zero fitted maps.

- runner:
  `experiments/structure_net/tinyllm_cross_seed_symmetry_feature_swap.py`;
- tests:
  `tests/structure_net/test_tinyllm_cross_seed_symmetry_feature_swap.py`;
- primary root:
  `data/experiments/tinyllm_cross_seed_symmetry_feature_swap/20260806_d8_preregistered`;
- report:
  `docs/08 - Analysis/2026-08-06_tinyllm-cross-seed-symmetry-feature-swap.md`;
- meta hypothesis:
  `tinyllm-cross-seed-symmetry-feature-gauge-v1`.

## Method boundaries

The learned equivariant vector contains trained temporal weights and invariant
mixing, so equivariance alone does not mathematically force equal coordinates
across seeds. A positive result establishes compatibility with these five
frozen continuations on the declared generator and shifts; it is not a
universal uniqueness theorem. A negative result does not refute acquisition
equivariance—it distinguishes within-checkpoint invariance from cross-seed
gauge fixing. All task comparisons remain conditioned on each target's frozen
scalar map, embedding, transformer, and decoder.
