# TinyLLM nuisance-scalar transformation-law preregistration

**Status:** COMPLETED — SCALAR IS ACTION-DEPENDENT IN 3/3 CHECKPOINTS; SEE MEASURED REPORT  
**Date:** 2026-08-07  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, `UNDERPOWERED`, post-outcome mechanistic diagnostic  
**Hypothesis:** `tinyllm-c2-nuisance-scalar-transformation-law-v1`  
**Schema:** `nal.tinyllm-c2-nuisance-scalar-transformation-law.v1`

**Validity amendment:** the first systems lifecycle exposed an arbitrary
carrier-basis sign mismatch between the regenerated CPU basis and stored
writer coordinates. The locked identity-only repair is documented in
`2026-08-07_tinyllm-nuisance-scalar-transformation-law-validity-amendment.md`;
scientific targets, thresholds, and classifications are unchanged.

**Measured report:**
[`2026-08-07_tinyllm-nuisance-scalar-transformation-law.md`](../08%20-%20Analysis/2026-08-07_tinyllm-nuisance-scalar-transformation-law.md)

## Decision question

The fresh-E scalar-sensor study established that the phase-conditioned task
covector transports, while source-fitted calibration, activation, post-MLP,
and output-posterior summaries do not predict its signed amplitude. The next
proposed intervention was a prospectively trained invariant/equivariant scalar
sensor.

Before training, test the necessary target-compatibility condition:

```text
an invariant scalar head s(g x) = s(x) can predict y(x)
only if the required correction target satisfies y(g x) = y(x).
```

Here `y` is the exact signed circular-output displacement from the frozen
order-four prediction to the direct rank-three state. It is the same scalar
used by the completed source-covector and task-activation sensor studies.

The primary hypothesis is that `y` is invariant under the declared
target-preserving observed similarity actions. If it is not, an invariant
canonicalizer is structurally mismatched to the correction target: the next
architecture must predict the frozen model's symmetry defect while retaining
the action context, rather than erase that context.

## Locked evidence

Reuse the completed exact-group campaign without generating a new cohort:

```text
data/experiments/tinyllm_local_metric_field_transport/
    20260807_d6_fresh_cohort/campaign_results.json
SHA-256 2aa80cff5882cb79754d732e9012c810c2624934bc5be9e03c9e77de4f525f55
implementation e3a05118971bba51e50e07e0fc426a8b321f0ecf823ded7dfec6b3539b630bfd
```

Reuse its frozen d6 degree-two checkpoints `7`, `29`, and `53`, rank-three
bases, order-four writers, group inputs, and exact action parameters. The
checkpoints are the replication units.

The existing group cohort uses seeds `430007/430008`, 16 quotient phases,
four nuisance replicates per phase, and 64 exact C2 orbits per regime. Its
input, token-change, target-pair, Jacobian, full-control, and local-tangent
contracts already pass in all three checkpoints.

No signed-scalar transformation metric from these group pairs was computed or
inspected before this document. Previously reported aggregate patch errors and
rank-two projector geometry do not determine the paired signed-scalar law.

## Declared group and target

For each composition/extrapolation example, compare the reference observation
with four target-preserving actions:

```text
g_(a, alpha, t) x = a R_alpha x + t
```

| arm | action |
| --- | --- |
| `amplitude` | positive scale only |
| `orientation` | planar rotation only |
| `offset` | constant planar translation only |
| `composed` | scale, rotation, and translation |

At the same frozen order-four predicted coordinate, compute the exact signed
output displacement for the reference and transformed cell:

```text
y(x)   = wrapped_bins(angle(direct_rank3(x)), angle(order4(x)))
y(gx)  = wrapped_bins(angle(direct_rank3(gx)), angle(order4(gx))).
```

The direct state, output angles, and finite-difference derivatives are used
only to define and validate the diagnostic target. They are not candidate
inputs to a deployable sensor.

## Metrics and controls

For every checkpoint, regime, and group arm, report paired:

- zero-referenced R2 of `y(x)` as a prediction of `y(gx)`;
- relative L2 and MAE;
- sign agreement where `|y(gx)| >= 0.01` bins;
- target RMS and action-difference RMS; and
- correlation when both variances are nonzero.

Use the existing phase-matched nuisance-replicate cycle as a correspondence
control. It preserves quotient phase but pairs `y(gx)` with a different base
nuisance realization. Report paired-minus-shuffled R2. This control tests
example correspondence; it is not required for the mathematical invariance
gate because a genuinely phase-only invariant scalar could match multiple
nuisance replicates.

## Validity gates

A checkpoint is valid only if:

1. the group campaign, result, array, checkpoint, basis, writer, and producing
   implementation hashes match;
2. all eight transformed group cells retain the existing input and target-pair
   contracts;
3. reference and transformed fine/coarse task covectors are finite and meet
   the established local-linearization thresholds: derivative cosine at least
   `0.98`, derivative relative L2 at most `0.15`, signed-error R2 at least
   `0.50`, residual-MAE fraction at most `0.50`, and sign agreement at least
   `0.75` above `0.01` bins; and
4. exact and direct rank-three controls pass while the zero state fails in
   every recomputed cell.

A validity failure is not evidence for or against scalar invariance.

## Primary invariance gate

A group cell satisfies the signed-scalar invariance law only when all three
conditions hold:

```text
paired zero-referenced R2 >= 0.90
paired relative L2 <= sqrt(0.10)
paired sign agreement >= 0.90.
```

A checkpoint passes only if all four group arms pass in both composition and
extrapolation. The hypothesis requires `3/3` checkpoints.

Separately report an example-correspondence specificity gate requiring paired
R2 to exceed phase-matched shuffled R2 by at least `0.10` in every cell. It
cannot rescue or invalidate the primary invariance law.

## Fixed classifications and decisions

Apply the first matching classification:

1. `invalid` if any provenance, input, numerical, local-linearization, or
   target control fails;
2. `scalar_nuisance_invariant_and_correspondence_specific` if every invariance
   and shuffled-specificity cell passes;
3. `scalar_nuisance_invariant_nonspecific` if invariance passes but
   correspondence specificity does not;
4. `scalar_action_dependent` if any valid action cell fails invariance.

Outcome-directed next actions are locked:

| Outcome | Consequence |
| --- | --- |
| invariant `3/3` | a prospectively trained invariant scalar head is target-compatible; preregister typed versus parameter-matched untyped training |
| action-dependent | do not train an invariant scalar head; formulate a symmetry-defect sensor that retains declared action/calibration context |
| checkpoint-stratified | retain action law as checkpoint-local and do not claim one shared scalar interface |
| invalid | repair only the digital/numerical contract under a disclosed new root |

This diagnostic cannot establish that a scalar sensor is learnable. It only
tests whether invariance is a structurally valid output type before training.

## Artifacts

- runner:
  `experiments/structure_net/tinyllm_nuisance_scalar_transformation_law.py`
- tests:
  `tests/structure_net/test_tinyllm_nuisance_scalar_transformation_law.py`
- result root:
  `data/experiments/tinyllm_nuisance_scalar_transformation_law/20260807_d6_existing_group`
- report:
  `docs/08 - Analysis/2026-08-07_tinyllm-nuisance-scalar-transformation-law.md`
- meta hypothesis:
  `tinyllm-c2-nuisance-scalar-transformation-law-v1`

The runner must preserve strict JSON, producing-code hashes, source artifact
hashes, per-result and NPZ hashes, exact resume, and the fact that no model,
writer, encoder, or observer is trained.

## Post-run contract amendment (not preregistered)

The first completed root exposed an immutable-resume mismatch after the runner
was strengthened to make the carrier-coordinate gauge explicit. Because the
primary signed scalar is basis-gauge-sensitive, the authoritative replay now
aligns each regenerated rank-three basis to the stored group-campaign
coordinates and requires maximum all-cell coordinate error at most `1e-5`.

Per the locked stop rule, the first root was not overwritten. The corrected
campaign was run under the new root
`data/experiments/tinyllm_nuisance_scalar_transformation_law/20260807_d6_existing_group_gauge_replay`.
The new basis-gauge contract passes `3/3`; maximum all-cell replay error is
`3.14e-6`. All primary values and all three `scalar_action_dependent`
classifications are unchanged. This paragraph records a digital-validity
repair, not a change to the hypothesis, endpoint, threshold, cohort, or
interpretation rule.
