# TinyLLM local metric-field specificity corrective preregistration

**Status:** PREREGISTERED — CORRECTIVE OUTCOMES NOT INSPECTED  
**Date:** 2026-08-07  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, `UNDERPOWERED`, frozen-artifact corrective diagnostic  
**Hypothesis:** `tinyllm-c2-local-metric-field-specificity-v1`  
**Schema:** `nal.tinyllm-c2-local-metric-field-specificity.v1`

## Reason for the corrective

The completed fresh-cohort metric-field campaign used a phase-matched
nuisance-replicate permutation as a negative specificity control. That control
is structurally incompatible with the registered hypothesis. If the local task
projector is invariant to the declared nuisance group, every nuisance replicate
at the same semantic phase belongs to the same equivalence class and should
match. Requiring that match to fail cannot distinguish successful nuisance
invariance from a nonspecific projector.

The original campaign remains immutable at:

```text
data/experiments/tinyllm_local_metric_field_transport/
    20260807_d6_fresh_cohort/campaign_results.json
SHA-256 2aa80cff5882cb79754d732e9012c810c2624934bc5be9e03c9e77de4f525f55
implementation e3a05118971bba51e50e07e0fc426a8b321f0ecf823ded7dfec6b3539b630bfd
```

Its registered `0/3` primary result is control-invalid and will not be used as
evidence for a nuisance-specific field. No corrective metric described below
has been computed before this document is written.

## Question

On the exact stored fresh-cohort Jacobian fields, is the local task-tangent
plane simultaneously:

1. invariant across the declared acquisition nuisance transformations;
2. equivalent across independent nuisance replicates at fixed semantic phase;
3. distinct from a projector at a different semantic phase; and
4. distinct from an isotropic random rank-two plane?

This is a geometry-only corrective. It does not reinterpret the original
causal control and does not fit a map, writer, probe, metric, or model.

## Locked artifacts and units

Reuse exactly the three seed records and compressed projector arrays referenced
by the immutable source campaign. The checkpoints `7`, `29`, and `53` are the
replication units. The eight composition/extrapolation by acquisition-action
cells within a checkpoint are repeated measurements.

For each cell, use the stored rank-two projectors for:

```text
reference observation x
transformed observation g x
```

The source campaign's provenance, input/pair, numerical, full-control, and
fresh local-tangent gates must all pass before a checkpoint is eligible.

## Corrected controls

The 64 orbits are ordered as 16 quotient-phase values with four nuisance
replicates each.

### Positive nuisance-equivalence control

Cyclically advance the reference nuisance replicate by one while holding its
quotient phase fixed. This is now correctly treated as an equivalence control.

### Negative semantic-phase control

Cyclically advance the reference quotient phase by exactly five of the 16
bins while preserving nuisance-replicate index. Five bins is locked because it
is nonzero and is neither a quarter-turn nor a half-turn of the sampled cycle.
No shift is selected from outcomes.

### Negative random-plane control

Generate one deterministic isotropic rank-two projector per orbit, seed,
regime, and arm from the SHA-256-derived control seed. No direction is selected
from outcomes.

All kernel comparisons use absolute line cosine, so eigenvector sign is
irrelevant.

## Locked cell gates

A cell passes only when all of the following hold:

1. paired median kernel cosine is at least `0.95`;
2. paired 10th-percentile cosine is at least `0.90`;
3. paired p95 projector distance is at most `sqrt(1 - 0.90^2) = 0.43589`;
4. nuisance-equivalence median cosine is at least `0.95`;
5. the absolute paired-versus-nuisance-equivalence median difference is at
   most `0.02`;
6. paired median cosine exceeds the five-bin semantic-phase control by at
   least `0.10`; and
7. paired median cosine exceeds the random-plane control by at least `0.20`.

A checkpoint passes only if all eight cells pass. The hypothesis is supported
only if all `3/3` checkpoints pass. It remains underpowered regardless of
outcome.

## Fixed classifications

Apply the first matching rule:

1. `invalid` when a locked source, provenance, validity, shape, rank, or finite
   contract fails;
2. `nuisance_invariant_phase_specific_field` when all eight corrected cells
   pass;
3. `phase_nonspecific_field` when paired geometry and nuisance equivalence
   pass every cell but semantic-phase specificity does not;
4. `nuisance_variant_field` when paired geometry fails any cell;
5. `mixed_metric_field_geometry` otherwise.

No mean across arms, regimes, or checkpoints can rescue the joint gate.

## Outcome meanings and stop rule

| Outcome | Interpretation | Next shortest action |
| --- | --- | --- |
| 3/3 phase-specific | nuisance invariance is geometrically supported after correcting the control | run one phase-shifted causal continuation control |
| phase nonspecific | the task plane is too broad or constant for correspondence evidence | stop transport claims; do not train a sidecar |
| nuisance variant | the identity nuisance action is false | retain checkpoint-local causal charts |
| mixed | geometry is arm, shift, or checkpoint dependent | localize the failed group generator before any causal rerun |
| invalid | the stored campaign cannot support the corrective | repair provenance or numerics under a new root |

Link cobordism is not triggered by any outcome here. It remains gated on a
separately declared codimension-two degree or branch-locus defect.

## Artifacts

- runner:
  `experiments/structure_net/tinyllm_local_metric_field_specificity.py`
- tests:
  `tests/structure_net/test_tinyllm_local_metric_field_specificity.py`
- primary root:
  `data/experiments/tinyllm_local_metric_field_specificity/20260807_corrective_v1`
- report:
  `docs/08 - Analysis/2026-08-07_tinyllm-local-metric-field-specificity.md`
- meta-hypothesis:
  `tinyllm-c2-local-metric-field-specificity-v1`

## Method boundaries

The correction reuses selected checkpoints and already-generated fresh
cohorts. It tests projector geometry only; causal use of the phase-shifted
control is explicitly deferred. The five-bin shift is one deterministic
semantic mismatch, not a complete group orbit. The observed acquisition group
covers scale, planar orientation, and constant offset, not every N3 nuisance.

