# TinyLLM local task-metric field transport preregistration

**Status:** PREREGISTERED — FRESH-COHORT TRANSPORT OUTCOMES NOT INSPECTED  
**Date:** 2026-08-07  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, `UNDERPOWERED`, frozen-checkpoint mechanistic diagnostic  
**Hypothesis:** `tinyllm-c2-local-metric-field-transport-v1`  
**Schema:** `nal.tinyllm-c2-local-metric-field-transport.v1`

## Evidence boundary and question

The completed local-continuation sequence is known to show:

- orbit-local rank-two circular-moment tangents repair all 12 reused
  composition/extrapolation cells in the valid corrective campaign;
- the complementary rank-one kernel is causally inert at that residual scale;
- two independently preregistered scalar-angle tangent protocols also repair
  all 12 cells from different failed writers; and
- no one checkpoint-level task metric is stable under the previous unpaired
  cross-cell comparison.

Those observations do not determine whether the metric variation is arbitrary
or follows a declared nuisance-symmetry transport law. This experiment asks:

> On genuinely fresh, phase-matched sensor pairs related by an exact observed
> nuisance-group action, does the frozen continuation's local task-tangent
> projector transport with the neutral carrier's declared representation?

No transport, alignment, probe, decoder, writer, sidecar, basis, or model is
fitted. Outcomes from the fresh cohort have not been generated or inspected.

## Why the deck action is not the test

The measured three-coordinate carrier is the neutral Reynolds defect. Its
declared deck representation is therefore the trivial representation of
`C2`. After exact sheet averaging, testing only sheet exchange would make the
projector law tautological.

The nontrivial group is the observed acquisition-similarity subgroup acting on
each three-channel sensor history:

```text
g_(a,alpha,t) x = a R_alpha x + t,
```

where `a > 0`, `R_alpha` rotates the first two sensor channels and leaves the
third fixed, and `t in R3` is constant over time. This transformation does not
use phase, branch, target, or labels and leaves the semantic target unchanged.
It generates the amplitude, orientation, and offset nuisance axes in a
declared exact order: rotate, scale, then translate.

Because the rank-three carrier is declared nuisance-neutral and no learned
coordinate transport is allowed, lock

```text
rho(g) = I_3.
```

The contragredient covector law and its row-space form are therefore

```text
J(g x) = J(x),
P_T(g x) = P_T(x),
```

up to finite tokenization and continuation numerics. This is a falsifiable
architectural-interface claim, not a fitted alignment.

## Locked sources and replication units

Reuse exactly the three selected d6 degree-two checkpoints `7`, `29`, and
`53`, their source-fitted rank-three block-0-attention defect bases, frozen
order-four quotient writer, calibrated circular readout, and continuation
code from:

```text
data/experiments/tinyllm_local_continuation_tangent_kernel/
    20260807_d6_corrective_v2/campaign_results.json
SHA-256 8b2f25dc61d1ffb93ba9dc66f1a85f36c07bda5a43ed3ed34a9627ca1485c12a
implementation 31ef191a31bbd5d509fb912cd5164385d846039a3b7c98aca04e6f5e29835c38
```

The original writer-capacity predecessor remains locked at SHA-256
`7ab6079d56bc8dc78cc5e1f4011c581b8f507bcd8ec20573bbf8b11227df8d1b`.
The frozen checkpoints are the three replication units; within-checkpoint
phases, nuisance replicates, arms, and shifts are repeated measurements.

## Fresh paired cohort

Use cohort seeds:

```text
composition   430007
extrapolation 430008
```

Neither seed appeared in the preceding alignment-fit or held-out cohorts.
Each regime contains 16 uniformly spaced quotient phases with four nuisance
replicates per phase, for 64 exact `C2` orbits. The two deck sheets of an orbit
share all nuisance values and noise.

For every orbit, generate one canonical observed history `x` with fresh N3
direction, speed, harmonic, drift, and noise values. Apply four paired group
elements while keeping those other nuisance values and the target fixed:

| arm | group element |
| --- | --- |
| `amplitude` | `g_(a,0,0)` |
| `orientation` | `g_(1,alpha,0)` |
| `offset` | `g_(1,0,t)` |
| `composed` | `g_(a,alpha,t)` |

Composition parameters use the generator's held-out N3 composition ranges;
extrapolation parameters use its declared outside-range values. Group actions
are applied to the continuous observed history before the unchanged tokenizer.

The numerical input contract requires, in every arm and regime:

- pre-token clipping fraction at most `0.01`;
- non-finite value count zero; and
- mean changed-token fraction from the canonical pair at least `0.02`.

Failure invalidates the checkpoint rather than excluding an arm.

## Frozen metric field

At the canonical and transformed block-0 post-attention writer states,
differentiate the frozen two-dimensional circular posterior moment with
respect to the same three carrier coordinates:

```text
J_x, J_gx in R^(2 x 3).
```

Use autograd for the registered Jacobian. A centered finite-difference check
at coordinate step `1e-2` must have maximum relative error at most `0.05`,
matching the validated corrective protocol. SVD uses relative tolerance
`1e-6` and absolute tolerance `1e-10`; every Jacobian must have rank two.

Define

```text
P_x  = J_x^+ J_x,
P_gx = J_gx^+ J_gx.
```

For each exact pair measure:

```text
d_P = ||P_gx - P_x||_F / sqrt(2),
c_K = |k_gx dot k_x|,
```

where `k` is the unit rank-one kernel line. For rank-two projectors in three
dimensions, `d_P` is the sine of the kernel-line angle.

The deterministic phase-matched correspondence control cyclically shifts the
canonical nuisance replicate by one within each four-replicate phase group.
A second control is a deterministic isotropic rank-two projector. Neither
control fits or selects a direction from outcomes.

## Geometric transport endpoint

Each arm and regime passes the trivial-transport geometry gate only when:

1. median paired kernel-line absolute cosine is at least `0.95`;
2. the 10th-percentile paired cosine is at least `0.90`;
3. p95 projector distance is at most `sqrt(1 - 0.90^2) = 0.43589`; and
4. median correct-pair cosine exceeds the phase-matched nuisance-shuffled
   median by at least `0.10`.

A checkpoint passes geometric transport only if all eight arm/regime cells
pass. The four arms may be reported separately but cannot rescue the joint
gate.

## Causal transported-tangent endpoint

Let `e_g` be the exact-minus-order-four coordinate residual at the transformed
state. Continue these unchanged frozen states:

| state | transformed carrier write | role |
| --- | --- | --- |
| `predicted` | `c4` | failed-writer reference |
| `full` | `c4 + e_g` | exact rank-three control |
| `local_tangent` | `c4 + P_gx e_g` | local positive control |
| `transported_tangent` | `c4 + P_x e_g` | primary symmetry-law intervention |
| `shuffled_tangent` | `c4 + P_x,shift e_g` | phase-matched correspondence control |
| `random_tangent` | `c4 + P_random e_g` | isotropic projector control |

Retain the validated continuous endpoint: circular alignment loss at most
`0.005`, mean shift at most `0.125` bins, p95 shift at most `0.50` bins,
resolved sampling, and winding degree within `0.10` of degree two.

For each checkpoint, causal transport passes only when:

1. `full` and `local_tangent` pass every arm/regime cell;
2. `transported_tangent` passes every cell;
3. `shuffled_tangent` and `random_tangent` each fail at least one arm in both
   composition and extrapolation; and
4. transported tangent improves regime-aggregate mean shift by at least
   `0.05` bins over each control.

The primary checkpoint gate requires both geometric and causal transport.
The campaign supports the hypothesis only if all `3/3` selected checkpoints
pass. It remains underpowered regardless of outcome.

## Fixed classifications

Apply the first matching rule per checkpoint:

1. `invalid` if provenance, input, rank, finite-difference, decomposition, or
   the full rank-three control fails;
2. `invariant_metric_field_transport_supported` if both geometric and causal
   transport pass;
3. `causal_transport_without_geometric_invariance` if only causal transport
   passes;
4. `geometric_invariance_without_causal_transport` if only geometric
   transport passes;
5. `local_tangent_not_fresh_cohort_sufficient` if the local tangent fails any
   fresh cell;
6. `nuisance_specific_metric_field` otherwise.

No secondary mean, arm, checkpoint, or descriptive control can rescue the
joint primary gate.

## Outcome meanings and stop rules

| Outcome | Interpretation | Next shortest action |
| --- | --- | --- |
| geometry and causal transport pass 3/3 | the neutral carrier admits the declared nuisance-invariant task-metric interface | test a source-only scalar coefficient before training |
| causal passes, geometry fails | exact projector equality is too strong but transported corrections remain task-equivalent | retain a causal equivalence class; do not fit metric tensors |
| geometry passes, causal fails | local projector similarity does not control the finite task correction | stop the sidecar branch; measure only the mixed second derivative |
| local tangent fails fresh cells | prior tangent sufficiency was support-relative | stop transport work and retain the prior atlas |
| both transport gates fail | the metric field is nuisance/state specific under the declared group | stop universal-sidecar claims; keep checkpoint-local causal charts |
| invalid | the digital group or numerical continuation contract failed | repair under a new root without interpreting transport outcomes |

## Artifacts and execution plan

- runner:
  `experiments/structure_net/tinyllm_local_metric_field_transport.py`
- tests:
  `tests/structure_net/test_tinyllm_local_metric_field_transport.py`
- systems-only root:
  `data/experiments/tinyllm_local_metric_field_transport/20260807_shakedown_cuda`
- primary root:
  `data/experiments/tinyllm_local_metric_field_transport/20260807_d6_fresh_cohort`
- report:
  `docs/08 - Analysis/2026-08-07_tinyllm-local-metric-field-transport.md`
- meta hypothesis:
  `tinyllm-c2-local-metric-field-transport-v1`

Focused CPU contracts, a real CUDA lifecycle, immutable aggregate resume,
strict JSON, implementation hashing, and scientific fingerprints must pass
before the fresh primary aggregate is interpreted.

## Method boundaries

The acquisition-similarity group covers scale, planar orientation, and
constant channel offset, not speed, direction, harmonic order, drift, or
noise as group generators. Tokenization makes the model's digital action only
an approximation to the exact continuous sensor action; the clipping and
changed-token contracts delimit that approximation. Exact residual amplitudes
remain diagnostic and unavailable at inference time. The carrier basis,
writer, readout, and three checkpoints were selected in prior work. Fresh
sensor pairs improve support evidence but do not create independent model
seeds or population prevalence.
