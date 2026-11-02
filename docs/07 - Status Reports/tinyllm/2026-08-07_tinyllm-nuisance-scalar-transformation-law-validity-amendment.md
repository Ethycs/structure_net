# TinyLLM nuisance-scalar transformation-law validity amendment

**Status:** LOCKED DIGITAL-REPLAY REPAIR — SCIENTIFIC TARGETS AND GATES UNCHANGED  
**Date:** 2026-08-07  
**Applies to:** `tinyllm-c2-nuisance-scalar-transformation-law-v1`  
**Supersedes only:** evidence roots without an explicit carrier-basis gauge replay contract

## Trigger

The separate CPU systems lifecycle at
`20260807_shakedown_cpu` was invalid. Provenance, exact input replay, and all
target controls passed, but the inherited full-residual local-linearization
gate failed. The invalid run is quarantined and supplies no evidence about the
scalar transformation law.

Inspection was restricted to the failed validity quantities and the absolute
magnitude of the invalid signed scalar. No paired invariance, shuffled-control,
or checkpoint-aggregate metric was used to choose this repair.

The failure is a digital coordinate-gauge mismatch. The predecessor stored its
rank-three target and writer coordinates but did not store the fitted basis
tensor. Re-fitting the same SVD subspace on CPU preserved the basis summary but
flipped its first two coordinate axes. For systems seed 7, the least-squares
map from the predecessor coordinates to the regenerated coordinates was

```text
[-1.00000003,  0.00000005, -0.00000000]
[-0.00000009, -1.00000006,  0.00000001]
[ 0.00000018,  0.00000008,  1.00000014]
```

Its maximum orthogonality error was `2.82e-7`; it reduced maximum coordinate
replay error from `15.5434` to `1.60e-6`. Injecting predecessor writer
coordinates through the unanchored regenerated basis therefore changed the
state and made the validity calculation meaningless.

## Locked repair

For each checkpoint independently:

1. regenerate the source-fitted rank-three basis exactly as before;
2. regenerate only the composition/reference group cell;
3. load `composition__reference__target` from that checkpoint's already hashed
   predecessor NPZ;
4. solve the deterministic least-squares gauge map
   `Q = lstsq(C_stored, C_regenerated)` on CPU;
5. set `B_replay = Q B_regenerated`; and
6. require coordinates computed with `B_replay` to reproduce every stored
   reference/action target in both regimes.

This is an identity replay, not a scientific fit. It changes only the arbitrary
sign/rotation convention of the same rank-three subspace. The repair is valid
only if all of the following pass:

```text
max |Q^T Q - I|                       <= 1e-5
composition/reference gauge-fit error <= 1e-5
maximum target replay error, all cells <= 1e-5
```

The predecessor writer-map digest must remain unchanged. The aligned basis
must also remain orthonormal within `1e-5`.

## Unchanged scientific contract

The exact signed-scalar definition, four nuisance actions, two regimes,
three checkpoints, invariance thresholds, phase-matched shuffled control,
classifications, and outcome-directed decisions remain exactly as registered.
No threshold is relaxed. No model, writer, encoder, observer, or scientific
mapping is fit.

The authoritative repaired campaign and its cross-device checks use distinct
roots:

```text
systems: data/experiments/tinyllm_nuisance_scalar_transformation_law/
         20260807_shakedown_cpu_v2
primary: data/experiments/tinyllm_nuisance_scalar_transformation_law/
         20260807_d6_existing_group_gauge_replay
cross-device CPU replay: data/experiments/tinyllm_nuisance_scalar_transformation_law/
                         20260807_d6_existing_group_v2
```

The invalid `20260807_shakedown_cpu` root remains retained as lifecycle audit
evidence. The earlier GPU `20260807_d6_existing_group` campaign reaches the
same classification and numerical measurements but predates the explicit
gauge-replay gate, so it is retained as superseded audit evidence rather than
pooled with the authoritative campaign. The CPU cross-device replay agrees
with the authoritative paired R2 values within `1e-5` and is likewise a
reproducibility check, not an additional checkpoint replicate.
