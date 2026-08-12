# TinyLLM observable C3 temporal-quotient Stage-0 registration

**Status:** FROZEN BEFORE LIFECYCLE EXECUTION

**Date:** 2026-08-11

**Conformance:** NAL-STD-EXPERIMENT `SYSTEMS LIFECYCLE / NOT SCIENTIFIC EVIDENCE`

**Parent hypothesis:** `tinyllm-c3-temporal-quotient-training-v1`

**Parent design:** [C3 temporal-quotient training design](2026-08-11_tinyllm-c3-temporal-quotient-training-design.md)

## Decision

This registration authorizes only the implementation and systems lifecycle in
Stage 0 of the parent design. It does not authorize a primary seed, a population
claim, or any use of shakedown task metrics as trained-model evidence.

Stage 0 passes only if the generator/action contracts, matched-arm protocol,
architecture contracts, real optimization path, checkpoint reload, and exact
resume all pass. A failure licenses a systems repair under a new implementation
hash; it cannot be interpreted as evidence for or against architectural
invariance.

## Prospective training-support definition

The no-training preflight fixed the composition and extrapolation families but
did not operationalize the training family. Before any model outcome exists,
freeze the following support:

- phase is uniform on the circle;
- signed speed uses the composition magnitude range `[.04,.12]`;
- the deck element is uniform in `C3`;
- each latent example selects exactly one active calibration-nuisance family:
  amplitude, offset, drift, or the identity family;
- the active nuisance uses its full composition marginal range while the other
  values remain at `A=1.2`, `o=0`, and `d=0`.

The composition evaluation independently combines amplitude, offset, and drift
inside those same marginal ranges. The extrapolation evaluation uses the wider
ranges already frozen by the preflight. Thus composition is a held-out product
of seen one-factor nuisance families, not an IID alias.

Training contains 4,096 observed examples made from 2,048 latent histories.
Each history appears under two distinct deck elements with one shared target and
calibration packet. Every minibatch contains complete two-sheet pairs. Raw,
analytic, and learned arms receive byte-identical token tensors, calibration,
targets, latent values, deck labels, and pair/minibatch indices.

## Model interface

All arms use `BOS + eight temporal feature tokens + query` and the same fixed
sixteen-token ordered interval answer set.

| Arm | Temporal feature | Trainable front-end capacity |
| --- | --- | --- |
| raw | three calibrated channel values | `3 -> d_model` injection |
| analytic | real/imaginary cubic first-character carrier | `2 -> d_model` injection |
| learned | exact invariant produced by a shared scalar map, fixed first-character transform, complex-linear mixer, normalization, and cubing | same `2 -> d_model` injection plus shared map/mixer |

The analytic and learned injections MUST initialize identically. TinyLLM MUST
initialize identically in all three arms for a shared preset and seed. The
learned encoder receives only task gradients.

## CPU lifecycle

Run all three arms with the repository's `tiny` systems preset:

```text
steps             2
split point       1
training examples 64
batch size        8
evaluation count  96 per shift
device            CPU
```

For each arm compare uninterrupted two-step training with a one-step checkpoint
followed by reload and the second step. Require exact tensor-byte equality of
the final system and optimizer states, identical final loss, finite gradients,
nonzero parameter change, and identical evaluation posteriors.

## CUDA analytic shakedown

If CPU lifecycle passes, run only the analytic positive-control arm on d6:

```text
steps              64
split point        32
training examples 512
batch size         64
evaluation count  512 per shift
device             CUDA
```

Require the same exact-resume and finiteness gates, checkpoint reload, and an
unchanged analytic carrier contract. Metrics are diagnostic anchors for the
later numeric preregistration, not quality evidence. No raw or learned d6
primary cell may run in Stage 0.

## Contract gates

Stage 0 jointly requires:

1. the pinned preflight still classifies the generator as
   `c3_temporal_quotient_preflight_passed`;
2. exact token group laws, target invariance, no saturation, paired-target
   equality, distinct deck sheets, and a valid target-changing derangement;
3. the raw feature changes under a nonidentity deck action while the analytic
   and learned features remain invariant below `1e-5`;
4. learned invariance passes at initialization, a deterministic perturbed
   parameter state, and after its CPU lifecycle optimization;
5. data and minibatch hashes match across arms;
6. TinyLLM initialization matches across arms and the structured injection
   initialization matches between analytic and learned arms;
7. exact parameter counts are reported by TinyLLM, injection, and learned
   encoder;
8. all CPU lifecycle cells pass exact resume;
9. the analytic CUDA shakedown passes exact resume; and
10. every JSON value is finite and every saved artifact reloads with matching
    provenance.

The only passing classification is:

```text
c3_temporal_quotient_stage0_passed
```

Any failed contract is:

```text
c3_temporal_quotient_stage0_invalid
```

After a pass, freeze a separate dated primary preregistration containing the
numeric task, representation, causal, replay, and specificity gates. Stage-1
training remains unauthorized until that document exists and is pinned by the
campaign runner.
