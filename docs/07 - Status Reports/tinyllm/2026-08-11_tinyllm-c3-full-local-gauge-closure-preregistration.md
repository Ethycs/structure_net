# TinyLLM C3 full local-gauge closure audit preregistration

**Status:** FROZEN BEFORE GENERATOR-ACTION AUDIT

**Date:** 2026-08-11

**Conformance:** NAL-STD-EXPERIMENT `ARTIFACT-LINEAGE NO-TRAINING CAUSAL CONTRACT`

**Hypothesis:** `tinyllm-c3-local-gauge-invariant-closure-v1`

## Decision question

The hidden suffix-jump studies establish that a coherent charged trajectory
requires a connection, but they also show fixed invariant task closure in
`5/5` seeds. Before adding more jumps or training a connection model, decide
the stronger algebraic scope:

> Does the pointwise cubic carrier make the complete frozen physical task
> invariant under the full local gauge group `C3^8`, thereby closing every
> multiple-jump extension of the same observation/target law?

This is a causal group-action audit on the ten sealed fresh cells from
`tinyllm-c3-gauge-jump-joint-typed-score-v1`. It adds no examples and makes no
new population-performance claim.

## Algebraic contract

Let `c_t` be the calibrated charged first character and let

```text
g = (g_0,...,g_7) in C3^8
```

act independently at every observed frame:

```text
(g c)_t = exp(-2*pi*i*g_t/3) c_t.
```

The pointwise quotient is

```text
q_t = c_t^3.
```

Because every cube root of unity has cube one,

```text
q(g c)_t = exp(-2*pi*i*g_t) c_t^3 = q(c)_t.
```

Therefore every computation of the form `D(q(c), calibration)` is invariant
under all of `C3^8`; no temporal connection appears in that factorization.
The sixteen single-frame nonidentity actions generate the complete finite
group, so exact closure on those generators plus the group law certifies all
`3^8 = 6,561` local actions.

Any sequence of suffix jumps is an element of this local group. Thus adding
more hidden jumps cannot create task-level model rent while the target and
fixed decoder continue to factor through `q`.

## Frozen source population

Reuse, without pooling into a new task-performance estimate, the ten cells:

```text
seeds:   773, 821, 1003, 1031, 1039
shifts:  composition, extrapolation
count:   4,096 examples per cell
```

Regenerate them only from the source runner's frozen streams and require exact
dataset/corruption hashes to match its primary artifact.

For each corrupted token tensor, evaluate:

1. the identity observation;
2. each of the sixteen generators `(time=0..7, element=1 or 2)`;
3. one deterministic arbitrary per-example local action with stream bases
   `1103107 + seed` for composition and `1105107 + seed` for extrapolation;
4. a second action for composition-law checks with stream bases
   `1107107 + seed` and `1109107 + seed`;
5. inverse, composition, and order-three token contracts for those deterministic
   arbitrary local actions.

The lifecycle test may exhaust all 6,561 action vectors on a small subset from
the already-declared seed-1117 pilot. Lifecycle outcomes are not evidence.

## Frozen measurements

For the identity and every transformed observation, compute only the existing
fixed invariant switch/deletion decoder:

```text
tokens -> calibrated first character -> pointwise cube
       -> invariant 24-way switch/deletion selector -> physical forecast.
```

Record per cell:

- exact Eisenstein pointwise-cube integer mismatch count;
- analytic cubic-carrier maximum error;
- invariant forecast maximum error;
- selector-change count, descriptive only;
- exact token inverse/composition/order-three error counts;
- inherited source fixed-ceiling status and source-result identity.

No charged connection decoder, target fit, learned map, checkpoint, or
optimizer is needed.

## Gates

The audit passes only if:

- all ten source dataset and corruption hashes match exactly;
- every source cell remains valid and its inherited fixed invariant arm passes;
- exact pointwise-cube mismatch count is zero for every generator and arbitrary
  local action;
- all token identity, inverse, composition, and order-three errors are zero;
- analytic cubic-carrier and invariant forecast errors are each `<=2e-12`;
- the source fixed invariant joint seed count remains `5/5`;
- accounting reports zero fresh examples, models, checkpoints, optimizer
  steps, changed parameters, reusable fits, and target-using fits.

Selector identities are not an endpoint because tied physical charts may
produce the same invariant forecast.

## Locked classifications

| Outcome | Classification | Decision |
| --- | --- | --- |
| every algebraic/numeric gate passes and inherited invariant closure is `>=4/5` | `pointwise_cubic_quotient_closes_full_local_c3_gauge` | close all same-law multiple-jump and connection-learning studies |
| exact integer closure passes but carrier or forecast tolerance fails | `algebraic_local_gauge_closure_numeric_implementation_defect` | repair arithmetic only |
| inherited invariant closure is `<4/5` | `local_gauge_invariant_but_task_decoder_insufficient` | change the physical decoder; do not learn a connection |
| any lineage or group-law contract fails | `invalid_full_local_c3_gauge_audit` | infrastructure repair only |

`multiple_jump_experiment_licensed=false`,
`compact_connection_model_licensed=false`, and
`tinyllm_training_licensed=false` in every valid row.

## Scientific boundary

A positive result does **not** say connections are meaningless. It says they
are unnecessary for this target because a sufficient pointwise invariant is
observed. A legitimate connection-learning task must remove that factorization
while preserving identifiability—for example, a gauge-invariant relational or
holonomy target with an observed/partially observed edge connection, an
unknown group action, or observations from which no sufficient pointwise
invariant can be formed.

## Frozen source lineage

| Source | SHA-256 |
| --- | --- |
| joint typed-score preregistration | `e4629a11cac991b1bd64d641f3276b4517296ee31ac3b9e0a3837e5cb5ce4663` |
| joint typed-score runner | `6a9b1b849c97fc30bef7292ed0bbc097c4829db2920d3abe8f698e7546489944` |
| joint typed-score result | `f52ce2103a07086a7118975d69f49b7cbeca01ac0ca7c5a15fd6d2a96fbc51fa` |
| joint typed-score report | `281f375c069fb58b9949b7e2d0c98c895e4bfedf8a9f4e7160204e2dc3bb852b` |

## Expected artifact

```text
data/experiments/tinyllm_c3_full_local_gauge_closure/
  20260811_artifact_audit/result.json
```
