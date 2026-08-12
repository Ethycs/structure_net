# TinyLLM C3 relational connection acquisition preregistration

**Status:** FROZEN PROSPECTIVE FIVE-SEED ACQUISITION PROTOCOL

**Date:** 2026-08-11

**Hypothesis:** `tinyllm-c3-relational-connection-acquisition-v1`

**Evidence role:** matched learned acquisition after exact identifiability and
function-class prerequisites

## Decision being tested

The sealed predecessor establishes three facts without optimization:

1. the observed `C3` edge connection is necessary and sufficient for the
   non-pointwise endpoint target;
2. a 187-parameter connection-invariant module contains the analytic solution;
3. the true task supplies a finite, control-specific gradient to that module.

This campaign asks the remaining narrow question:

> Does ordinary matched gradient training acquire the identified connection-
> conditioned relation reliably from random initialization?

The campaign is not a TinyLLM training study. It instantiates no transformer
and cannot establish superiority over the fixed analytic decoder.

## Locked predecessors

| Source | SHA-256 |
| --- | --- |
| relational preflight runner | `2ab21b7885fa93c49427973f67e5c57913b7f13c154ad2e970ef9636b2b90214` |
| relational preflight result | `ae0321b3c3d5b7e8c80ebbd57bfaea10c0522c4b2241082daab007a54e55984e` |
| function-class runner | `7010794c3be5fda05a035e5a0b4a178aacd40c934dc1f5f7b36eb2eb03ea96b1` |
| function-class result | `2292e971bb655db246565675fece8dfd9e1546692b9b782e940a8bbef49de82c` |
| function-class report | `66641c7f4b90328f775d525ec2fbe2166cdbc6dd33ed3832e45ef3fd74bf638a` |

The primary runner must reject changed bytes or a predecessor classification
other than
`connection_invariant_function_class_contains_transport_and_task_gradient`.

## Pilot disclosure and frozen optimization choice

No primary seed or stream was inspected while selecting the optimizer.
Three excluded pilot seeds `(1409, 1451, 1499)` used training stream
`1_150_107 + seed`, held-out streams `1_152_107 + seed` and
`1_154_107 + seed`, and batch stream `1_156_107 + seed`.

The bounded pilot compared:

- interval-posterior cross-entropy, AdamW `3e-4`, weight decay `.01`, through
  1,200 steps;
- scalar MSE, AdamW `3e-4`, weight decay `.01`, through 2,400 steps;
- scalar MSE, AdamW `1e-3`, zero weight decay, through 2,400 steps.

Only the last protocol acquired the relation in all three pilot seeds. At step
2,400, across both shifts, its worst scalar correlation was `.9999966`, worst
RMSE `.003051`, minimum exact-bin accuracy `.96973`, maximum target
cross-entropy `1.29213`, and minimum predicted-bin coverage `16`.

Those observations freeze the primary choice. There will be no learning-rate,
loss, weight-decay, step-count, warm-start, seed, or threshold sweep after
primary outcomes.

## Population and streams

Primary seeds are:

```text
(1453, 1471, 1483, 1531, 1543)
```

They are disjoint from every predecessor and pilot seed. For replicate seed
`s`, the frozen stream seeds are:

| Material | Seed |
| --- | ---: |
| training composition cohort | `1_151_107 + s` |
| held-out composition cohort | `1_153_107 + s` |
| held-out extrapolation cohort | `1_155_107 + s` |
| minibatch schedule | `1_157_107 + s` |
| training connection permutation | `1_159_107 + s` |
| composition connection permutation | `1_161_107 + s` |
| extrapolation connection permutation | `1_163_107 + s` |
| training target permutation | `1_165_107 + s` |
| local-action audit | `1_167_107 + s` |

Every cohort is regenerated deterministically and hashed before optimization.
The train cohort has 4,096 examples; each held-out cohort has 1,024. Training
uses the composition nuisance law only. Composition and extrapolation
evaluation examples are never used for optimization or model selection.

## Arms

All four learned arms share exact initial parameters, training examples,
minibatches, optimizer, and step count within a seed:

| Arm | Training input | Training target | Evaluation input |
| --- | --- | --- | --- |
| `learned_true` | observed connection | true cosine | observed connection |
| `learned_no_connection` | all-zero connection | true cosine | all-zero connection |
| `learned_connection_shuffled` | fixed derangement of connection rows | true cosine | independently deranged connection rows |
| `learned_target_shuffled` | observed connection | fixed derangement of targets | observed connection and true target |

The fifth arm is the zero-fit analytic transport solution. It is evaluated
before learned scheduling and is the positive-control stop gate.

All permutations are fixed-point-free Sattolo derangements. Protocol validity
requires nonzero target and connection changes, identical initial state hashes,
identical batch hashes, zero quantizer saturation, and complete target-bin
coverage.

## Module and optimizer

The learned module is exactly the sealed 187-parameter function class:

```text
shared 1->16->8 GELU channel map
  -> charge-one C3 character
  -> learned complex mixing and normalization
  -> exact observed-connection transport
  -> neutral endpoint product
  -> 2->1 scalar head
```

There is no raw bypass. The frozen optimizer is:

| Setting | Value |
| --- | ---: |
| optimizer | AdamW |
| objective | scalar MSE against `cos(theta_7-theta_0)` |
| learning rate | `1e-3` |
| weight decay | `0` |
| gradient clip | `1.0` |
| batch size | `64` |
| steps | `2,400` |
| midpoint | `1,200` |

## Lifecycle and symmetry contract

Before the primary campaign, separate two-step CPU and CUDA shakedowns must
pass. They are systems evidence only.

For every primary learned arm:

1. save a weights-plus-optimizer checkpoint at step 1,200;
2. save a final checkpoint at step 2,400;
3. reload the final checkpoint with exact state, optimizer, and prediction
   replay;
4. resume from step 1,200 and reproduce the second-half history, final state,
   optimizer, and predictions exactly;
5. audit the observed-pair local action at initialization, midpoint, and final
   state, requiring maximum output error `<=2e-5` at every cut.

The local action transforms both charged tokens and their edge connection.
For the no-connection and shuffled controls, the transformed connection is the
coboundary update of that arm's supplied base connection; the test remains an
architectural identity rather than a data-semantics claim.

Initial, midpoint, and final winding number plus minimum raw charged magnitude
on a fixed ideal phase circle are recorded as exploratory optimization
diagnostics. They do not change any gate.

## Gates

### Analytic stop gate

The fixed analytic arm must pass both held-out shifts in at least four of five
fresh seeds using the sealed ceiling thresholds:

```text
scalar correlation       >= .999
scalar RMSE              <= .01
exact-bin accuracy       >= .98
target cross-entropy     <= 1.35
predicted-bin coverage    = 16
```

If it fails, the campaign stops with zero learned optimizer steps.

### Learned joint endpoint

A learned arm passes a seed only when it simultaneously passes composition and
extrapolation:

```text
scalar correlation       >= .999
scalar RMSE              <= .01
exact-bin accuracy       >= .95
target cross-entropy     <= 1.35
predicted-bin coverage    = 16
maximum action error     <= 2e-5 at initialization, midpoint, and final
exact final reload        = pass
exact midpoint resume     = pass
```

The primary hypothesis passes only if:

```text
learned_true joint passes                  >= 4/5
learned_no_connection joint passes         <= 1/5
learned_connection_shuffled joint passes   <= 1/5
learned_target_shuffled joint passes       <= 1/5
```

All twenty learned cells must be finite and lifecycle-valid. Control arms are
judged by the same positive endpoint; they are not granted weaker bespoke
failure thresholds.

## Locked classifications

| Outcome | Classification |
| --- | --- |
| analytic stop gate fails | `analytic_connection_ceiling_failed_on_fresh_streams` |
| any provenance, protocol, finiteness, replay, or symmetry contract fails | `invalid_connection_acquisition_campaign` |
| any control exceeds `1/5` | `connection_acquisition_specificity_failed` |
| true arm reaches `>=4/5` and every control remains `<=1/5` | `connection_invariant_relation_acquired_by_gradient_training` |
| valid controls but true arm reaches `<4/5` | `exact_function_class_but_population_acquisition_unreliable` |

Every classification leaves unrestricted TinyLLM training closed. A positive
result would establish acquisition in a compact typed module, not incremental
utility over the six-weight analytic solution. A negative result would close
optimizer tuning for this function class and motivate an architectural
initialization or fixed analytic transport, not a larger transformer.

## Accounting and preservation

The result must report primary and resume-verification optimizer steps
separately and assert that zero TinyLLM models were instantiated. Per-seed
results, midpoint/final checkpoints, the campaign bundle, report, and
meta-hypothesis record must be stored under `data/`, covered by DVC, pushed to
the configured remote, and committed to the registered lakeFS branch before
the result is treated as preserved.
