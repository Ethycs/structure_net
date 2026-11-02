# TinyLLM C3 relational connection preflight preregistration

**Status:** FROZEN BEFORE FIVE-SEED PRIMARY GENERATION

**Date:** 2026-08-11

**Conformance:** NAL-STD-EXPERIMENT `PROSPECTIVE NO-TRAINING IDENTIFIABILITY AND FIXED-CEILING CONTRACT`

**Hypothesis:** `tinyllm-c3-relational-connection-preflight-v1`

## Decision question

The predecessor audit proved that the old temporal target factors through a
pointwise cubic carrier and is therefore invariant under the complete local
group `C3^8`. A connection cannot earn task rent in that scope.

This preflight changes the task rather than adding optimization:

> When the target compares two charged endpoint states and no sufficient
> pointwise invariant exists, does an observed discrete edge connection make
> the relation identifiable and accurately computable while connection-free
> and wrong-connection controls fail?

This is a fresh five-seed analytic experiment. It loads no checkpoint,
instantiates no TinyLLM, and performs no fit or optimizer step.

## Observation and target contract

For eight frames, draw endpoint phase `theta_0`, six independent interior
phases, and

```text
delta ~ Uniform[-pi, pi]
theta_7 = theta_0 + delta.
```

The interior phases are independent of the endpoint relation. This prevents a
smooth-path prior from unwrapping the endpoint relation through pointwise
cubic increments.

The calibrated continuous three-channel observation is

```text
y_(t,c) = A cos(theta_t + 2*pi*c/3) + offset + drift*t,
```

with amplitude, offset, and drift observed in the calibration packet. Use the
existing 1,024-bin quantizer and the existing composition/extrapolation
nuisance ranges. The target is

```text
T = cos(theta_7 - theta_0) = cos(delta).
```

Draw an independent local sensor frame `g_t in C3` at every time, roll the
three channel tokens by `g_t`, and expose only the edge connection

```text
a_t = g_(t+1) - g_t  (mod 3),  t=0,...,6.
```

The connection contains no phase, delta, or target label. It is sensor-frame
metadata computed only from the independently sampled gauges.

## Identifiability and action law

Let `c_t` be the normalized charged first character of the calibrated
observation. Under the local frame,

```text
c_t = exp(i theta_t) exp(-2*pi*i*g_t/3).
```

Let `A_07 = sum_t a_t = g_7 - g_0 (mod 3)`. Then

```text
Re[c_7 exp(+2*pi*i*A_07/3) conjugate(c_0)]
  = cos(theta_7 - theta_0).
```

Thus the continuous calibrated observation plus the edge connection
identifies the target exactly. Quantization is a measurement approximation,
not part of the symbolic equality; its adequacy is separately governed by the
fixed positive-control gates below.

For an additional local action `h_t`, transform

```text
c_t -> exp(-2*pi*i*h_t/3)c_t
a_t -> a_t + h_(t+1) - h_t  (mod 3).
```

The analytic prediction must remain unchanged. Identity, inverse,
composition, and order-three token/connection laws must be exact.

## Explicit insufficiency witness

Freeze a continuous pair with equal amplitude/calibration:

```text
state A: theta_7 = theta_0,         g_7 = g_0
state B: theta'_7 = theta_7+2*pi/3, g'_7 = g_7+1.
```

Keep every earlier physical phase and local frame fixed. The charged endpoint
observation, all pointwise cubes, and the complete token tensor are identical,
but the targets are `1` and `-1/2`. The last edge connection differs by one.

This witnesses that charged tokens without the connection, and therefore the
pointwise cubic sequence, cannot determine the relational target. Under the
uniform `delta` law, the three compatible roots are equiprobable and

```text
E[cos(delta) | exp(i*3*delta)] = 0.
```

The zero function is therefore the declared pointwise-invariant Bayes
baseline; a principal-cube-root chart is retained as a second descriptive
failure control.

## Frozen fresh population

```text
seeds:   1213, 1231, 1277, 1301, 1321
shifts:  composition, extrapolation
count:   4,096 examples per seed/shift
```

Primary dataset stream bases are `1,121,107` for composition and `1,123,107`
for extrapolation. Connection-shuffle bases are `1,125,107` and `1,127,107`.
Local-action bases are `1,129,107` and `1,131,107`; second-action bases are
`1,133,107` and `1,135,107`. Target-shuffle bases are `1,137,107` and
`1,139,107`. No pilot examples or streams may enter the primary result.

## Frozen arms

Evaluate the following without fitted parameters:

1. exact continuous connection oracle, validity only;
2. calibrated quantized charged character plus the observed connection;
3. the same charged endpoints with no connection;
4. the observed connection applied with the wrong sign;
5. a Sattolo-deranged connection from another example;
6. the principal cube-root chart of the pointwise-invariant endpoint ratio;
7. the exact pointwise-invariant Bayes predictor, zero.

Also score the analytic observed-connection prediction against a separately
Sattolo-deranged target.

## Frozen per-cell gates

The observed-connection arm passes a cell only if all hold:

```text
scalar correlation             >= .999
scalar RMSE                    <= .01
exact-bin accuracy             >= .98
target cross-entropy           <= 1.35
predicted-bin coverage         == 16
local-action prediction error  <= 2e-12
```

Each no-connection, wrong-sign, shuffled-connection, and principal-root
control passes its failure gate only if:

```text
absolute scalar correlation <= .10
scalar RMSE                 >= .80
complete task gate          == false
```

The pointwise Bayes-zero control must have scalar RMSE in `[.65,.76]`, exact
accuracy `<=.10`, one predicted bin, and fail the complete task gate. The
shuffled-target control must have absolute correlation `<=.10` and RMSE
`>=.80`.

Every primary cell must additionally have:

- deterministic dataset and control streams;
- zero quantizer saturation;
- all sixteen target bins represented and predicted by the positive control;
- charged-character magnitude `>=.25`;
- continuous-oracle error `<=1e-12`;
- exact token and edge-connection identity/inverse/composition/order-three
  laws;
- exact observed-pair covariance and local-action prediction invariance;
- a passing explicit collision witness;
- zero fits, checkpoints, models, parameters changed, and optimizer steps.

The population requirement is a joint pass on both shifts in at least four of
five seeds. Do not combine different seeds for different endpoints.

## Locked classifications

| Outcome | Classification | Decision |
| --- | --- | --- |
| observed connection, specificity controls, and pointwise-insufficiency gates each pass `>=4/5` joint seeds | `observed_edge_connection_identifies_nonpointwise_c3_relation` | license one no-training connection-conditioned function-class/lifecycle preflight; do not yet train |
| observed connection passes but a connection-free or pointwise arm also passes `>=4/5` | `relational_scope_has_connection_free_shortcut` | close connection learning and identify the shortcut |
| continuous oracle passes but quantized observed connection is `<4/5` | `relational_connection_quantization_ceiling_failed` | repair acquisition/precision only |
| observed connection passes but wrong/shuffled controls do not fail | `relational_connection_not_causally_specific` | redesign the intervention; do not train |
| any lineage, action, determinism, saturation, collision, or accounting contract fails | `invalid_relational_connection_preflight` | infrastructure repair only |

`unrestricted_tinyllm_training_licensed=false` and
`matched_training_directly_licensed=false` in every row. Only the first row
licenses the cheaper function-class and lifecycle prerequisite.

## Scientific boundary

A positive result establishes an identifiable, non-pointwise relational task
for which connection data are causally necessary and a fixed analytic ceiling
exists. It does not establish that TinyLLM can learn the connection law, that
learning is cheaper than using the fixed solution, or that the result extends
to unknown, noisy, or continuous gauge groups.

## Frozen source lineage

| Source | SHA-256 |
| --- | --- |
| C3 generator and quantizer | `812ac0a574bab812ff9fbbea7a0997c71e8ed0c50f98b270b8344234956ec118` |
| interval task likelihood | `b027eb55d971b87f598bd600131b33ead82463e3e73766fcdce38fdc3a5497c6` |
| full local-gauge closure preregistration | `0f620df51c2f1d4278a8bbd82f8a6071c3159bdeb5e9420f4e08722c41115f90` |
| full local-gauge closure runner | `dae960759a2412451f8b15e15b0b6fb479603938a323ea54c86c77c68839005d` |
| full local-gauge closure result | `31f7c7301c889db67e436fba4b8de1909dfbc372573681a9950b5b330f22db35` |
| full local-gauge closure report | `e1a237410bd3dfb050d22aadbfdf5b68f8b19a83c2977ae483fdebdbe468718e` |

## Expected artifact

```text
data/experiments/tinyllm_c3_relational_connection_preflight/
  20260811_preregistered/result.json
```
