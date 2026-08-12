# TinyLLM C3 relational connection preflight result

**Status:** VALID PROSPECTIVE NO-TRAINING RESULT — OBSERVED CONNECTION IS NECESSARY AND SUFFICIENT

**Date:** 2026-08-11

**Hypothesis:** `tinyllm-c3-relational-connection-preflight-v1`

**Classification:** `observed_edge_connection_identifies_nonpointwise_c3_relation`

**Preregistration:** [relational connection preflight](../07%20-%20Status%20Reports/2026-08-11_tinyllm-c3-relational-connection-preflight-preregistration.md)

**Primary artifact:** `data/experiments/tinyllm_c3_relational_connection_preflight/20260811_preregistered/result.json`

## Verdict

The first non-pointwise `C3` relational scope passes every preregistered
analytic gate. An observed edge connection reconstructs the physical endpoint
relation in both nuisance regimes in all five fresh seeds. Removing the
connection, applying it with the wrong sign, shuffling it between examples, or
using only the pointwise cubic invariant destroys the relation.

```text
observed connection:         5/5 joint seeds
connection specificity:     5/5 joint seeds
pointwise insufficiency:     5/5 joint seeds
connection-free shortcut:   0/5 joint seeds
fresh examples / fits:       40,960 / 0
TinyLLM / optimizer steps:   0 / 0
```

This is the missing positive answer behind the earlier negative gauge-jump
studies: a connection pays task rent when the target compares charged states
and no sufficient pointwise invariant exists. It did not pay rent for the old
forecast target because that target already factored through `c_t^3` at every
frame.

## What changed scientifically

The predecessor target used the pointwise invariant carrier

```text
q_t = c_t^3.
```

That made the entire task insensitive to all 6,561 elements of `C3^8`. The new
target is instead

```text
T = cos(theta_7 - theta_0).
```

Interior phases are independently drawn, so they cannot supply a smoothness
prior that unwraps the endpoint relation. The local sensor frame is observed
only through edge differences

```text
a_t = g_(t+1) - g_t  (mod 3).
```

The accumulated connection transports the final charged character into the
initial frame:

```text
Re[c_7 exp(+2*pi*i*sum(a)/3) conjugate(c_0)]
  = cos(theta_7 - theta_0).
```

The connection is therefore not an auxiliary feature correlated with the
answer. It supplies the missing comparison convention between two local
frames.

## Primary results

Means over five independent `4,096`-example cohorts per shift:

| Shift | Arm | Scalar RMSE | Minimum corr | Exact-bin acc | Cross-entropy |
| --- | --- | ---: | ---: | ---: | ---: |
| composition | observed connection | `.001653` | `.9999971` | `.99053` | `1.28235` |
| extrapolation | observed connection | `.001754` | `.9999967` | `.99028` | `1.27929` |
| composition | no connection | `1.00481` | — | — | — |
| extrapolation | no connection | `.99938` | — | — | — |
| composition | wrong-sign connection | `1.00218` | — | — | — |
| extrapolation | wrong-sign connection | `1.00361` | — | — | — |
| composition | shuffled connection | `.99864` | — | — | — |
| extrapolation | shuffled connection | `.99680` | — | — | — |
| composition | principal pointwise chart | `1.09131` | — | — | — |
| extrapolation | principal pointwise chart | `1.10082` | — | — | — |

Every observed-connection cell predicts all sixteen bins. Its worst scalar
RMSE is `.001814`, its worst exact-bin accuracy is `.98950`, and its maximum
cross-entropy is `1.28644`; all are well inside the frozen `.01`, `.98`, and
`1.35` limits.

The no-connection absolute correlation never exceeds `.0308`. Wrong-sign,
shuffled-connection, principal-root, and shuffled-target controls all meet
their registered failure criteria in all ten cells.

## Exact identifiability witness

Two states in the declared observation family were constructed with:

```text
state A: theta_7 = theta_0,           g_7 = g_0
state B: theta'_7 = theta_7 + 2*pi/3, g'_7 = g_7 + 1.
```

They have:

| Measurement | Result |
| --- | ---: |
| continuous charged-observation error | `0` |
| quantized token mismatches | `0` |
| pointwise-cube error | `0` |
| differing connection edges | `1` |
| target separation | `1.5` |

Thus no decoder of the charged tokens alone—and no decoder of their pointwise
cubes—can be exact. Under the uniform endpoint-difference law, the three cube
roots are equiprobable and their cosine sum is zero. The registered
pointwise-invariant Bayes predictor is consequently zero; its measured RMSE is
`.6986-.7135`, it predicts one bin, and it passes no task cell.

This is an exact observation-equivalence argument, not a failure inferred
from a weak learned baseline.

## Gauge covariance and integrity

An independent local action was applied to every example and its observed
connection transformed covariantly. Across all ten cells:

| Contract | Result |
| --- | ---: |
| maximum continuous-oracle error | `1.665e-15` |
| maximum charged-character covariance error | `1.099e-15` |
| maximum transported-prediction error | `1.332e-15` |
| token/connection group-law errors | `0` |
| quantizer saturation | `0` |
| deterministic datasets and controls | `10/10` |
| target-bin coverage | `16/16` in every cell |
| checkpoints / models / fitted parameters | `0 / 0 / 0` |

The symbolic identifiability statement belongs to the continuous calibrated
observation. Finite token quantization is an approximation; the ten-cell
fixed ceiling establishes that its error is materially below the declared
task tolerance rather than silently calling it exact.

## Program decision

This result licenses exactly one cheaper prerequisite:

```text
connection-conditioned function-class and lifecycle preflight: licensed
matched training directly:                                  closed
unrestricted TinyLLM training:                              closed
```

The next test should construct the smallest architecture whose allowed
functions respect the observed-pair action

```text
(c, a) -> (h.c, a + dh)
```

and verify, without primary training, that:

1. a closed-form parameter state exactly implements the analytic transported
   endpoint relation;
2. the output remains invariant for every parameter state by construction;
3. a fresh initialization receives a finite downhill true-task gradient;
4. connection-shuffled and target-shuffled controls remain distinct;
5. checkpoint/restart and device lifecycle contracts are exact.

Only that positive prerequisite could license a matched learned acquisition
study. The fixed analytic connection remains the positive-control mechanism
and the cheaper deployed solution for this known group law.

## Scope boundary

The result establishes necessity and sufficiency of an exact observed
discrete connection for one calibrated `C3` endpoint relation. It does not
establish:

- learned connection use;
- TinyLLM utility;
- robustness to missing, noisy, or partially observed edges;
- recovery of an unknown group action;
- continuous-gauge or nonabelian transport;
- superiority over the fixed analytic decoder.

It also does not reverse the full local-gauge closure result. The two outcomes
have different task factorizations and are jointly the useful conclusion:

```text
pointwise-invariant target  -> connection cannot add task information
charged relational target  -> observed connection supplies the missing frame comparison.
```

## Reproduction and provenance

```bash
MPLCONFIGDIR=/tmp/mpl-c3-relational-primary \
pixi run python -m \
  experiments.structure_net.tinyllm_c3_relational_connection_preflight
```

| Artifact | SHA-256 |
| --- | --- |
| primary result | `ae0321b3c3d5b7e8c80ebbd57bfaea10c0522c4b2241082daab007a54e55984e` |
| runner | `2ab21b7885fa93c49427973f67e5c57913b7f13c154ad2e970ef9636b2b90214` |
| preregistration | `702196e0eaaaacda90293d202859e258889e1bed553dea4e2f7986f1ac8a57cc` |
| predecessor result | `31f7c7301c889db67e436fba4b8de1909dfbc372573681a9950b5b330f22db35` |
