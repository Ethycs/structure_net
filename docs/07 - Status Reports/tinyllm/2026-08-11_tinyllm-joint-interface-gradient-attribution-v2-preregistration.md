# TinyLLM Joint-Interface Gradient Attribution v2 Preregistration

**Status:** FROZEN NUMERICAL-VALIDITY CORRECTION BEFORE V2 RUN

**Date:** 2026-08-11

**Conformance:** NAL-STD-EXPERIMENT `REGISTERED POST-OUTCOME CORRECTIVE`

**Hypothesis:** `tinyllm-joint-interface-gradient-attribution-v2`

**Supersedes for inference:** [v1 gradient attribution preregistration](2026-08-11_tinyllm-joint-interface-gradient-attribution-preregistration.md)

## Why v1 is invalid

The complete v1 computation produced all 20 result files with no worker
failure. Five cells failed only the registered float32 gradient-additivity
tolerance: their maximum absolute residual was `1.1444091796875e-5` against a
ceiling of `1e-5`. All state-identity, finite-array, exact-reload, source, and
schedule checks passed.

Gradient additivity is an algebraic identity. The observed residual is roughly
float32 roundoff on gradient vectors with norms reaching hundreds; it is not a
scientific discrepancy. Nevertheless, v1 remains invalid and its root is
preserved unchanged.

## Sole correction

V2 changes only the numerical validity check:

```text
maximum absolute additivity error <= 2e-5
maximum relative additivity error <= 1e-6
```

For each snapshot, relative error is the maximum absolute additivity error
divided by `max(1, global gradient norm)`. Both conditions are required.

No source, state, batch, objective, parameter block, gradient vector,
scientific threshold, population gate, classification rule, or interpretation
changes. The initial-starvation and persistent-conflict gates remain exactly:

- both first/last initial-state batches require encoder block clip `1.0`,
  global clip `<= 0.10`, cross-block suppression `<= 0.10`, and nonzero sensor
  gradient;
- both first/last final-state batches require nonzero sensor gradient and
  sensor descent ratio `<= 0`;
- each population gate requires `4/5` separately in d6 and d10 learned cells.

The v1 scientific gate values have been partially exposed by worker summaries,
so v2 is a corrective numerical replay, not fresh confirmation. Its legitimate
claim is limited to validating and classifying the already registered local
gradient mechanism.

## Artifact root

`data/experiments/tinyllm_joint_interface_gradient_attribution/20260811_d6_d10_registered_v2`
