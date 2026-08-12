# TinyLLM C3 full local-gauge closure audit

**Status:** VALID ARTIFACT-LINEAGE CAUSAL RESULT — FULL `C3^8` TASK CLOSURE

**Date:** 2026-08-11

**Hypothesis:** `tinyllm-c3-local-gauge-invariant-closure-v1`

**Classification:** `pointwise_cubic_quotient_closes_full_local_c3_gauge`

**Preregistration:** [full local-gauge closure](../07%20-%20Status%20Reports/2026-08-11_tinyllm-c3-full-local-gauge-closure-preregistration.md)

**Primary artifact:** `data/experiments/tinyllm_c3_full_local_gauge_closure/20260811_artifact_audit/result.json`

## Verdict

The same-law multiple-gauge-jump branch is closed without another experiment
or model. The pointwise cubic carrier makes the complete fixed invariant task
computation insensitive to the full framewise group

```text
C3^8, with |C3^8| = 6,561.
```

All sixteen nonidentity single-frame generators pass exact integer and numeric
closure in every one of the ten sealed source cells. Exact composition,
inverse, and order-three token laws pass; one arbitrary per-example local
action also passes per cell. Because those generators span the finite group,
the result certifies every framewise gauge assignment, including any sequence
of hidden suffix jumps.

```text
inherited fixed invariant task closure:  5/5
exact local-gauge closure:               5/5
numeric local-gauge closure:             5/5
fresh examples / models / optimizer:     0 / 0 / 0
```

Multiple-jump experiments, a compact connection model, and TinyLLM training
are all unlicensed under the same observation and target law.

## Proof object

Let the calibrated charged first character be `c_t`, and let an arbitrary
local gauge `g=(g_0,...,g_7)` act by

```text
(g c)_t = exp(-2*pi*i*g_t/3) c_t.
```

The existing sufficient carrier is pointwise:

```text
q_t = c_t^3.
```

Therefore

```text
q(g c)_t
  = exp(-2*pi*i*g_t) c_t^3
  = c_t^3
  = q(c)_t.
```

The frozen invariant decoder is a function only of `q` and unchanged
calibration. Consequently its forecast is constant on every `C3^8` orbit. A
temporal connection is mathematically absent from this factorization.

The numerical audit verifies the implementation rather than substituting for
the proof. It checks both nonidentity elements at each of eight frames, plus
arbitrary per-example actions. A separate lifecycle enumerates all 6,561
action vectors on pilot observations and finds zero exact integer-cube errors.

## Measured contracts

| Measurement | Result |
| --- | ---: |
| source cells / exact hash replays | `10 / 10` |
| generator actions per cell | `16` |
| generator counterfactual evaluations | `655,360` |
| arbitrary local counterfactual evaluations | `40,960` |
| exact pointwise-cube integer errors | `0` |
| token group-law errors | `0` |
| maximum analytic cubic-carrier error | `6.133e-15` |
| maximum invariant forecast error | `1.692e-12` |
| maximum stabilization displacement | `7.036e-13` |
| selector identity changes | `42` |
| source examples reused / fresh | `40,960 / 0` |
| reusable or target-using fits | `0 / 0` |
| checkpoints / models / optimizer steps | `0 / 0 / 0` |

The 42 selector changes are intentionally not failures. Floating ties can
choose different switch/deletion labels while the invariant forecast remains
within `1.692e-12`. This is another concrete example of why latent label
recovery is weaker than the causal endpoint.

## What this changes

The one-jump charged experiment correctly showed that a charged trajectory
needs a connection: without one, charged forecast RMSE was about `.87`. That
does not imply the task needs a connection. The task already descends through
the framewise invariant `q`, whose fixed decoder passes `5/5`.

Thus the appropriate statement is:

```text
connection required for coherent charged coordinates
                         !=
connection required for an invariant task computation.
```

Adding more hidden jumps would increase the nuisance description while leaving
the sufficient task statistic exactly unchanged. Such an experiment cannot
make a learned model pay rent.

## Next legitimate scope

A connection-learning study must remove pointwise invariant sufficiency while
keeping the target identifiable. Candidate scopes include:

- a gauge-invariant relational or holonomy target with an observed or
  partially observed edge connection;
- an unknown group action whose invariant/charged decomposition is not supplied;
- partial observations from which no sufficient pointwise invariant exists;
- dynamics whose task-relevant transport cannot be enumerated analytically.

Each requires a new observation-equivalence proof and an oracle/fixed ceiling
before training. “More jumps” under the present target does not.

## Reproduction and provenance

```bash
MPLCONFIGDIR=/tmp/mpl-local-gauge-primary \
pixi run python -m \
  experiments.structure_net.tinyllm_c3_full_local_gauge_closure
```

| Artifact | SHA-256 |
| --- | --- |
| result | `31f7c7301c889db67e436fba4b8de1909dfbc372573681a9950b5b330f22db35` |
| runner | `dae960759a2412451f8b15e15b0b6fb479603938a323ea54c86c77c68839005d` |
| preregistration | `0f620df51c2f1d4278a8bbd82f8a6071c3159bdeb5e9420f4e08722c41115f90` |
| source result | `f52ce2103a07086a7118975d69f49b7cbeca01ac0ca7c5a15fd6d2a96fbc51fa` |

