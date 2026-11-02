# TinyLLM fixed-chart function-class preflight

**Status:** FROZEN BEFORE EXECUTABLE PREFLIGHT

**Date:** 2026-08-11

**Conformance:** NAL-STD-EXPERIMENT `DESIGN / NO-TRAINING ARCHITECTURE PREFLIGHT`

**Hypothesis:** `tinyllm-fixed-chart-function-class-v1`

## Decision question

The learned d6/d10 sensor is strongly ordered but support-relative. Joint
physical loss, blockwise clipping, complete-continuation training, and
training-cohort affine transport have all failed their population gates. Before
another training campaign, ask the structural question:

> Can a nontrivial learned scalar correction be guaranteed to preserve the
> exact physical cosine chart on the current noiseless identifiable task?

This preflight performs no optimization, loads no TinyLLM checkpoint, and
creates no primary model outcome. It compares two function classes against the
existing analytic positive control.

## Fixed source contracts

| Source | SHA-256 |
| --- | --- |
| calibrated generator and front ends | `73e152b9b6c7ab51ad08e6a80786c32a71566f9d4fb410ff2cf3df53761faf77` |
| fixed interval interface | `b027eb55d971b87f598bd600131b33ead82463e3e73766fcdce38fdc3a5497c6` |

The task is the default sixteen-bin future-cosine task. Re-run the declared
identifiability contract and generate 4,096 fresh composition plus 4,096 fresh
extrapolation observations from fixed preflight seeds. The analytic
canonicalizer must pass:

```text
composition:   corr >= .99, RMSE <= .05
extrapolation: corr >= .99, RMSE <= .08.
```

Failure is `invalid_source_contract`; no function-class conclusion may be
drawn.

## Candidate A: monotone fixed-endpoint scalar correction

Use the explicit one-parameter family

```text
m_a(u) = u + a u (1 - u^2),       a in {-0.4, +0.4}.
```

For `|a| < 1/2`, this family is odd, fixes `-1`, `0`, and `1`, maps the interval
to itself, and is strictly orientation-preserving because

```text
m_a'(u) = 1 + a(1 - 3u^2) > 0.
```

Evaluate both witnesses on a fixed 65,537-point grid over `[-1,1]`. Require:

- maximum endpoint, oddness, and range violation at most `1e-6`;
- analytic derivative lower bound and measured finite-difference slope both
  strictly positive;
- maximum nonidentity displacement at least `.05`;
- at least one percent of fixed sixteen-bin argmax assignments differ from the
  identity chart.

Passing all constraints while changing the fixed interval task is a constructive
counterexample: sign, order, range, and endpoints do not uniquely type physical
cosine. No learned fit is involved.

## Candidate B: frozen physical spine with learned auxiliary carrier

Define the public scalar only through the existing analytic canonicalizer:

```text
physical_scalar(x, calibration) = analytic_cosine(x, calibration).
```

A separately parameterized equivariant carrier may be learned, but it is
returned under a distinct typed field and cannot write the public scalar. Check
two independently randomized auxiliary parameter states on both fresh cohorts:

- physical outputs are identical within `1e-7`;
- physical outputs match the analytic positive control exactly;
- the physical output has no gradient path to any auxiliary parameter;
- the auxiliary carrier has nonzero parameter count and changes between the
  two states.

This is a type-separation contract, not a claim that the auxiliary carrier is
task-useful.

## Locked classification

| Outcome | Classification | Decision |
| --- | --- | --- |
| source contracts pass; monotone counterexample exists; typed split passes | `exact_physical_chart_requires_frozen_scalar_spine` | current task cannot support a nontrivial learned scalar while guaranteeing the exact chart; do not train |
| source contracts pass; no allowed monotone witness changes the interval task | `monotone_endpoint_typing_sufficient_in_declared_family` | a prospective constrained scalar head is licensed |
| typed split leaks auxiliary parameters into the physical scalar | `typed_split_implementation_invalid` | repair systems code only |
| identifiability, source hashes, analytic fidelity, or finiteness fails | `invalid_source_contract` | stop without interpretation |

No threshold may be weakened after inspecting the preflight. A successful
counterexample is not a negative result about quotient learning; it proves that
the proposed architectural contract is under-specified.

## Consequence for the research program

If the expected counterexample and typed split both pass, the current exact
synthetic task has reached a design boundary. Exact physical typing by
construction reduces the scalar path to the analytic solution already used as
the positive control. A new learned-sensor experiment is scientifically
meaningful only if it changes scope, for example:

- noisy or incomplete acquisition, where the analytic spine is an estimator
  rather than an oracle;
- a richer calibrated group with a nontrivial invariant carrier;
- a different identifiable task whose sufficient representation is not already
  available analytically; or
- a typed multi-channel interface where the frozen physical scalar and learned
  auxiliary semantics have separately declared downstream roles.

Do not launch another five-seed d6/d10 optimization campaign on the present
scalar task unless this preflight licenses it.
