# J-space guided compression with a retained residual

Date: 2026-11-13.
Status: research direction clarified by the user; no new experiment launched.

The architecture contains a compressed component and a residual component. The
objective is to reliably remove the original computation covered by the basis
while preserving the combined model's task quality. A compressed branch that
independently replaces a complete attention mechanism, layer or network function
is not required.

This clarification supersedes full-replacement and autonomous-closure entry gates
in the [earlier research programme](2026-11-13_jspace-groebner-research-programme.md).
Existing protocols, measurements and reports remain historical records.

## Decomposition and execution

For an original operation F, seek an implementation

```text
F_hat(x) = C_B(x) + R_B(x)
```

C_B computes the contribution assigned to the discovered basis B in a cheaper
form. R_B preserves the remaining contribution and any necessary correction.
The branches can share inputs and retained context. Cross-component interactions
must remain represented; the basis component need not have autonomous dynamics.
If C_B approximates the covered contribution, its error can require correction
inside the retained concept span as well as outside it. The residual is therefore
not automatically an orthogonal complement.

For a linear matrix, choosing W_B gives the exact identity W = W_B + W_R with
W_R = W - W_B. This is a useful correctness reference. Execution savings require
a cheaper representation or schedule for the combined factors: storing a dense
W_R and adding W_B does not by itself remove dense work. For nonlinear blocks,
F - C_B defines a reference residual, but evaluating the original F to obtain
that residual is not the compressed implementation.

The executable decomposition may remove selected heads, matrix blocks, directions
or redundant polynomial terms, while retaining the computation that handles the
uncovered part. Count the actual operators removed and the complete cost of both
branches, shared computations and their merge.

## Role of concepts and algebra

The intended path is:

```text
Task families and concept contrasts
  -> task-Jacobian space and candidate CAVs
  -> functional validation of the candidate contribution
  -> relations and transformations involving those coordinates
  -> Groebner reduction and an explicit covered component
  -> cheaper covered computation plus retained residual
```

Functional validation concerns the claimed contribution. It does not require that
the compressed component alone produce the complete answer. Groebner analysis
should identify which expressions or transformations can be removed or rewritten,
with exact statements limited to the justified algebraic representation and
empirical error assessed on the neural computation.

Global closure is relevant only if an experiment additionally attempts to run a
component independently of all discarded context. It is not an entry requirement
for the proposed residual architecture. The compressed branch may depend on
context supplied by the retained computation, whose cost remains explicit.

## Required evidence for the next experiment

1. Specify the original computation assigned to C_B and how its contribution is
   identified. Distinguish explained variation, semantic influence and removable
   arithmetic; their percentages are different measurements.
2. Implement both branches and actually delete or bypass the assigned original
   work. Preserve the required interactions and interfaces.
3. Evaluate the combined model on held-out tasks, combinations and interventions.
   Report any approximation error and changes in task quality. A weak standalone
   compressed branch is not a failure criterion.
4. Measure complete memory and runtime, including the residual, encoding, routing
   and merge. Compare against the original and a cost-matched decomposition
   without Groebner reduction.
5. For further iterations, analyze the remaining residual for another removable
   component. Retain only iterations that meet the combined quality and cost
   criteria, accounting for accumulated error and overhead.

Numerical gates and a reservation-backed execution protocol will be frozen before
the next GPU experiment. The scientific question is how much useful computation
can be removed reliably while the residual maintains the rest of the behavior.
