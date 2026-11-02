# TinyLLM final-query task-kernel barycenter preregistration

**Status:** PREREGISTERED — NO TASK-KERNEL OUTCOME GENERATED OR INSPECTED  
**Date:** 2026-08-10  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, `UNDERPOWERED`,
frozen-checkpoint no-fit causal decomposition  
**Hypothesis:** `tinyllm-final-query-task-kernel-barycenter-v1`

## Prior result and unresolved question

The task-relative activation-barycenter campaign averaged the complete
residual sequence over exact shared-nuisance opposite-phase pairs. In the valid
d8 cosine checkpoint, the intervention improved accuracy and target
cross-entropy but changed the frozen posterior beyond the locked
Jensen--Shannon ceiling at every cut. At final post-MLP, support and
outside-range posterior JS were `0.02746` and `0.20593`.

An exact architecture corollary rules out token scope as the explanation. At
final post-MLP the continuation reads only the query token, so full and
query-only barycenters are output-identical.

The remaining low-cost question is:

> Is the failed query barycenter caused by a small task-active component of the
> opposite-sheet chord, while most of that chord lies in the frozen answer
> map's local kernel and can be removed without changing the task?

This study tests that question at the final cut only. It does not scan depth,
fit a carrier, train a probe, or retrain a model.

## Frozen sources

Reuse the retained d8 cosine-interval seed-7 checkpoint with SHA-256
`8170856da5e6b1f8b7a7f1c2c2121ffd4c53edaaf2ea5aabe01fa14d2f063f3f`.
Validate:

- parent campaign SHA-256
  `b5d6b613683326024eeb00944a3d0aba4dd7a251dd22f5fa7e8561b5e9d6aae4`;
- parent d8 result SHA-256
  `79119a4fa6df16a2fe2d65c34ec1aca30072a0b53d3d8ea312c92b29f86d1476`;
- parent d8 diagnostics SHA-256
  `ef86c31d418c04b7ffd4bec3b66135057c4ca825b3e4da8263a920cbce0de1dd`;
  and
- frozen parent runner SHA-256
  `67e0854162cfd5ef4ff7dbc18f2cd2b7e9c6e88e10543e30bf0baf6b87b1c8ee`.

The parent result must be the valid d8 causal null with no isolated or mature
front. The checkpoint and model/system-state digests must remain unchanged.

## Exact cohorts

Regenerate the parent's `512` exact two-sheet fibers for:

- `training_support`, seed `850011`; and
- `outside_range`, seed `850021`.

Within each pair, phase changes from `arccos(u)` to
`2 pi - arccos(u)`; cosine target, all declared nuisances, and the complete
pre-quantization noise array remain identical. Revalidate every parent cohort
contract and require regenerated baseline and full-barycenter posteriors to
match the stored parent arrays within `2e-6`.

The d8 outside-range baseline must again satisfy the parent's `0.15`
exact-bin accuracy floor.

## Final-query decomposition

Let `h+` and `h-` be the two final post-MLP query vectors and

```text
b = (h+ + h-) / 2,
v = b - h.
```

For the frozen final answer map, define centered answer logits

```text
g(h) = C W LN(h),
C = I - (1/K) 11^T,
K = 16.
```

Compute the exact analytic Jacobian at the pair barycenter:

```text
J_b = Dg(b),
P_task = pinv(J_b) J_b.
```

Use the Euclidean Moore--Penrose pseudoinverse with `rtol=1e-6` and
`atol=1e-10`. Decompose each row's barycenter displacement:

```text
v_task   = P_task v,
v_kernel = (I - P_task) v.
```

Evaluate four frozen interventions:

| Arm | Patched query | Meaning |
| --- | --- | --- |
| replay | `h` | unchanged computation |
| full | `h + v = b` | parent barycenter |
| task-only | `h + v_task` | component visible to the local centered-logit Jacobian |
| kernel-only | `h + v_kernel` | component locally invisible to the centered-logit Jacobian |

Context-token residuals remain unchanged in every new arm. Only final layer
normalization and the frozen answer head are run after the patch.

## Rank-matched random control

For each regime, draw one deterministic Gaussian matrix using seeds
`851011` and `851021`, orthonormalize it, and retain exactly the median
rank observed for `J_b`. Its row-space projector `P_random` defines

```text
v_random_kernel = (I - P_random) v.
```

This matches the task projector's rank and Euclidean decomposition but does
not use the answer map. It is fixed before intervention outcomes.

## Numerical contracts

Require:

1. every centered-logit Jacobian is finite and has rank exactly `15`;
2. maximum relative reconstruction error of
   `v_task + v_kernel = v` is at most `1e-8`;
3. maximum relative leakage `|J_b v_kernel| / max(|J_b v|, 1e-10)` is at
   most `1e-6`;
4. on the first `32` pair barycenters per regime, a symmetric directional
   finite difference with step `1e-3` agrees with the analytic Jacobian to
   maximum relative error `0.02`;
5. all states, logits, posteriors, decompositions, and metrics are finite; and
6. exact replay, parent-array replay, source, and model-state contracts pass.

Failure of any numerical or source contract invalidates candidate endpoints.

## Task-preservation endpoint

Reuse the parent simultaneous task-sufficiency gate relative to replay:

```text
accuracy loss <= 0.03
target cross-entropy increase <= 0.05
posterior JS <= 0.02.
```

No endpoint threshold may be selected from the new outcomes.

## Geometric and causal attribution endpoints

For each regime, compute the kernel-only patched pair-distance ratio

```text
rho_kernel =
mean_i ||h_i,+^kernel - h_i,-^kernel||
/
mean_i ||h_i,+ - h_i,-||.
```

The contraction gate is `rho_kernel <= 0.25`.

Using actual nonlinear centered-logit changes from replay, define

```text
Delta_full    = g(h + v)      - g(h),
Delta_task    = g(h + v_task) - g(h),
Delta_kernel  = g(h + v_kernel) - g(h).
```

Over rows whose `Delta_full` norm exceeds `1e-8`, require:

- mean relative task residual
  `||Delta_full - Delta_task|| / ||Delta_full|| <= 0.25`;
- p95 relative task residual at most `0.75`;
- median cosine alignment between `Delta_task` and `Delta_full` at least
  `0.90`;
- mean kernel/full effect ratio at most `0.25`; and
- p95 kernel/full effect ratio at most `0.75`.

These are actual post-intervention logit effects, not first-order predictions.

## Primary gate

The hypothesis passes only if, in both regimes:

1. the kernel-only arm passes simultaneous task sufficiency;
2. `rho_kernel <= 0.25`;
3. every nonlinear task-effect attribution threshold passes;
4. the task-only arm fails task sufficiency, matching the parent full arm's
   material output change;
5. the rank-matched random-kernel arm fails task sufficiency and has mean
   posterior JS at least `0.01` worse than the task-kernel arm; and
6. every numerical, replay, source, finite, and state-integrity contract
   passes.

This is one retained checkpoint and remains explicitly underpowered. A pass
identifies a checkpoint-local final-readout mechanism; it does not establish
population prevalence or a depth-wise quotient front.

## Interpretation table

| Outcome | Interpretation |
| --- | --- |
| kernel patch preserves task, contracts the chord, task arm explains full effect, random control fails | most opposite-sheet variation is causally task-null at the final readout; the full barycenter fails because it also removes a small task-active component |
| kernel patch preserves task but does not contract | the local kernel is inert but does not contain most fiber variation |
| kernel patch contracts but changes the task | first-order kernel geometry is not a valid finite intervention because of curvature |
| task arm does not explain the full effect | tangent/kernel interaction or higher-order layer-normalization geometry dominates |
| random control also passes | the result lacks task-Jacobian specificity |

## Boundaries and stopping rule

This experiment concerns only the final post-MLP query vector, one d8 seed-7
cosine checkpoint, two exact synthetic regimes, the centered 16-answer-logit
map, and Euclidean local tangent/kernel geometry at the fiber barycenter. A
local Jacobian kernel is not a globally invariant subspace, and contraction is
not literal information erasure.

Regardless of outcome, do not train a model, fit a probe or carrier, scan
earlier layers, relax a threshold, or select a different metric in this branch.
A failure closes first-order final-query kernel projection. A pass licenses
only a fresh multi-seed replication of the same frozen diagnostic before any
architectural intervention.
