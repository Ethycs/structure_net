# TinyLLM final-query semantic-kernel preregistration

**Status:** PREREGISTERED — NO SEMANTIC-KERNEL OUTCOME GENERATED OR INSPECTED  
**Date:** 2026-08-10  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, `UNDERPOWERED`,
frozen-checkpoint no-fit causal decomposition  
**Hypothesis:** `tinyllm-final-query-semantic-kernel-v1`

## Question and directional prediction

The valid d8 whole-query barycenter changes the complete answer posterior, and
the centered-logit decomposition shows why: a rank-15 answer-sensitive row
space contains most of the opposite-sheet chord. That does not determine
whether the same chord is null for the one-dimensional cosine coordinate that
defines the supervised task geometry.

This study asks:

> Does the local kernel of the frozen posterior-mean cosine coordinate contain
> most of the exact same-cosine final-query chord and support a finite
> coordinate-preserving quotient patch, even though that patch does not
> preserve the complete 16-answer posterior?

The directional prediction is a strict separation:

```text
scalar cosine coordinate: kernel patch preserves and contracts
complete answer posterior: the same patch fails posterior preservation.
```

This is not a relaxed reanalysis of the complete-posterior null. The scalar
coordinate and complete posterior are declared as different computational
quotients, and both endpoints remain visible.

## Frozen sources and unit of evidence

The unit of evidence is one retained d8 cosine-interval seed-7 checkpoint. The
campaign is explicitly underpowered and cannot establish population
prevalence.

Reuse and validate:

- checkpoint SHA-256
  `8170856da5e6b1f8b7a7f1c2c2121ffd4c53edaaf2ea5aabe01fa14d2f063f3f`;
- task-relative parent campaign SHA-256
  `b5d6b613683326024eeb00944a3d0aba4dd7a251dd22f5fa7e8561b5e9d6aae4`;
- parent d8 result SHA-256
  `79119a4fa6df16a2fe2d65c34ec1aca30072a0b53d3d8ea312c92b29f86d1476`;
- parent d8 diagnostics SHA-256
  `ef86c31d418c04b7ffd4bec3b66135057c4ca825b3e4da8263a920cbce0de1dd`;
- final-query posterior-kernel campaign SHA-256
  `93d9e22d766aa56943f0bd0c41b31ed25dc592e97ae0667863f3963588c38cde`;
- final-query posterior-kernel diagnostics SHA-256
  `ad1798b960375503381cb59725dfb84db9d823f488ab8073fea178d0399e0974`;
- frozen parent runner SHA-256
  `67e0854162cfd5ef4ff7dbc18f2cd2b7e9c6e88e10543e30bf0baf6b87b1c8ee`;
  and
- frozen posterior-kernel runner SHA-256
  `52b59f96e53e76c1794e9d2d251760a76eb2300f8e64e70dbe165090a5899895`.

The parent must remain a valid d8 causal null with no whole-residual sufficient
front. The posterior-kernel study must remain valid, primary-failing, and
classified `task_rowspace_contains_material_fiber_chord`. Checkpoint and
model/system-state digests must remain unchanged.

## Exact cohorts and fixed controls

Regenerate the same `512` exact two-sheet fibers in each regime:

- `training_support`, seed `850011`; and
- `outside_range`, seed `850021`.

Within each pair, cosine target, nuisance values, and complete
pre-quantization noise remain identical while phase changes between the two
preimages. Revalidate the full parent cohort contract. Baseline and full
barycenter posteriors must replay both stored parent arrays and the immediately
preceding posterior-kernel arrays within `2e-6`.

Fix the model, architecture, checkpoint, final post-MLP query cut, final layer
normalization, answer head, examples, ordering, batch size, task metrics, and
all thresholds. No model parameter, probe, decoder, carrier, metric, or
threshold is fit.

## Declared scalar task map

Let the frozen answer posterior at final query state `h` be

```text
p(h) = softmax(W LN(h)) in Delta^15.
```

The cosine task already evaluates its continuous prediction with fixed answer
centers

```text
c = linspace(-1, 1, 16).
```

Declare the scalar semantic coordinate

```text
s(h) = c^T p(h).
```

This is exactly the posterior-mean coordinate used by the registered cosine
task-map correlation and RMSE. It is computed only from the frozen model
output and fixed task vocabulary; it does not use the latent phase, branch,
target label, or evaluation outcome.

## Analytic semantic tangent/kernel decomposition

For paired final-query states `h+`, `h-`, define

```text
b = (h+ + h-) / 2,
v = b - h.
```

Using the exact centered-logit Jacobian `J_g(b)` from the preceding study and
`p_b = p(b)`, compute

```text
j_s(b) = D s(b)
       = sum_k p_b,k (c_k - c^T p_b) J_g,k(b).
```

Use the Euclidean Moore--Penrose projector with `rtol=1e-6` and
`atol=1e-10`:

```text
P_sem = pinv(j_s) j_s,
v_sem = P_sem v,
v_ker = (I - P_sem) v.
```

Evaluate:

| Arm | Patched query | Meaning |
| --- | --- | --- |
| replay | `h` | unchanged frozen computation |
| full | `h + v` | exact pair barycenter |
| semantic-only | `h + v_sem` | locally visible scalar component |
| semantic-kernel | `h + v_ker` | locally scalar-null component |
| random-kernel | `h + (I-P_random)v` | deterministic rank-1 random control |

Use random-projector seeds `852011` and `852021` for support and outside range.
Only final layer normalization and the answer head follow the intervention.

## Numerical and nesting contracts

Require in each regime:

1. all `512` scalar Jacobians are finite and have rank exactly `1`;
2. maximum decomposition reconstruction error is at most `1e-8`;
3. maximum relative scalar-kernel leakage is at most `1e-6`;
4. symmetric finite differences on the first `32` pair barycenters, step
   `1e-3`, agree with the analytic scalar Jacobian to maximum relative error
   `0.02`;
5. the scalar Jacobian lies in the centered-logit Jacobian row space to maximum
   relative leakage `1e-8`;
6. the preceding posterior projector again has rank `15` on every fiber;
7. baseline/full posterior replay error is at most `2e-6` against both frozen
   sources;
8. every state, posterior, decomposition, and metric is finite; and
9. model and system state remain byte-digest unchanged.

Any failure invalidates the candidate scientific endpoints.

## Scalar-coordinate preservation endpoint

For each row and arm define `s_arm = c^T p_arm`. Relative to replay, the
semantic-kernel arm must satisfy all of:

- mean absolute coordinate change at most `0.01`;
- p95 absolute coordinate change at most `0.03`;
- cosine task-map correlation loss at most `0.01`; and
- cosine RMSE increase at most `0.01`.

The absolute thresholds are below one quarter of the fixed answer-center
spacing `2/15`. They are frozen before any semantic-kernel output is produced.

## Fiber contraction and quotient-separation endpoints

Compute the semantic-kernel patched pair-distance ratio

```text
rho_sem = mean_i ||h_i,+^ker - h_i,-^ker||
          / mean_i ||h_i,+ - h_i,-||.
```

Require `rho_sem <= 0.25`. The stored complete-posterior kernel ratio must
replay at least `0.75` in each regime.

To establish that the quotient claims differ rather than silently relaxing the
old endpoint, the semantic-kernel arm must also fail complete-posterior
preservation with mean Jensen--Shannon divergence strictly above `0.02` in
both regimes. Accuracy and cross-entropy remain descriptive and cannot replace
either endpoint.

## Nonlinear attribution and specificity

Using actual scalar changes,

```text
Delta_full = s(h+v) - s(h),
Delta_sem  = s(h+v_sem) - s(h),
Delta_ker  = s(h+v_ker) - s(h),
```

evaluate rows with `|Delta_full| > 1e-5`. Require at least `10%` of rows to be
material and:

- mean `|Delta_full-Delta_sem|/|Delta_full| <= 0.25`;
- p95 relative residual at most `0.75`;
- sign agreement between `Delta_sem` and `Delta_full` at least `0.90`;
- mean `|Delta_ker|/|Delta_full| <= 0.25`; and
- p95 kernel/full ratio at most `0.75`.

The semantic-only arm must fail the scalar-coordinate preservation endpoint,
showing that the locally visible component is causally active.

The random rank-1 kernel arm must fail scalar-coordinate preservation and its
mean absolute coordinate change must exceed the semantic-kernel value by at
least `0.01`. This tests task-map specificity rather than low-rank geometry
alone.

## Primary gate

The hypothesis passes only if both regimes simultaneously satisfy:

1. every source, cohort, numerical, replay, nesting, finite, baseline, and
   state-integrity contract;
2. semantic-kernel scalar-coordinate preservation;
3. `rho_sem <= 0.25` while the replayed posterior-kernel ratio is at least
   `0.75`;
4. semantic-kernel complete-posterior JS is strictly greater than `0.02`;
5. all nonlinear scalar-attribution gates;
6. semantic-only causal activity; and
7. rank-1 random-kernel specificity.

This remains one retained checkpoint. A pass supports a checkpoint-local
distinction between the scalar semantic quotient and autonomous posterior
closure; it does not confirm a population-level invariant representation.

## Interpretation table

| Outcome | Interpretation |
| --- | --- |
| scalar preserves and contracts; posterior fails; controls pass | the final query contains a causal scalar quotient hidden by the stricter complete-posterior criterion |
| scalar preserves but does not contract | a local scalar kernel is finite-patch inert but does not contain most of the fiber chord |
| scalar contracts but does not preserve | first-order scalar-kernel geometry fails as a finite intervention because of curvature |
| scalar and posterior both preserve | the prior posterior null does not replay or the quotient-separation claim is false |
| random control also preserves | scalar preservation lacks task-map specificity |
| semantic-only attribution fails | higher-order interaction, not the rank-1 tangent, drives the full scalar change |

## Artifacts, execution, and stopping rule

The primary root will be

```text
data/experiments/tinyllm_final_query_semantic_kernel/
  20260810_d8_seed7_preregistered/
```

Store strict-JSON campaign results and NPZ diagnostics, including all arm
posteriors, scalar coordinates, ranks, numerical errors, pair norms, and
nonlinear-attribution arrays. A separate underpowered CUDA shakedown must be
labeled `systems_lifecycle_only_not_quality_evidence` and cannot contribute to
the scientific record. Completed-result reuse must be byte-identical under the
scientific fingerprint.

Regardless of outcome, do not scan earlier layers, fit a new carrier or metric,
change the scalar coordinate, relax either posterior or scalar thresholds, or
retrain in this branch. Failure closes the simple local scalar-kernel account.
A pass licenses only a fresh multi-seed frozen replication before any
architectural intervention.
