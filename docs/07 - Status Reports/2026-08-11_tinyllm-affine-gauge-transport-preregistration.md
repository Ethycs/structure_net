# TinyLLM Affine Gauge and Inverse-Embedding Transport Preregistration

**Status:** FROZEN BEFORE TRAINING-COHORT SENSOR EXTRACTION

**Date:** 2026-08-11

**Conformance:** NAL-STD-EXPERIMENT `REGISTERED POST-OUTCOME ARTIFACT-ONLY DIAGNOSTIC`

**Hypothesis:** `tinyllm-affine-gauge-transport-v1`

## Question

Is the failed learned physical sensor merely expressed in a checkpoint-local
affine gauge that can be canonicalized while transporting the scalar embedding
to leave the complete trained function unchanged?

The full-interface campaign establishes that the learned sensor remains mixed
in d6 and sign-reversed in d10, while the full-depth scalar is positively
oriented in every seed. Before constructing another architecture, this audit
tests the cheapest remaining distinction:

```text
pure affine coordinate gauge
                 versus
support-relative or nonlinear sensor calibration defect.
```

No model parameter is optimized and no checkpoint is selected or changed.

## Outcome-known boundary

The complete full-interface result, all held-out scalar arrays, and the `0/5`
population outcomes are known. An exploratory composition-cohort self-fit
performed before this registration showed optimistic composition repair but
only two extrapolation task-floor passes per preset. That calculation is
excluded from registered evidence.

The novel locked calculation fits the affine law only on the exact sealed
4,096-example training cohort, checks algebraic embedding transport from the
saved checkpoints, and applies that one law unchanged to the registered
composition and extrapolation arrays. The result is corrective mechanism
evidence, not fresh confirmation.

## Population and artifacts

Use all ten physical and all ten pair-shuffled full-interface arms:

```text
preset: d6, d10
seed:   7, 17, 29, 41, 53
```

Pin the parent campaign SHA-256:

`cf8f27e088f9022b78f36d285f2ddb49920bd6bc740d71e6efac7b04ab877cc1`

For each arm require exact hashes for its full checkpoint and diagnostics,
the parent source result, training tensor, pair schedule, target permutation,
held-out cohorts, task floors, and final model/interface states.

## Training-cohort affine chart

Reload only the saved encoder and scalar embedding required by the diagnostic.
Regenerate the exact sealed training observation tensor and compute the saved
sensor scalar `s` without gradients.

For `physical_true`, let `y = cos(phi)`. For `pair_shuffled`, reuse the exact
sealed pair-preserving target permutation from the parent campaign. Fit the
unique ordinary least-squares law

```text
s = alpha y + beta
```

using all 4,096 training examples. No robust loss, split selection, clipping,
regularization, nonlinear map, or held-out refit is allowed. Require
`abs(alpha) >= 1e-4` for validity.

Define the canonical scalar

```text
u_hat = (s - beta) / alpha.
```

Apply the frozen `(alpha, beta)` unchanged to composition and extrapolation.

## Exact inverse transport

For the saved scalar embedding

```text
E(s) = W s + b,
```

define

```text
W_new = alpha W,
b_new = b + beta W.
```

Then evaluate the identity

```text
E_new(u_hat) = E(s)
```

on the training, composition, and extrapolation scalars in float32. Maximum
absolute error must be at most `2e-6`. This is an algebraic counterfactual;
the transported parameters are not saved as a repaired model and cannot count
as a prospective task repair.

Because the injected embedding is unchanged, retain the parent full-depth
scalar, task, and branch measurements exactly. Record their byte/source
identity rather than rerunning or refitting them.

## Canonical-front endpoint

At the canonical front on composition and extrapolation, retain the parent
joint endpoint:

```text
Pearson correlation(u_hat, physical cosine)             >= .90
conditional branch balanced accuracy                    <= .55
conditional branch log-loss gain                         <= .02
fixed interval-decoder exact-bin accuracy                >= inherited source floor
```

An invertible affine scalar change preserves the represented information, so
reuse the arm's registered conditional-branch measurements and require that
they already pass. Recompute correlation and fixed interval task metrics from
`u_hat`; do not recalibrate the task decoder.

A seed passes only when its canonical front passes both shifts and its
unchanged parent full-depth endpoint passes both shifts.

## Population gate and locked outcomes

Require at least `4/5` physical seed passes separately in d6 and d10, at most
`1/5` pair-shuffled seed pass separately, all ten cells valid, and all twenty
embedding-transport identities within tolerance.

| Outcome | Classification | Meaning |
| --- | --- | --- |
| both physical populations pass; controls specific | `affine_gauge_transport_sufficient` | the learned solution is a physically rechartable affine gauge; build the fixed chart into the interface |
| one physical population passes | `architecture_conditional_affine_gauge_transport` | affine gauge is not portable across the declared family |
| both canonical-front populations pass but the joint population gate fails | `front_gauge_repaired_continuation_insufficient` | chart repair is affine, but full-depth task calibration remains inadequate |
| canonical-front population gate fails in either preset | `support_relative_affine_gauge_insufficient` | one training-cohort affine chart does not transfer; require a stricter fixed-chart sensor construction |
| shuffled controls exceed `1/5` | `specificity_control_failed` | the diagnostic does not isolate physical correspondence |
| any artifact, slope, transport, or finiteness contract fails | `invalid` | preserve the root and repair systems only |

No in-sample training fit, absolute correlation, sign correction alone,
composition-only success, held-out affine refit, or unchanged full-depth output
can rescue a failed population gate.

## Derived diagnostics

Record without additional gates:

- training, composition, and extrapolation affine `R2`;
- canonical RMSE and exact-bin accuracy;
- training-to-shift changes in best-fit slope and intercept;
- seedwise orientation sign;
- parent full-depth endpoint pass pattern;
- float32 and float64 inverse-transport error.

## Artifact root

Primary diagnostic:

`data/experiments/tinyllm_affine_gauge_transport/20260811_d6_d10_registered`

Systems shakedowns use separate roots and cannot enter the aggregate.
