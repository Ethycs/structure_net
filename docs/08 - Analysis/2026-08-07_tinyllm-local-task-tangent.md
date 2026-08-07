# TinyLLM fixed-writer scalar task-tangent replication

**Status:** NOT CONFIRMED — SCALAR TANGENT ENDPOINT 3/3, COMPLETE GATE 2/3  
**Date:** 2026-08-07  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, `UNDERPOWERED`, post-outcome diagnostic  
**Hypothesis:** `tinyllm-c2-local-task-tangent-v1`  
**Preregistration:** [local task-tangent preregistration](../07%20-%20Status%20Reports/2026-08-07_tinyllm-local-task-tangent-preregistration.md)

## Verdict

The complete preregistered hypothesis is **not confirmed**. Seeds 29 and 53
pass the local task-tangent gate, while seed 7 misses one conservative
specificity margin. The campaign conclusion is therefore
`checkpoint_stratified_local_geometry`, not a universal three-checkpoint
result.

The narrower causal result replicates exactly across writer choice. The
rank-one scalar circular-angle tangent correction passes the unchanged
continuous endpoint in every one of the 12 held-out composition and
extrapolation cells. Its two-dimensional first-order kernel leaves the output
almost unchanged from the failed order-four write. Fine/coarse derivatives
converge, and the local scalar model explains `0.986--0.997` of the signed
direct-minus-writer output change.

```text
fixed quotient-only order-four writer error
  -> one orbit-local decoder covector identifies the task-relevant correction
  -> its two-dimensional scalar-angle kernel retains the writer failure.
```

This is not a new universal portable metric. The exact held-out residual is
still used to determine the correction amplitude, and the registered
specificity gate passes only `2/3` checkpoints. It is, however, independent
support for the local-task-metric interpretation: a separate campaign using a
context-conditioned writer reached the same `12/12` tangent endpoint, while a
richer complex-moment Jacobian independently found an inert finite-scale
kernel.

## Preregistered gates

| Gate | Result | Required | Verdict |
| --- | ---: | ---: | --- |
| predecessor order-four replay | **3/3** | 3/3 | pass |
| numerical decomposition contract | **3/3** | 3/3 | pass |
| zero/exact/direct-rank-three target controls | **3/3** | 3/3 | pass |
| adequate scalar local linearization | **3/3** | 3/3 | pass |
| scalar tangent passes all four cells | **3/3** | 3/3 | pass |
| every named control fails at least one cell | **3/3** | 3/3 | pass |
| every control is at least `0.125` bins worse in aggregate | **2/3** | 3/3 | **fail** |
| complete local task-tangent gate | **2/3** | 3/3 | **fail** |

All four control families meet both specificity conditions for seeds 29 and
53. For seed 7, `kernel_only` fails three of four cells but its aggregate
margin over tangent is `0.119106` bins, `0.005894` below the locked `0.125`
threshold. The flipped, shuffled, and random controls remain specific. The
formal seed-7 class is `nonunique_or_curved_correction`; that fixed fallback
label must not be read as positive evidence for curvature or a second
sufficient direction.

## Checkpoint evidence

| seed | signed-error `R2` | residual MAE fraction | order-4 mean | tangent mean | direct rank-3 mean | tangent cells | complete gate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 7 | 0.9973 | 0.0306 | 0.14181 | **0.02235** | 0.02093 | **4/4** | no |
| 29 | 0.9859 | 0.0566 | 0.22078 | **0.04918** | 0.04408 | **4/4** | **yes** |
| 53 | 0.9889 | 0.0454 | 0.15723 | **0.01807** | 0.01593 | **4/4** | **yes** |

The tangent correction nearly reaches the actual rank-three positive control
without adding the scalar-angle kernel. Its aggregate improvement over the
failed writer is `0.11946`, `0.17160`, and `0.13916` bins for seeds 7, 29, and
53.

## Direction and correspondence controls

| seed | tangent | kernel | flipped | shuffled | random | specific controls |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | **0.02235** | 0.14146 | 0.28152 | 0.16589 | 0.15369 | 3/4 |
| 29 | **0.04918** | 0.22075 | 0.42237 | 0.28049 | 0.22886 | **4/4** |
| 53 | **0.01807** | 0.15677 | 0.31189 | 0.18273 | 0.15409 | **4/4** |

Every kernel, flipped, shuffled, and random family fails at least one cell.
The kernel output is essentially the original order-four output: its aggregate
mean differs from order four by only `0.00003--0.00046` bins. Reversing the
tangent roughly doubles the original error. Shuffling the tangent across
orbits and rotating it to a norm-matched random direction also destroy the
complete correction. The intervention therefore depends on sign,
example-level correspondence, and direction rather than perturbation norm
alone.

## Residual geometry

The scalar tangent is one-dimensional per orbit, while its first-order kernel
is two-dimensional. Their mean norm fractions are:

| seed | tangent fraction | kernel fraction |
| ---: | ---: | ---: |
| 7 | 0.528 | 0.732 |
| 29 | 0.642 | 0.657 |
| 53 | 0.586 | 0.734 |

The fractions combine in squared norm, not by ordinary addition. The kernel
is therefore not a negligible Euclidean remainder. It carries substantial
coordinate error while being causally irrelevant to the tested circular task
endpoint at this intervention scale.

The numerical chart is well behaved:

- fine/coarse derivative cosine is `0.99999914--0.99999994`;
- relative derivative difference is `0.000353--0.001316`;
- signed-error `R2` is `0.9859--0.9973`;
- residual MAE is `3.06--5.66%` of observed error;
- sign agreement is `0.9959--1.0000` above the registered magnitude floor;
- maximum tangent/kernel cosine is below `5.8e-15`; and
- predecessor replay error is exactly zero in the stored metrics.

## Reconciliation with the other continuation audits

This campaign is not a duplicate of the two neighboring studies:

1. `tinyllm-c2-local-continuation-tangent-v1` starts from the
   context-conditioned `context_m04` writer. Its scalar tangent also passes
   all 12 cells, but seed 29 narrowly misses random-control specificity.
2. `tinyllm-c2-local-continuation-tangent-kernel-v1` differentiates the full
   two-component complex posterior moment. Its rank-two row-space tangent
   passes all 12 cells and its one-dimensional kernel is inert, but the
   normalized metric direction is not stable across cells.
3. This campaign starts from the separate quotient-only `quotient_order4`
   writer and uses only the scalar circular angle. It again passes all 12
   tangent endpoints; the narrow specificity miss moves to seed 7.

Together, the three interventions reject a writer-specific accident and show
that the circular-angle component alone is sufficient at the tested states.
They do **not** establish one checkpoint-level or cross-checkpoint metric
tensor, and they do not provide the unavailable correction amplitude at
inference time.

## Next shortest experiment

Do not train another writer yet. Apply the declared symmetry group to the
orbit-local covector field itself. In standardized carrier coordinates, a
group action with carrier representation `rho(g)` predicts the contragredient
transport law

```text
gradient(g x) = rho(g)^(-T) gradient(x).
```

Compare rank-one projectors rather than signed vectors when the covector sign
is gauge-ambiguous. Use exact phase-matched nuisance transformations, the
existing checkpoints, and no fitted alignment. Test composition and
extrapolation separately, then causally patch a transported tangent. If the
projector and patch gates pass, the symmetry law is the minimal typed sidecar
interface. If they fail, retain an orbit-local causal atlas and stop the
universal-sidecar claim.

## Campaign integrity

| Item | Value |
| --- | --- |
| requested / completed / failed / excluded | 3 / 3 / 0 / 0 |
| checkpoints | d6 seeds 7, 29, 53 |
| held-out cells | 3 checkpoints x 2 cohorts x 2 shifts = 12 |
| exact `C2` orbits per cell | 64 |
| trained models / fitted observers / fitted writers | 0 / 0 / 0 |
| finite-difference perturbation continuations | 144 |
| causal component states | 108 |
| device | NVIDIA GeForce RTX 2060 SUPER, CUDA |
| PyTorch / Python | 2.5.1+cu121 / 3.11.13 |
| peak allocated CUDA memory | 0.262 GiB |
| campaign analysis time | 13.29 seconds |
| implementation SHA-256 | `a89039dd2999b933c69067b76f3f9b311af65f516a6b956fe391d0b1a68d5063` |
| campaign SHA-256 | `824a655b5c6d74f3c77259b9b7cacce3b4b3ea868ba74f48ba63fd5a24395130` |
| final DVC data root | `7053c5bcd5433ee6822ec9825b782b53.dir` (`1,847` files, `39,814,283,869` bytes) |
| lakeFS snapshot | `fd8392ef275fda3a4e98fbe957208151061cfb8c0aeb48451b6455ecc326ed55` |

The one-checkpoint CUDA lifecycle used all 64 orbits but is explicitly
`systems_lifecycle_only_not_quality_evidence`; it is not pooled. Re-running
the primary command returned the immutable aggregate without rewriting any
result byte.

The final DVC root was pushed to the configured
`lakefs://artifacts/main/structure-net/` remote and is contained in the cited
clean lakeFS commit.

## Artifacts and reproduction

- aggregate:
  `data/experiments/tinyllm_local_task_tangent/20260807_d6_preregistered_diagnostic/campaign_results.json`
- per-checkpoint records:
  `data/experiments/tinyllm_local_task_tangent/20260807_d6_preregistered_diagnostic/runs/seed_*/result.json`
- systems-only lifecycle:
  `data/experiments/tinyllm_local_task_tangent/20260807_shakedown_cuda/`
- runner:
  `experiments/structure_net/tinyllm_local_task_tangent.py`
- tests:
  `tests/structure_net/test_tinyllm_local_task_tangent.py`
- meta-hypothesis:
  `data/meta_hypotheses/tinyllm-c2-local-task-tangent-v1.json`

The named hypothesis and all three direct experiment records passed
authoritative Chroma readback. Legacy NumPy-2.0 consumer and telemetry warnings
were non-fatal; the readback gate and strict JSON ledger are authoritative.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
MPLCONFIGDIR=/tmp/matplotlib-structure-net \
pixi run python -m \
  experiments.structure_net.tinyllm_local_task_tangent \
  --device cuda:0 \
  --output \
  data/experiments/tinyllm_local_task_tangent/20260807_d6_preregistered_diagnostic
```

## Method boundaries

The tangent is defined by the frozen answer-token circular decoder and is not
an intrinsic representation decomposition. The intervention uses the exact
held-out coordinate residual and is diagnostic rather than deployable. The
rank-three basis, quotient writer, and held-out cells were selected or
inspected in predecessor studies. Every component patch is off manifold.
Three selected checkpoints do not establish population prevalence, and a
passing scalar correction does not establish a stable group-transport law.
