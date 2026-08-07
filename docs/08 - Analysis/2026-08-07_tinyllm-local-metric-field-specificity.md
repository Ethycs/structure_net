# TinyLLM local metric-field transport and specificity corrective

**Status:** NOT CONFIRMED — CHECKPOINT-STRATIFIED GEOMETRY, 1/3  
**Date:** 2026-08-07  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, `UNDERPOWERED`, frozen-checkpoint/frozen-artifact diagnostics  
**Hypotheses:** `tinyllm-c2-local-metric-field-transport-v1`, `tinyllm-c2-local-metric-field-specificity-v1`  
**Preregistrations:** [initial transport](../07%20-%20Status%20Reports/2026-08-07_tinyllm-local-metric-field-transport-preregistration.md), [specificity corrective](../07%20-%20Status%20Reports/2026-08-07_tinyllm-local-metric-field-specificity-preregistration.md)

## Verdict

One universal nuisance-invariant local task-metric interface is **not
supported**. After correcting a structurally invalid control, only seed 29
preserves its rank-two local task projector across every amplitude,
orientation, offset, and composed acquisition action on both fresh shifts.
Seeds 7 and 53 fail paired tail geometry in all eight cells. The corrected
checkpoint gate passes `1/3`, below the required `3/3`.

The correction matters. The initial registered control permuted nuisance
replicates while preserving semantic phase and required that permutation to
fail. Under nuisance invariance those states are equivalent, so the control
was mathematically incompatible with the hypothesis. Its apparent `0/3`
result is control-invalid and is not evidence for a nuisance-specific field.

The locked corrective makes that permutation a positive equivalence control
and uses a five-bin semantic-phase shift plus deterministic random rank-two
planes as negative controls. All three checkpoints then pass:

- same-phase nuisance equivalence;
- semantic-phase specificity; and
- random-plane specificity.

The remaining failure is therefore localized: seeds 7 and 53 contain a
minority of acquisition-transformed states whose local task plane moves too
far for the identity action. This is a checkpoint-stratified metric atlas, not
one portable invariant plane field.

```text
same-phase nuisance replicates agree        3/3
different semantic phases are distinguishable 3/3
random rank-two planes are distinguishable 3/3
exact acquisition-action pairing passes all cells 1/3
-------------------------------------------------
universal nuisance-invariant metric field   rejected
```

## Why the initial causal result is not interpretable

The initial fresh-cohort CUDA campaign itself was numerically valid:

| Gate | Result |
| --- | ---: |
| input and target-pair contract | 3/3 |
| Jacobian / finite-difference / decomposition contract | 3/3 |
| exact rank-three continuation control | 3/3 |
| local tangent passes every fresh cell | 3/3 |
| registered geometry gate | 0/3 |
| registered causal-specificity gate | 0/3 |

The transported tangent passed every arm and regime descriptively. However,
the same-phase nuisance-permuted tangent and random rank-two tangent also
passed the loose continuous endpoint. Requiring the nuisance permutation to
fail was wrong, while random-plane success shows that this endpoint cannot by
itself identify the transported plane. The campaign therefore supports only
the fresh replication of **local tangent sufficiency**, not a specific causal
transport law.

No causal metric-transport claim is entered into the evidence system from the
initial `0/3` classification.

## Corrected primary gates

| Gate across all eight cells | seed 7 | seed 29 | seed 53 |
| --- | ---: | ---: | ---: |
| source validity | pass | pass | pass |
| paired nuisance geometry | **fail** | **pass** | **fail** |
| same-phase nuisance equivalence | pass | pass | pass |
| five-bin semantic-phase specificity | pass | pass | pass |
| random-plane specificity | pass | pass | pass |
| complete corrected checkpoint gate | **fail** | **pass** | **fail** |

The universal hypothesis required all three complete checkpoint gates and
therefore fails `1/3`.

## Tail geometry, not median geometry, separates checkpoints

| seed | minimum paired median cosine | minimum paired p10 cosine | maximum paired p95 distance | minimum nuisance-equivalence cosine | minimum phase margin | classification |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 7 | 0.9979 | **0.5708** | **0.9686** | 0.9969 | 0.6549 | nuisance variant |
| 29 | 0.9994 | **0.9882** | **0.3642** | 0.9994 | 0.6717 | invariant, phase specific |
| 53 | 0.9914 | **0.7919** | **0.7738** | 0.9884 | 0.5075 | nuisance variant |

The thresholds were paired p10 at least `0.90` and paired p95 projector
distance at most `0.43589`. Medians alone would have produced a false positive:
every failed seed/cell still has paired median cosine above `0.99`;
the nonportable behavior lives in the distribution tails.

Seeds 7 and 53 fail every amplitude, orientation, offset, and composed cell
under the joint tail gate. The failure is not localized to one removable group
generator. Seed 29 passes all four actions under composition and
extrapolation, demonstrating that the declared identity law is achievable but
not checkpoint universal.

## Interpretation

The local continuation studies remain positive at their actual scope:
orbit-local task tangents repair the failed writer on reused and fresh
cohorts. What fails is the stronger architectural interface

```text
P_T(g x) = P_T(x)
```

for every declared acquisition action, shift, and checkpoint. The causal
coordinate error can be locally corrected without admitting one stable
nuisance-invariant rank-two plane field.

This also explains why a post-hoc symmetry layer is not yet justified. Exact
equivariance of the observation front end and near-perfect median projector
agreement do not fix the weak, state-dependent directions that control tail
transport. A sidecar trained now would be asked to average across a mechanism
that the frozen checkpoints realize differently.

The next low-cost diagnostic, if pursued, should use the stored Jacobians to
test the normalized pullback metric `J^T J / tr(J^T J)` or dominant task
covector under the same corrected controls. That weighting discounts the very
weak second singular direction rather than giving it equal topological weight
through a rank-two projector. It must be preregistered and remain geometry-only
until it passes all three checkpoints; no new training is warranted now.

Link cobordism is not implicated by this outcome. No canonical
codimension-two zero set or degree defect was produced, so there is no declared
link whose cobordism class could certify the transport failure.

## Campaign integrity

| Item | Initial fresh CUDA campaign | Corrective frozen-artifact campaign |
| --- | --- | --- |
| requested / completed / failed / excluded | 3 / 3 / 0 / 0 | 3 / 3 / 0 / 0 |
| trained models / fitted writers / fitted observers | 0 / 0 / 0 | 0 / 0 / 0 |
| fresh acquisition-action cells | 24 | reused exactly |
| stored Jacobian fields | 30 | reused exactly |
| device | NVIDIA GeForce RTX 3060 | CPU |
| analysis time | 15.86 seconds | 0.11 seconds |
| implementation SHA-256 | `e3a05118971bba51e50e07e0fc426a8b321f0ecf823ded7dfec6b3539b630bfd` | `01e15e21fc3bae68553ac81d5dbffc8252a18628ab58b8c6be19fb53515ae5f0` |
| campaign SHA-256 | `2aa80cff5882cb79754d732e9012c810c2624934bc5be9e03c9e77de4f525f55` | `b2be5e8bcbeacd82baab6193a123c3f0e2002836edde161e0b371c162b548fe5` |
| final DVC data root | `9f9077c17fbbc668805088bf604deafc.dir` (`1,904` files, `39,816,811,567` bytes) | same final root |
| lakeFS snapshot | `8eccad2c763ea0230fde1e484b2d8c631dbe91524799c21920686bd23d704872` | same clean snapshot |

Fingerprint-matched reruns left both completed aggregates byte-identical.
The final DVC root is synchronized to
`lakefs://artifacts/main/structure-net/`; the exact directory object exists in
the cited clean lakeFS commit and the branch reports no uncommitted objects.

## Artifacts and reproduction

- initial aggregate:
  `data/experiments/tinyllm_local_metric_field_transport/20260807_d6_fresh_cohort/campaign_results.json`
- initial result/projector arrays:
  `data/experiments/tinyllm_local_metric_field_transport/20260807_d6_fresh_cohort/runs/seed_*/`
- corrective aggregate:
  `data/experiments/tinyllm_local_metric_field_specificity/20260807_corrective_v1/campaign_results.json`
- corrective result records:
  `data/experiments/tinyllm_local_metric_field_specificity/20260807_corrective_v1/runs/seed_*/result.json`
- initial runner:
  `experiments/structure_net/tinyllm_local_metric_field_transport.py`
- corrective runner:
  `experiments/structure_net/tinyllm_local_metric_field_specificity.py`
- focused tests:
  `tests/structure_net/test_tinyllm_local_metric_field_transport.py`,
  `tests/structure_net/test_tinyllm_local_metric_field_specificity.py`
- meta-hypothesis:
  `data/meta_hypotheses/tinyllm-c2-local-metric-field-specificity-v1.json`

```bash
MPLCONFIGDIR=/tmp/matplotlib-structure-net \
pixi run python -m \
  experiments.structure_net.tinyllm_local_metric_field_transport \
  --output data/experiments/tinyllm_local_metric_field_transport/20260807_d6_fresh_cohort \
  --device cuda

pixi run python -m \
  experiments.structure_net.tinyllm_local_metric_field_specificity \
  --output data/experiments/tinyllm_local_metric_field_specificity/20260807_corrective_v1
```

## Method boundaries

The correction reuses three selected checkpoints and already-generated fresh
cohorts. It tests projector geometry, not a corrected phase-shifted causal
intervention. The five-bin phase control is one locked mismatch rather than a
complete orbit scan. The acquisition group covers scale, planar orientation,
and constant offset, not speed, direction, harmonics, drift, or noise. Three
checkpoints do not establish population prevalence.
