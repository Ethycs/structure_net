# TinyLLM cross-seed symmetry-feature swap

**Status:** NOT CONFIRMED — EXACT EQUIVARIANCE, CHECKPOINT-LOCAL GAUGES  
**Date:** 2026-08-06  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, frozen-checkpoint causal diagnostic  
**Hypothesis:** `tinyllm-cross-seed-symmetry-feature-gauge-v1`  
**Preregistration:** [cross-seed symmetry-feature swap preregistration](../07%20-%20Status%20Reports/2026-08-06_tinyllm-cross-seed-symmetry-feature-swap-preregistration.md)

## Verdict

Calibrated architectural equivariance made each checkpoint invariant to the
declared acquisition group, but it did **not** fix one portable feature gauge
across independently trained models. All five encoders passed the exact
positive-similarity group contract, with maximum feature discrepancies of only
`3.58e-7` to `6.56e-7`. Nevertheless, none of the 20 directed feature swaps
passed all four held-out cells; 16 were required. No target checkpoint accepted
passing features from three sources; four of five targets were required.

The one-dimensional front-end scalar was even less portable: it passed zero of
80 cells and zero of 20 directed pairs. Thus the negative feature result cannot
be explained by choosing the cut one nonlinear map too early.

The supported conclusion is:

> Exact symmetry constrains how a representation transforms, but does not
> select a unique basis, sign, scale, or downstream convention inside the
> allowed representation. These five front ends are invariant within each
> checkpoint and co-adapted to their own scalar maps, embeddings, transformers,
> and decoders.

## Preregistered gates

| Gate | Result | Required | Verdict |
| --- | ---: | ---: | --- |
| acquisition-group contract | **5/5** checkpoints | 5/5 | pass |
| complete directed feature swaps | **0/20** | at least 16/20 | fail |
| target checkpoints accepting at least three sources | **0/5** | at least 4/5 | fail |
| secondary complete scalar swaps | **0/20** | descriptive | fail |

The full hypothesis is not confirmed. The target-checkpoint gate treats the
five independently trained targets as the replication units; the 20 directions
are not reported as 20 independent models.

## Campaign integrity

The campaign loaded the five retained d8
`learned_calibrated_equivariant` systems from the successful calibrated
identifiability study. Every `model.pt`, `frontend.pt`, and source `result.json`
was hash-checked, and both model-only and full-system state digests reproduced
the stored training records before evaluation. No model, observer, calibration,
or coordinate map was trained or fit.

| Item | Value |
| --- | --- |
| requested / completed / failed / reused | 20 / 20 / 0 / 0 |
| independently trained checkpoints | seeds 7, 17, 29, 41, 53 |
| held-out cells | two composition and two extrapolation cohorts |
| examples per cell | 512 paired N3 examples |
| trained parameters / fitted maps | 0 / 0 |
| direct replay maximum posterior error | exactly 0 |
| device | NVIDIA GeForce RTX 2060 SUPER, CUDA |
| peak allocated CUDA memory | 1.014 GiB |
| analysis time | 62.0 seconds |
| implementation SHA-256 | `0508821da71b2129b5a7a437c2f1d9aa60f3fa4b400ff3a74a59a968b9d65a50` |
| campaign SHA-256 | `b2c683c803f0dc5ec90b4b84f7858bd19ca8ddd862cf7736589d63d3a1a43feb` |
| final DVC data root | `f29e1f0e920aff74661e2a64d7ec56c1.dir` (`1,796` files, `39,812,097,258` bytes) |
| lakeFS snapshot | `71cda38c5b84bfa364c136a0741dd4ff6e77040395f4e24b5d50d8419c11a648` |

A separate two-checkpoint, 16-example CUDA lifecycle verified loading,
continuation, schema, and immutable resume. Its evidence role is
`systems_lifecycle_only_not_quality_evidence`; none of its quality outcomes
entered this report. The final DVC root is current locally, was pushed to the
configured `lakefs://artifacts/main/structure-net/` remote, and is contained in
the cited clean lakeFS commit.

## The feature gauge is bimodal

The three-channel invariant feature

```text
z = (dot(equivariant_vector, orientation),
     cross(orientation, equivariant_vector),
     signed_speed)
```

separated into two sign classes:

```text
class A: seeds 7, 17
class B: seeds 29, 41, 53
```

Across the 48 cross-class cells, the mean angular displacement was `3.084`
radians, the mean feature distance was `1.998`, and target scalar-map outputs
had mean correlation `-0.989` with the direct target scalar. No cross-class cell
passed.

Within a class, the 32 cells had mean angular displacement `0.0575` radians,
mean feature distance `0.0575`, and target scalar-map correlation `0.994`.
Even there, only 8/32 cells passed. Every passing cell was a composition cell
from one of four directions:

```text
7 <-> 17
29 <-> 53
```

This is the expected unresolved sign gauge: if `v(x)` is `SO(2)`-equivariant,
then `-v(x)` is equally equivariant. The task objective and downstream network
can choose either convention.

## Extrapolation breaks the closest charts

All 40 extrapolation cells failed. The closest pairwise charts show why:

| Direction | composition feature angle | extrapolation feature angle | composition cells passing | extrapolation cells passing |
| --- | ---: | ---: | ---: | ---: |
| 7 -> 17 | `0.0153` | `0.0840` | 2/2 | 0/2 |
| 17 -> 7 | `0.0153` | `0.0840` | 2/2 | 0/2 |
| 29 -> 53 | `0.0138` | `0.0419` | 2/2 | 0/2 |
| 53 -> 29 | `0.0138` | `0.0419` | 2/2 | 0/2 |

For `7 -> 17`, mean Fisher--Rao distance from the direct posterior grew from
approximately `0.082` on composition to `0.439` on extrapolation. For
`29 <-> 53`, composition distance was `0.073--0.081`, while extrapolation was
`0.228--0.248`. The primary extrapolation failures in these four directions
therefore persist even after restricting attention to the same sign class.

This refines the interpretation: a discrete sign ambiguity explains the large
cross-class failure, but it does not explain the support-dependent continuous
drift within the closest class-compatible pairs.

## Task and control behavior

The frozen target systems remained eligible. Averaged over the ten unique
target/cohort cells per shift, direct posterior-mean correlation was `0.998` on
composition and `0.987` on extrapolation; exact-bin accuracy was `0.721` and
`0.517`, respectively. Direct continuation replay was exact in every cell.

Across all directed cells:

| Intervention | Composition correlation | Extrapolation correlation | Mean composition Fisher distance | Mean extrapolation Fisher distance | Passing cells |
| --- | ---: | ---: | ---: | ---: | ---: |
| source feature through target scalar map | `-0.200` | `-0.197` | `1.691` | `1.759` | 8/80 |
| source scalar through target continuation | `0.198` | `0.200` | `1.539` | `1.569` | 0/80 |

The negative means reflect the sign-class mixture rather than uniformly weak
models. Shuffling source features and applying a feature half-turn strongly
damaged the successful same-chart composition cases. In cross-class cells the
primary intervention itself behaved like a wrong-sign control, so it correctly
failed the preregistered control-margin gate.

## Mechanistic interpretation

The calibrated study previously established stable *within-model* quotient
geometry. This intervention now separates that claim from cross-model
coordinate uniqueness:

```text
acquisition group respected exactly
    does not imply
one portable invariant chart
    does not imply
one portable scalar/continuation interface.
```

Architectural symmetry removed nuisance variation from the function class, but
the equivariant carrier still had a multiplicity/sign gauge and the learned
scalar interface retained support-relative scale and nonlinear conventions.
The result agrees with the earlier cross-seed residual-carrier studies: shared
representation type and shared task geometry are weaker than a shared causal
chart.

## Architectural consequence

Do not add another post-hoc alignment loss. A portable implementation must fix
the convention in the architecture itself:

1. define the sensor carrier relative to a fixed observed orientation/time
   anchor, including its sign;
2. fix channel order, normalization, and metric;
3. use a shared or analytic neutral fusion/scalar map;
4. inject the result through a fixed typed embedding, rather than allowing each
   transformer to invent a scalar convention.

The shortest learned comparison is a matched typed architecture with that
interface frozen versus a parameter-matched equivariant encoder whose final
gauge remains learned. The current experiment says merely enforcing
`E(gx)=rho(g)E(x)` again is insufficient.

## Boundaries

This is a frozen decomposition of five already-known successful checkpoints,
not independent retraining. Pair directions share source and target models.
All conclusions are conditional on the synthetic calibrated generator and the
target checkpoint's frozen downstream computation. A failed swap does not
refute within-checkpoint invariance, and a fixed shared interface has not yet
been trained in a matched architecture.

## Artifacts and reproduction

- Aggregate:
  `data/experiments/tinyllm_cross_seed_symmetry_feature_swap/20260806_d8_preregistered/campaign_results.json`
- Pair records:
  `data/experiments/tinyllm_cross_seed_symmetry_feature_swap/20260806_d8_preregistered/runs/source_*/target_*/result.json`
- Systems-only lifecycle:
  `data/experiments/tinyllm_cross_seed_symmetry_feature_swap/20260806_shakedown_cuda/`
- Runner:
  `experiments/structure_net/tinyllm_cross_seed_symmetry_feature_swap.py`
- Tests:
  `tests/structure_net/test_tinyllm_cross_seed_symmetry_feature_swap.py`
- Meta-hypothesis:
  `data/meta_hypotheses/tinyllm-cross-seed-symmetry-feature-gauge-v1.json`

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
MPLCONFIGDIR=/tmp/matplotlib-cache \
pixi run python -m \
  experiments.structure_net.tinyllm_cross_seed_symmetry_feature_swap \
  --device cuda:0 \
  --output \
  data/experiments/tinyllm_cross_seed_symmetry_feature_swap/20260806_d8_preregistered
```
