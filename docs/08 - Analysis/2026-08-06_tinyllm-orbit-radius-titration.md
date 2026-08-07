# TinyLLM causal orbit-radius titration

**Status:** PRIMARY GATE PASSED; ROBUST STABILITY NOT REPLICATED — EARLY FRONTS STABLE, LATER FRONTS COHORT-SENSITIVE  
**Date:** 2026-08-06  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, with pre-outcome device Amendment A  
**Hypothesis:** `tinyllm-causal-orbit-radius-threshold-v1`  
**Preregistration:** `../07 - Status Reports/2026-08-06_tinyllm-orbit-radius-titration-preregistration.md`

## Verdict

This campaign passed its preregistered degree-two gate. On its fresh exact deck
orbits, all five frozen models failed at zero orbit radius, passed at full
radius, and crossed the task gate exactly once under both composition and
extrapolation. Four of five seeds localized the two shift thresholds within the
preregistered `0.125` tolerance.

That result does **not** establish a checkpoint-global stable threshold. A
contemporaneous independently generated C2 fusion-radius campaign evaluated the
same five checkpoints and frozen cuts on a different fresh 64-orbit cohort,
with transplanted-character and random-direction controls. It found stable
onsets in only 3/5 seeds. Seed 17 lost the composition endpoint, and seed 41's
onsets differed by `0.50` across shifts. The meta-level stability claim is
therefore not independently replicated.

The cross-campaign result is sharper: the three block-0-attention fronts
(seeds 7, 29, and 53) were monotone and shift-stable in both cohorts. The two
later fronts (seeds 17 and 41) were cohort-sensitive. The controlled C2 study
also found that neither transplanted nor norm-matched random directions
reproduced the effect in any seed. Exact group direction is causal and specific;
support stability depends on the model's synthesis-front regime.

The causal mechanism is simpler than its full residual geometry suggests. By
radius `0.5`, the partial exact defect had median cosine `0.987` to the final
defect and recovered about `0.93` of its downstream Fisher effect. The linear
chord reached the causal gate within one grid step of the exact partial-orbit
path in all ten degree-two seed/shift cells; the quadratic chord did so in only
six. This supports a fixed task-effective direction whose magnitude accumulates
nonlinearly until a decoder threshold is crossed. It does not support the prior
universal quadratic Taylor account.

Degree three supplied the predicted contrast. Only 2/5 seeds replicated both
endpoints, only 2/5 had a single crossing under both shifts, and no seed had a
shift-stable threshold. Its quotient synthesis remains support-relative and
can already be sufficient at zero radius or fail even at the full observed
radius on fresh composition orbits.

## Campaign integrity

The campaign reused ten frozen d6 checkpoints and their frozen character-
coupling comparators. It performed no training, fitting, probe selection, or
parameter changes. Every cell validated checkpoint and comparator SHA-256,
used a distinct scientific fingerprint, regenerated 64 exact nuisance-matched
orbits per shift, and evaluated the same nine-point radius grid.

The intended CUDA shakedown failed before producing a result because the
execution environment exposed no NVIDIA device. Amendment A was recorded
before any 64-orbit primary outcome was executed or inspected. The unchanged
campaign then ran on CPU. A separate eight-orbit lifecycle validated execution
but was under-resolved for the topology endpoint and is not pooled as evidence.

| Item | Value |
| --- | --- |
| requested / completed / failed | 10 / 10 / 0 |
| frozen checkpoints | 10; five each for `k=2,3` |
| training runs | 0 |
| new exact orbits | 64 per shift and cell |
| radii | `0, 0.125, ..., 1.0` |
| paths | exact partial orbit; linear chord; quadratic chord |
| environment | CPU, PyTorch 2.5.1+cu121, Python 3.11.13 |
| analysis time | 269.1 s total; 19.4--32.9 s per cell |
| implementation SHA-256 | `8595427ac1edb8e7d11d70731ec5001210756abc4a688903f31328f39de2afed` |
| campaign SHA-256 | `9884c0ee31aba28bf41118bd8e5e8d4d3932260cbd52af0ea74bf308b7fc1517` |

## Preregistered gates

Each primary gate required at least four of five degree-two seeds, jointly
within seed across composition and extrapolation.

| Gate | `k=2` | Required | Verdict |
| --- | ---: | ---: | --- |
| zero radius fails and full radius passes | **5/5** | 4/5 | pass |
| exact curve crosses once under both shifts | **5/5** | 4/5 | pass |
| critical radii differ by at most `0.125` | **4/5** | 4/5 | pass |

The preregistered gate passed under this campaign's operational definition.
Seed 17 was the only shift-stability miss: its composition threshold was `0.50`
and its extrapolation threshold was `0.25`. The independent-cohort result below
prevents promoting that pass to a general stable-threshold claim.

## Independent-cohort reconciliation

The C2 fusion-radius campaign used evaluation seed offset `+9101`; this campaign
used `+4501`. Both used 64 exact orbits per shift, the same checkpoints, frozen
synthesis cuts, and causal output conjunction. The C2 campaign additionally
included two direction controls and an off-manifold radius; its primary radius
grid omitted `0.625` and `0.875`.

| Seed | This campaign comp. / extrap. | Independent C2 comp. / extrap. | Cross-cohort verdict |
| ---: | --- | --- | --- |
| 7 | 0.750 / 0.625 | 0.750 / 0.750 | stable early front |
| 17 | 0.500 / 0.250 | missing / 0.125 | later front is cohort-sensitive |
| 29 | 0.625 / 0.500 | 0.750 / 0.750 | stable early front |
| 41 | 0.750 / 0.750 | 1.000 / 0.500 | later front is cohort-sensitive |
| 53 | 0.625 / 0.500 | 0.750 / 0.750 | stable early front |

Only 3/5 checkpoints support a robust shift-stable onset across both fresh
cohorts. This is not a post-hoc rescue of either preregistration: the current
campaign's gate remains a pass, the C2 depth-locality hypothesis remains
failed, and the combined interpretation is explicitly narrower than either
standalone design.

## Degree-two causal thresholds

| Seed | Frozen synthesis depth | Composition `r*` | Extrapolation `r*` | Shift-stable |
| ---: | ---: | ---: | ---: | --- |
| 7 | 1 | 0.750 | 0.625 | yes |
| 17 | 5 | 0.500 | 0.250 | no |
| 29 | 1 | 0.625 | 0.500 | yes |
| 41 | 3 | 0.750 | 0.750 | yes |
| 53 | 1 | 0.625 | 0.500 | yes |

Mean critical radius was `0.650` on composition and `0.525` on extrapolation;
medians were `0.625` and `0.500`. Thresholds did not order monotonically by
synthesis depth: the latest front, seed 17 at residual sublayer five, had the
smallest radii. Front location and required orbit amplitude are therefore
distinct mechanistic coordinates.

## Defect-path geometry

The table reports five-seed medians for the exact partial-orbit defect. Fisher
effect is measured against the propagated `r=0` and exact `r=1` downstream
posteriors. Errors are normalized by squared full-defect norm.

| Shift | Radius | Norm / full | Cosine to full | Error to linear chord | Error to quadratic chord | Fisher effect recovered |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| composition | 0.25 | 0.178 | 0.957 | 0.009 | 0.016 | 0.200 |
| composition | 0.50 | 0.516 | 0.987 | 0.025 | 0.096 | 0.932 |
| composition | 0.75 | 0.800 | 0.998 | 0.014 | 0.075 | 0.995 |
| extrapolation | 0.25 | 0.178 | 0.955 | 0.009 | 0.018 | 0.196 |
| extrapolation | 0.50 | 0.515 | 0.986 | 0.025 | 0.102 | 0.929 |
| extrapolation | 0.75 | 0.799 | 0.998 | 0.014 | 0.074 | 0.995 |

The near-identical curves under both shifts are stronger than a correlation
between independently fitted probes: the same frozen sublayer, exact group
orbit, repeated barycenter patch, and unchanged downstream network generate the
effect. At the causal crossing, median recovered Fisher effect was `0.986` for
composition and `0.968` for extrapolation, although the range was broad
(`0.569--0.997` and `0.193--0.988`). The task gate can therefore flip before
most Fisher distance is recovered in some cells.

All ten exact-versus-linear critical radii differed by at most one grid step.
The quadratic chord was within one step in only six of ten cells and usually
required a larger radius. The observed defect direction stabilizes early, but
its amplitude is not governed by one universal `r^2` law.

## Degree-three diagnostic

| Seed | Composition `r*` | Extrapolation `r*` | Endpoint replication | Shift-stable |
| ---: | ---: | ---: | --- | --- |
| 7 | 0.000 | 0.750 | no: composition already passes at zero | no |
| 17 | 0.000 | 0.750 | no: composition already passes at zero | no |
| 29 | 0.750 | 0.500 | yes | no |
| 41 | 1.000 | 0.750 | yes | no |
| 53 | none | 1.000 | no: composition fails at full radius | no |

This is a direct out-of-sample failure of one global radial-threshold account
for degree three. It agrees with the earlier finding that degree-three
sufficiency can be destroyed and resynthesized across depth, but the current
result is stronger in one respect: instability appears at a fixed previously
measured synthesis cut under newly generated exact orbits.

## Interpretation and boundaries

The shortest cross-campaign-supported mechanism is:

```text
branch-bearing degree-two cover variation at an early block-0 front
    -> a nearly fixed invariant defect direction
    -> nonlinear accumulation with orbit radius
    -> a repeatable downstream task-sufficiency threshold.

later degree-two fronts
    -> the same direction-specific synthesis phenomenon
    -> cohort- and shift-sensitive endpoint/onset.
```

This is a causal task-level statement, not merely tested decodability. Removing
the group-orbit variation at `r=0` makes the frozen continuation fail; restoring
enough of the exact orbit computation makes it pass, without changing weights
or supplying a target label. The independent controls show that generic
norm-matched directions do not substitute for the exact group character.

The result does not establish a global Hessian, certify behavior away from the
observed deck directions, or show that the residual state lies on a literal
one-dimensional manifold. The causal gate is decoder-conditioned. Repeating
one patch across each tested fiber makes within-orbit branch chance exact but
does not prove global branch absence. Degree-three findings are secondary and
cannot strengthen the within-campaign degree-two gate pass.

## Artifacts and reproduction

Primary aggregate:
`data/experiments/tinyllm_orbit_radius_titration/20260806_d6_preregistered/campaign_results.json`

Independent C2 aggregate:
`data/experiments/tinyllm_c2_fusion_radius/20260806_d6_preregistered/campaign_results.json`

Independent C2 report:
`docs/08 - Analysis/2026-08-06_tinyllm-c2-fusion-radius.md`

Per-cell records:
`data/experiments/tinyllm_orbit_radius_titration/20260806_d6_preregistered/runs/k{2,3}/seed_<seed>/result.json`

Disposable lifecycle:
`data/experiments/tinyllm_orbit_radius_titration/20260806_shakedown_cpu/`

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
pixi run python -m experiments.structure_net.tinyllm_orbit_radius_titration \
  --output data/experiments/tinyllm_orbit_radius_titration/20260806_d6_preregistered \
  --device cpu
```
