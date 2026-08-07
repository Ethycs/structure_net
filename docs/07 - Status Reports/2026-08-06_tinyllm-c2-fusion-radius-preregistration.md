# TinyLLM degree-two character-fusion radius preregistration

**Status:** PREREGISTERED — PRIMARY OUTCOMES NOT INSPECTED  
**Date:** 2026-08-06  
**Profile:** NAL-STD-EXPERIMENT `PREREGISTERED`  
**Hypothesis:** `tinyllm-c2-character-fusion-radius-v1`  
**Schema:** `nal.tinyllm-c2-character-fusion-radius.v1`

## Question and prediction

At the frozen degree-two causal quotient front, is invariant synthesis a local,
monotone, shift-stable response to the amplitude of the populated nontrivial
deck character?

The preceding Reynolds character-coupling campaign established that the exact
degree-two synthesis front matches the independently measured causal quotient
front in all five seeds and both shifts. It also found a preregistered split:
seeds 7, 29, and 53 synthesize at block-0 attention and have a strong local
quadratic task effect, while seeds 17 and 41 synthesize later and do not. This
new campaign tests a prediction derived from that observed split on fresh exact
orbits:

- the three early-front seeds have causal onset radius at most `0.50`;
- the two later-front seeds require radius at least `0.75`;
- onset differs by at most `0.25` between composition and extrapolation;
- the exact radial response is causally monotone through the observed radius;
- matched non-character controls do not approximate the exact full-radius task
  effect at the exact onset.

Confirmation would support a depth-dependent locality claim. Failure of the
late-radius prediction would favor a poor local Taylor estimator over a truly
nonlocal mechanism. Shift instability would instead make the radial mechanism
support-relative.

## Frozen sources and replication unit

Reuse without retraining the five retained d6, degree-two checkpoints for seeds
`7,17,29,41,53` from:

`data/experiments/tinyllm_degree_k_ladder/20260806_d6_preregistered`.

The frozen synthesis cuts and checkpoint identities come from:

`data/experiments/tinyllm_reynolds_character_coupling/20260806_d6_preregistered`.

One independently trained checkpoint/seed is the replication unit. Source
checkpoint SHA-256, source result SHA-256, schema, degree, seed, and synthesis
cut must validate before analysis. Composition and outside-range extrapolation
are analyzed separately on 64 fresh exact nuisance-matched orbits per seed and
regime. No source weights, heads, probes, or thresholds are refit.

## Exact radial intervention

At the residual cut immediately before the frozen synthesis sublayer, write an
exact `C2` orbit as

```text
h_plus  = b + delta
h_minus = b - delta
b       = (h_plus + h_minus) / 2.
```

For the frozen attention or MLP residual sublayer `F`, define the even Reynolds
response at radius `s`:

```text
chi(s) = (F(b + s delta) + F(b - s delta)) / 2 - F(b).
```

Patch `F(b) + chi(s)` at the target cut, repeat that state across both fiber
members, and continue through the unchanged model. The fixed radius grid is:

```text
0, 0.125, 0.25, 0.375, 0.50, 0.75, 1.00, 1.25.
```

Radius `1.00` is the observed exact orbit. Radius `1.25` is a secondary
off-manifold stress test and cannot define primary onset. The primary onset is
the smallest radius in `0.125,...,1.00` whose repeated patch passes all frozen
deck-causal thresholds:

- circular alignment at least `0.90`;
- resolved sampling;
- winding degree within `0.10` of degree two;
- exact-bin accuracy loss no more than `0.03` from the untouched model.

If no primary radius passes, onset is declared missing and every onset gate
fails for that regime.

## Controls

At the same barycenters and with the same per-orbit deviation norms, compute two
deterministic controls:

1. **cross-orbit character transplant:** permute `delta` across quotient-phase
   orbits before forming the symmetric pair;
2. **matched random symmetric direction:** replace `delta` by a Gaussian
   direction with the same per-orbit norm and exact `(+d,-d)` symmetry.

Control patches are evaluated against the exact full-radius posterior, not
against a control-specific target. A control reproduces the mechanism when, at
or below the exact onset, it explains at least `0.70` of the downstream
Fisher--Rao effect of the exact radius-one response. Sheet exchange
`delta -> -delta` is also checked as an exact numerical invariance contract.

## Primary gates

All quantities are decided within seed before aggregation.

1. **Early-front locality:** seeds 7, 29, and 53 have onset at most `0.50` on
   both shifts.
2. **Later-front finite radius:** seeds 17 and 41 have onset at least `0.75` on
   both shifts.
3. **Shift-stable onset:** at least four of five seeds have nonmissing
   composition/extrapolation onsets differing by at most `0.25`.
4. **Monotone causal response:** in at least four of five seeds on both shifts,
   every primary-grid radius after the first passing radius also passes.
5. **Control specificity:** in at least four of five seeds, neither control
   reproduces the mechanism on either shift.
6. **Exchange invariance:** maximum posterior difference after sheet exchange
   is at most `1e-6` in every seed and shift.

The full hypothesis is confirmed only if all six gates pass. The early- and
later-front gates are exact cohort predictions rather than four-of-five gates
because the cohorts were fixed by the predecessor result.

## Secondary measurements

Report for every radius and condition:

- causal pass/fail and all component task diagnostics;
- Fisher--Rao effect explained relative to the exact radius-one posterior;
- residual norm of `chi(s)` and cosine to `chi(1)`;
- local log slope between adjacent nonzero radii;
- behavior at radius `1.25`.

These measurements explain a failed gate but cannot rescue it.

## Outcome meanings

| Outcome | Interpretation | Next action |
| --- | --- | --- |
| all gates pass | early fronts are locally quadratic-like while later fronts require finite character amplitude | use explicit invariant tensor fusion for early fronts and gated/orbit aggregation for later fronts |
| late seeds also turn on locally | the earlier Taylor failure was estimator- or task-direction-specific, not evidence of nonlocal synthesis | improve derivative estimation before changing architecture |
| onset is nonmonotone | the quotient is created and destroyed along the character ray | avoid a one-shot pooling layer; preserve typed carriers through depth |
| composition/extrapolation onsets diverge | the causal symmetry mechanism remains support-relative | repair the sensor/group representation before adding fusion capacity |
| controls reproduce the effect | the response is generic curvature, not character-specific fusion | reject the group-mechanism interpretation |
| exact radius one fails on fresh orbits | predecessor front does not replicate | audit generator/provenance before further mechanism claims |

## Boundaries

The intervention follows one observed residual-space character ray and does not
identify a global group representation. Repeated patches make within-orbit
sheet identity exactly constant; they do not prove global absence of branch
information. Fisher--Rao effects are conditioned on the frozen decoder. Radius
`1.25` is an artificial residual intervention. The early/late cohorts were
derived from a prior result and are tested here on fresh generated orbits, not
new independently trained models.

## Artifacts and execution

Primary root:

`data/experiments/tinyllm_c2_fusion_radius/20260806_d6_preregistered`

Planned entry point:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
pixi run python -m experiments.structure_net.tinyllm_c2_fusion_radius \
  --output data/experiments/tinyllm_c2_fusion_radius/20260806_d6_preregistered \
  --device cuda:0
```

Focused contract tests and a disposable eight-orbit shakedown must pass before
the five-seed campaign. Shakedown outcomes are systems-only and are not pooled.
