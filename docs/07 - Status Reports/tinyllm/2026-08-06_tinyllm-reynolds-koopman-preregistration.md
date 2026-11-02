# TinyLLM Reynolds–Koopman quotient-closure preregistration

**Status:** PREREGISTERED — outcomes not inspected  
**Date:** 2026-08-06  
**Hypothesis:** `tinyllm-reynolds-koopman-quotient-closure-v1`

## Question

Does the exact task-fiber barycenter become an approximately closed observable
algebra through depth, and does the resulting held-out operator-closure front
agree with the previously measured exact-orbit causal front? A secondary question
is whether quadratic cover invariants explain degree two earlier than cubic cover
invariants explain degree three.

## Frozen sources and independence

Reuse without retraining the ten retained d6 degree-ladder checkpoints for
`k=2,3` and seeds `7,17,29,41,53`. Validate source schemas, model-state digests,
checkpoint hashes, and exact full-fiber construction. The prior deck-action
campaign supplies a frozen causal-front comparator only; none of its Procrustes
actions or projectors enter the Koopman features.

Fit on an interpolation cohort of 384 exact nuisance-matched orbits. Freeze all
PCA bases, random character sketches, standardizers, ridge models, kernel
features, and nonlinear controls before evaluation on disjoint 192-orbit
composition and extrapolation cohorts. Use a separate fixed-nuisance 192-member
map cohort for winding and intervention diagnostics.

## States and cuts

At the frontend, block-0 pre-attention, every post-attention and post-MLP cut in
the three-block model, and full depth, retain the complete state needed for
downstream continuation. For residual cuts this is the full three-token residual
sequence, not only its query token.

For an exact orbit tuple `H=(h_0,...,h_(k-1))`, define

`b = mean_j h_j`, `delta_j = h_j-b`, and
`c_r = mean_j exp(-2 pi i r j/k) h_j`.

The frontend is included in remaining-task closure but excluded from autonomous
one-step closure because reconstructing the residual sequence also requires input
tokens and positions. Full depth has no outgoing one-step transition.

## Frozen observable dictionaries

Fit three nested linear dictionaries:

- `B`: intercept plus a 48-dimensional PCA chart of the barycenter;
- `B+Q`: `B` plus 24-sketch low-rank invariant quadratic character products;
- `B+Q+C`: `B+Q` plus invariant cubic products for `k=3`, or barycenter-modulated
  quadratic products for `k=2` so no non-invariant odd character is introduced.

All cover sketches use a seed-frozen Gaussian map. `k=2` quadratic coordinates
include squared and adjacent-pair products of the nontrivial real character.
`k=3` quadratic coordinates include squared magnitude and adjacent Hermitian
products; cubic coordinates include real and imaginary character cubes.

Do not materialize full residual tensor powers. Fit ordinary ridge regression in
each supplied dictionary; do not interpret fitted eigenvalues as a stationary
depth operator.

## Endpoints

At every eligible transition, predict the next barycenter in a train-fitted
48-dimensional target PCA chart. Report held-out variance-weighted `R2` for all
three dictionaries and incremental positive cover gain

`Delta_cover = max(0, R2_(B+Q+C) - R2_B)`.

At every cut, predict the frozen model's final centered log-posterior and circular
moment. Report moment `R2`, predicted-posterior exact-bin accuracy, circular
alignment, winding degree on the fixed map, and the same cover gain.

The **task-closure front** for one seed and regime is the earliest cut satisfying:

1. barycenter-only moment `R2 >= 0.90`;
2. map alignment `>= 0.90`, resolved winding within `0.10` of degree `k`, and
   predicted task accuracy no more than `0.05` below the actual frozen output;
3. positive cover gain `<= 0.02`.

The **autonomous one-step closure gate** is barycenter-only next-state `R2 >= 0.90`
with cover gain `<= 0.02`. It is reported separately and is not silently inferred
from final-task sufficiency.

Cover gain is substantial when it is at least `0.05`. A seed supports the proposed
transition only if at least one pre-causal-front cut has substantial task cover
gain and every cut from its task-closure front onward has gain at most `0.02` on
both held-out shifts.

## Cover-scaling intervention

At every cut form full-sequence states `b + lambda delta` and continue the frozen
network. Fit lifted response models on `lambda in {0, 0.5, 1}` and evaluate without
refitting on `{0.25, 0.75, 1.25}`. The intervention gate requires moment `R2 >=
0.80` and mean predicted-versus-actual circular cosine `>=0.90` for every unseen
lambda on both shifts at the declared closure front.

On fixed-nuisance maps, fit a complex cubic polynomial to the actual degree-`k`
harmonic coefficient versus lambda. Quadratic dominance for `k=2` and cubic
dominance for `k=3` are secondary mechanistic endpoints, not requirements that
can override a failed closure gate.

## Primary campaign gates

The full hypothesis is confirmed only if all conditions hold:

1. the task-closure front agrees with the frozen orbit-averaging causal front
   within one adjacent cut on both shifts in at least four of five seeds for each
   degree;
2. the task cover-gain transition passes in at least four of five seeds per degree;
3. the autonomous one-step gate passes at the task-closure front or its immediately
   preceding transition in at least four of five seeds per degree;
4. the unseen-lambda intervention gate passes in at least four of five seeds per
   degree;
5. the median `k=3` task-closure front occurs no earlier than the median `k=2`
   front.

## Controls and claim boundaries

Evaluate shuffled orbit membership, phase-shuffled character modes, a random
same-size feature dictionary, a barycenter-only nonlinear MLP, and generic random
Fourier-feature kernel ridge. Cross-seed generalization means independently
passing the frozen gates in held-out seeds; hidden-coordinate weights are not
transferred across seeds because the prior experiment rejected a shared linear
coordinate gauge.

This experiment can establish approximate predictive closure for declared task
observables on two operational shifts. It cannot prove exact finite-dimensional
Koopman invariance, a stationary operator across depth, a complete linearization
of the transformer, or global extrapolation beyond the declared generator.

## Append-only artifacts

The confirmatory root is
`data/experiments/tinyllm_reynolds_koopman/20260806_d6_preregistered`.
Shakedowns and amendments use separate roots.
