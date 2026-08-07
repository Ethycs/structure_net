# TinyLLM deck-action descrambler preregistration

**Status:** PREREGISTERED — outcomes not inspected  
**Date:** 2026-08-06  
**Hypothesis:** `tinyllm-deck-action-carrier-cover-v1`

## Question

Do the retained degree-`k` TinyLLM models organize their hidden carrier as an
approximate representation of the deck group `Z_k`, with quotient-invariant and
branch-sensitive information occupying separable isotypic components? If so, is
the branch-bearing component redundant, or causally required to compute the
degree-`k` task map?

## Frozen sources

Reuse, without retraining, all 15 d6 checkpoints from
`data/experiments/tinyllm_degree_k_ladder/20260806_d6_preregistered`: degrees
`k=1,2,3` and seeds `7,17,29,41,53`. Validate source schema, condition, seed,
checkpoint paths, and final model-state digest before analysis. `k=1` is the
trivial-action control; deck-localization gates apply to `k=2,3`.

## Exact cohorts

Construct complete nuisance-matched deck orbits. For a quotient coordinate
`theta`, use phases `(theta + 2 pi j)/k`, `j=0,...,k-1`, with identical nuisance,
direction, calibration, and observation-generator randomness within each orbit.
Fit only on an interpolation cohort. Freeze every fitted action, projector,
normalizer, and probe before evaluation on disjoint composition and extrapolation
cohorts.

## Cuts

Retain the full residual sequence for causal continuation and the query residual
for geometry at:

- analytic carrier output;
- block-0 pre-attention, post-attention, and post-MLP;
- block-1 post-attention and post-MLP;
- block-2 post-attention and post-MLP;
- full depth.

## Deck action and canonical decomposition

For row-oriented activation matrices fit `R` by orthogonal Procrustes,
`min ||H R - H_g||_F`. Report target-normalized error, mean paired cosine,
improvement over the identity transport, full-matrix closure, and activation-
weighted closure `||H(R^k-I)||/||H||`.

Use a real Schur decomposition and eigenvalue distance to the `k`-th roots of
unity to report invariant and nontrivial multiplicities. Use the declared group
average

`Pi_inv = (I + R + ... + R^(k-1))/k`

and `Pi_branch = I-Pi_inv`. Report projector idempotence, component energies, and
conditioned linear task/branch probes. Orthogonal descrambling is never interpreted
as information removal.

## Causal interventions

At each cut patch the exact full-sequence orbit mean into the frozen downstream
network. Compare with:

- an isotropic random projector with the same numerical rank as `Pi_inv`;
- random cross-fiber orbit pairing;
- phase-shuffled averaging;
- an isotropic equal-norm activation perturbation.

Measure output circular alignment, winding degree, sampling resolution, exact-bin
accuracy, branch decodability of the patched query activation, and posterior
Fisher `H1`. Original frozen output is the baseline.

Classify exact orbit averaging per seed and cut as:

- **preserved:** alignment at least `0.90`, resolved degree within `0.10` of `k`,
  and task-accuracy loss no more than `0.03`;
- **destroyed:** alignment below `0.50`, resolved degree differs from `k`, or
  task-accuracy loss exceeds `0.20`;
- **partial/unresolved:** neither condition.

The causal mechanism is reproducible when the same classification holds on both
composition and extrapolation in at least four of five seeds. Preservation means
redundant co-representation; destruction means a computational cover. A later
preservation front localizes quotient construction.

## Preregistered deck gates

For `k=2,3`, evaluate every residual cut and declare a cut stable when at least
four of five seeds pass both held-out shifts:

1. target-normalized deck-transport error `<=0.15`, mean paired cosine `>=0.95`,
   and at least 50% of identity-transport squared error removed;
2. activation-weighted group-closure error `<=0.10`;
3. branch-component conditioned linear accuracy is at least chance plus `0.20`
   and within `0.05` of the full-activation probe;
4. invariant-component conditioned branch accuracy is at most chance plus `0.05`;
5. exact orbit averaging receives a reproducible preserved or destroyed causal
   classification.

The full deck-action hypothesis is confirmed only if at least one residual cut
for each of `k=2` and `k=3` passes gates 1–4 and receives a reproducible causal
classification. Per-cut failures remain measurements, not reasons to retune gates.

## Cross-seed atlas and boundaries

Compare root-of-unity multiplicities, invariant/branch energy fractions, task-
decoder energy allocation, and attention/MLP changes across seeds. Report
nonuniqueness where Schur blocks have repeated eigenvalues; do not treat arbitrary
coordinates within an isotypic block as aligned neurons.

This study tests one orthogonal linear deck action, one family of conditioned
linear probes, and activation patching on synthetic exact orbits. It does not
erase information, establish minimal representations, or certify every nonlinear
group action.

## Append-only artifacts

The confirmatory root is
`data/experiments/tinyllm_deck_action_descrambler/20260806_d6_preregistered`.
Shakedowns and any amended analyses use separate roots.
