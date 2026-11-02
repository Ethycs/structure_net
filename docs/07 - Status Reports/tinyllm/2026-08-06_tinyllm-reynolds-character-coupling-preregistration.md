# TinyLLM Reynolds character-coupling causal decomposition preregistration

**Status:** PREREGISTERED — outcomes not inspected  
**Date:** 2026-08-06  
**Hypothesis:** `tinyllm-reynolds-character-coupling-synthesis-v1`

## Question

Does the exact causal quotient front mark the first residual sublayer that
synthesizes a quotient-sufficient invariant from branch-bearing deck-character
modes? Does the symmetry-neutral quadratic Taylor contribution explain most of
the downstream task effect of that synthesis?

## Frozen sources and replication unit

Reuse without retraining the ten retained d6 degree-ladder checkpoints for
`k=2,3` and seeds `7,17,29,41,53`. Validate model-state, checkpoint, frozen
deck-action comparator, and frozen Reynolds–Koopman comparator digests. Analyze
composition and extrapolation separately on 64 new exact nuisance-matched
orbits per cell. One checkpoint/seed, including all twelve attention/MLP
residual sublayers in blocks 0--5 and
both shifts, is the independent replication unit.

The known causal front is the earliest cut where the prior exact orbit-average
patch was classified as preserved. No new fitted observer defines that front.

## Exact sublayer quantities

For every attention and MLP residual sublayer `F_l`, reshape its source and
target states into exact deck fibers `h_l,j` and `F_l(h_l,j)`. Define per orbit:

`b_l = mean_j h_l,j`

`a_(l+1) = F_l(b_l)`

`b_(l+1) = mean_j F_l(h_l,j)`

`chi_l = b_(l+1) - a_(l+1)`.

Patch both `a_(l+1)` and `b_(l+1)` at the target cut and continue through the
unchanged frozen model. Every patched state is repeated identically across the
fiber, so conditional branch accuracy is exactly chance under the declared
within-orbit evaluation.

Classify each patch with the frozen deck causal thresholds: circular alignment
at least `0.90`, resolved sampling, winding degree within `0.10` of `k`, and
exact-bin accuracy loss no more than `0.03` relative to the untouched model.
The resulting sublayer regime is:

| propagated `a` | actual `b` | classification |
| --- | --- | --- |
| fail | fail | `cover_required_after_sublayer` |
| fail | pass | `invariant_synthesis` |
| pass | pass | `quotient_already_closed` |
| pass | fail | `quotient_corruption` |

The synthesis front is the first `invariant_synthesis` target cut.

## Character-neutral Taylor decomposition

At each source cut compute the exact deck Fourier components

`c_r = mean_j exp(-2 pi i r j/k) h_j`.

Report their energy fractions and reconstruction error. The primary
approximation uses the group-averaged directional Hessian

`chi2 = (1/(2k)) sum_j D2 F_l(b)[delta_j, delta_j]`,

where `delta_j=h_j-b`. Estimate each directional Hessian by a symmetric finite
difference with frozen scale `eta=0.25`:

`D2 F(b)[d,d] ~= [F(b+eta d)-2F(b)+F(b-eta d)]/eta^2`.

This group average retains only charge-neutral character combinations. For
`k=2` its lowest term is the `1+1=0 mod 2` coupling. For real `k=3` carriers it
includes the allowed `1+2=0 mod 3` quadratic coupling; cubic dominance is not a
prediction.

Also compute, as a secondary diagnostic,

`chi3 = (1/(6k)) sum_j D3 F_l(b)[delta_j,delta_j,delta_j]`

with the centered four-point stencil at `eta` and evaluate `chi2+chi3`. The
cubic result is interpreted only where the quadratic primary endpoint fails.

## Approximation endpoints

Patch `a+chi2` and `a+chi2+chi3` at the target cut. Report:

- residual explained fraction `1-||chi-approx||^2/||chi||^2`;
- cosine alignment with the exact defect;
- downstream Fisher-effect explained fraction
  `1-d_FR^2(D(a+approx),D(b))/d_FR^2(D(a),D(b))`;
- degree, circular alignment, exact-bin accuracy, and branch chance.

The primary quadratic endpoint passes when its downstream Fisher-effect
explained fraction is at least `0.70` at the exact synthesis front. Negative
explained fractions remain untruncated. A denominator below `1e-6` is declared
task-effect degenerate and cannot pass.

## Controls

Run the complete causal and quadratic analysis for two deterministic controls:

1. **shuffled orbit membership:** independently permute every non-reference
   sheet across quotient-phase orbits while keeping each sheet tensor intact;
2. **matched random directions:** replace the exact centered sheet deviations
   with zero-mean Gaussian directions, scaled per orbit to the same total
   residual norm.

A control reproduces the mechanism if it produces `invariant_synthesis` within
one sublayer of the exact synthesis front and reaches `0.70` downstream Fisher
effect explained there. Specificity requires neither control to reproduce it on
either shift.

## Primary gates

Require each gate in at least four of five seeds, separately for `k=2` and
`k=3`:

1. **Shift-stable causal regime:** the twelve exact four-regime classifications
   are identical under composition and extrapolation.
2. **Causal-front localization:** on both shifts, an exact synthesis front
   exists and its target cut lies within one residual sublayer of the frozen
   causal quotient front.
3. **Neutral quadratic sufficiency:** on both shifts, the quadratic contribution
   explains at least `0.70` of the downstream Fisher effect at the synthesis
   front and the effect is nondegenerate.
4. **Control specificity:** neither shuffled membership nor matched random
   directions reproduces the synthesis mechanism on either shift.

The full hypothesis is confirmed only if all four gates pass for both degrees.
No degree-ordering or cubic-dominance claim is part of the hypothesis.

## Outcome meanings

- Stable synthesis plus quadratic sufficiency supports a character-coupling
  mechanism for the causal quotient front.
- Stable synthesis with poor quadratic fit motivates the declared cubic and
  higher-order diagnostics without changing the primary result.
- Barycenter patches pass without a preceding synthesis regime means quotient
  dynamics was already closed at the inspected source cut.
- Shift-unstable regimes imply quotient synthesis is support-relative.
- Control reproduction means the effect is generic nonlinear Jensen curvature,
  not deck-character-specific invariant synthesis.

## Boundaries

Finite differences estimate local derivatives along the observed exact modes;
they are not interval certificates. The downstream Fisher endpoint measures
task-posterior effect, not equality of residual tensors. Exact within-orbit
branch chance follows from repeated patches and does not establish global
absence of branch information in arbitrary off-fiber states.

## Artifacts

The confirmatory root is
`data/experiments/tinyllm_reynolds_character_coupling/20260806_d6_preregistered`.
Shakedowns use a separate root.
