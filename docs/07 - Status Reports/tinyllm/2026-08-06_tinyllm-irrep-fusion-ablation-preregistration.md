# TinyLLM deck-irrep fusion ablation preregistration

**Status:** PREREGISTERED — outcomes not inspected  
**Date:** 2026-08-06  
**Hypothesis:** `tinyllm-deck-irrep-fusion-ablation-v1`

## Question

At a previously frozen causal synthesis front, are the nontrivial deck-character
modes causally necessary and orbit-specific? For degree three, does the
quotient-sufficient invariant depend on finite-`C3` phase invariants beyond the
quadratic radial invariant `|c1|^2`?

This is a checkpoint-only diagnostic. It does not fit a probe, choose a new
front, or retrain a model.

## Frozen sources and replication unit

Reuse the ten retained d6 checkpoints for `k=2,3` and seeds
`7,17,29,41,53`. For each composition and extrapolation cell, use the first
`invariant_synthesis` transition recorded by the completed Reynolds
character-coupling campaign. Validate its result digest and checkpoint digest.

Regenerate the same 64 exact nuisance-matched orbits used by that campaign.
One checkpoint/seed, containing both shifts, is the independent replication
unit. No outcome from this study may move the frozen front.

## Exact character intervention

At the source of the frozen sublayer `F`, write the exact orbit states as

`h_j = b + delta_j = sum_r c_r exp(2 pi i r j/k)`.

The propagated and exact next barycenters are

`a = F(b)` and `q = mean_j F(h_j)`.

Apply three intervention families before `F`, average its outputs, patch the
result at the target cut, and run the unchanged frozen continuation.

### 1. Charged-mode amplitude path

For the frozen grid `alpha in {0, 0.25, 0.5, 0.75, 1}`, use

`h_j(alpha) = b + alpha delta_j`.

`alpha=0` removes every nontrivial irrep, whereas `alpha=1` restores the exact
orbit. Record the first amplitude whose patched barycenter passes the frozen
causal gate.

As a non-derivative quadratic-homogeneity diagnostic, compare the exact
`q(alpha)` to

`q2(alpha) = a + alpha^2 (q(1)-a)`.

This diagnostic asks whether the full-radius defect behaves quadratically; it
does not estimate a Hessian.

### 2. Orbit-matched carrier substitution

Keep each orbit's barycenter, but cyclically move the complete centered irrep
carrier from the next orbit and rescale it to the recipient carrier norm. This
preserves exact zero mean, the deck-sheet organization, and per-orbit carrier
norm while breaking the relationship between the invariant base and its
charged carrier.

The control reproduces synthesis only if its patch passes the frozen causal
gate and preserves at least `0.70` of the exact downstream Fisher effect.

### 3. `C3` carrier-phase intervention

For `k=3`, rotate the conjugate character pair by

`c1 -> exp(i theta)c1`, `c2 -> exp(-i theta)c2`

on the frozen twelve-point grid `theta=2 pi m/12`. The reconstructed sheets
remain real and retain their barycenter and character energies.

The angles `0, 2 pi/3, 4 pi/3` are exact deck rotations and must leave the
averaged sublayer output invariant. The remaining angles preserve the radial
quadratic invariant `c1 c2=|c1|^2`, but change finite-`C3` invariants such as
`c1^3+c2^3`.

Define normalized continuous-phase sensitivity as the median Fisher--Rao
distance from the unrotated posterior over non-deck angles, divided by the
Fisher--Rao effect between the `alpha=0` and exact `alpha=1` patches.

Classify each shift as:

- `radial`: sensitivity at most `0.10`;
- `mixed`: sensitivity between `0.10` and `0.25`;
- `finite_group_phase_sensitive`: sensitivity at least `0.25`.

Values on a causal-effect denominator below `1e-6` are degenerate and cannot
support a mechanistic classification.

For `k=2`, the only real carrier phases are signs. Sign reversal is the exact
deck swap and is retained as a numerical group-contract check rather than a
continuous-phase experiment.

## Frozen causal and task endpoints

Use the same causal gate as the source campaign: circular alignment at least
`0.90`, resolved sampling, winding degree within `0.10` of `k`, and exact-bin
accuracy loss no greater than `0.03` relative to the untouched model.

For an intervention posterior `p` define exact-effect preservation as

`1 - d_FR^2(p,q) / d_FR^2(a,q)`.

Report values without truncation. Repeating each patched barycenter across its
fiber makes within-orbit branch accuracy exactly chance; no stronger global
branch-erasure claim is made.

## Preregistered gates

Require at least four of five seeds separately by degree unless a gate is
explicitly `k=3` only:

1. **Exact group contract:** all true deck rotations change the averaged
   sublayer state by at most `1e-5` relative norm and its posterior by at most
   `1e-7` mean squared Fisher--Rao distance on both shifts.
2. **Charged-mode necessity:** the `alpha=0` patch fails and exact `alpha=1`
   passes at the frozen synthesis front on both shifts.
3. **Orbit-specific carrier:** the substituted-carrier control does not
   reproduce synthesis on either shift.
4. **Finite-`C3` phase mechanism (`k=3` only):** both shifts are
   nondegenerate and classified `finite_group_phase_sensitive`.
5. **Shift-stable phase phenotype (`k=3` only):** composition and
   extrapolation receive the same phase classification.

Quadratic homogeneity is a declared diagnostic, not a promotion criterion:
report its Fisher-effect explained fraction at `alpha=0.5` and `0.75`. Its
purpose is to distinguish a nonlocal radial mechanism from discrete-phase
dependence when the earlier local Hessian approximation fails.

## Outcome meanings

- Necessity and carrier specificity support a causal irrep-fusion mechanism.
- High `C3` phase sensitivity supports exact finite-group fusion terms rather
  than an `O(2)`-radial surrogate.
- Low phase sensitivity with poor quadratic homogeneity supports higher radial
  powers of `|c1|^2`.
- Low phase sensitivity with good quadratic homogeneity means the earlier HVP
  failure was primarily derivative-scale or state-space approximation error.
- Substituted carriers reproducing the effect means generic carrier energy or
  curvature is sufficient; orbit-specific character coupling is not established.
- Failure of the exact deck contract invalidates the intervention implementation.

## Boundaries and artifacts

The continuous phase rotation is an off-orbit intervention inside the real
`C3` isotypic carrier. It tests the local representation realized by these
activations; it does not assert that those rotated states occur naturally.

Confirmatory artifacts will be written to
`data/experiments/tinyllm_irrep_fusion_ablation/20260806_d6_preregistered`.
Disposable shakedowns use a separate root and are never pooled with evidence.
