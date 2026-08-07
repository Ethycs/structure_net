# TinyLLM equivariant Morse–Cerf quotient-front preregistration

**Status:** PREREGISTERED — outcomes not inspected  
**Date:** 2026-08-06  
**Hypothesis:** `tinyllm-equivariant-morse-cerf-quotient-front-v1`

## Question

Is the exact orbit-averaging causal quotient front a symmetry-constrained Morse
transition in the task-fiber mixture simplex, or merely a task-loss threshold?
Do the causal task potential and autonomous Reynolds-commutator potential undergo
the same transition?

## Frozen sources

Reuse without retraining the ten retained d6 degree-ladder checkpoints for
`k=2,3` and seeds `7,17,29,41,53`. Validate model-state and checkpoint digests.
Use the completed deck-action campaign only as a frozen source of exact-orbit
causal classifications and the completed Reynolds–Koopman campaign only as a
frozen autonomous-mismatch comparator. Neither fitted observer enters the Morse
potential.

Use 32 disjoint exact nuisance-matched orbits per regime and seed. Analyze
composition and extrapolation separately. `k=2` uses a 49-point interval grid;
`k=3` uses the complete barycentric lattice with 12 subdivisions (91 points).

## Intervention simplex and symmetric task potential

At a residual cut, reshape the full three-token states into exact deck orbits
`h_j`, and intervene with every convex mixture `H(w)=sum_j w_j h_j`. The smooth
task potential is mean target-posterior KL after frozen downstream continuation.
Exact cyclic symmetrization averages the potential over all cyclic permutations
of `w`. Accuracy remains a causal validation endpoint, never the Morse function.

For each potential, define the task-valid threshold as

`tau_valid = max_j V(e_j) + 0.05`.

Report barycenter excess loss

`E = V(center) - mean_j V(e_j)`

and merge barrier

`B = tau_merge - max_j V(e_j)`,

where `tau_merge` is the smallest finite-grid sublevel at which every vertex and
the barycenter lie in one connected component.

## Vertex-exact real-sublayer homotopy

There is no exact semigroup law for a gated transformer residual branch. Use the
following declared intervention-timing homotopy for each attention or MLP
sublayer. Let source sheets be `x_j`, exact residual updates be `d_j`,
`x_j(alpha)=x_j+alpha*d_j`, and let `f` be that sublayer's residual update.
For a mixture `z=sum_j w_j x_j(alpha)`, complete the sublayer with

`y = z + (1-alpha) f(z)
     + (1-alpha) sum_j w_j [d_j - f(x_j(alpha))]`.

At `alpha=0`, this is intervention before the sublayer. At `alpha=1`, it is
intervention after it. For every alpha and every vertex, `y(e_j)=x_j+d_j`, so
the original sheet outputs are fixed. The correction is part of the declared
Cerf homotopy; this is an intervention-timing family, not a neural-ODE flow.

Use nine equally spaced alpha values for the primary potential and five for each
control. Concatenate the six real sublayers as `s in [0,6]`.

## Autonomous commutator potential

Keep a separate potential

`C_alpha(w) = || F_alpha(sum_j w_j x_j(alpha))
                 - sum_j w_j F_alpha(x_j(alpha)) ||^2
               / (||sum_j w_j F_alpha(x_j(alpha))||^2 + epsilon)`,

where `F_alpha(q)=q+(1-alpha)f(q)`. Symmetrize identically. This is the exact
failure of the remaining gated sublayer to commute with Reynolds mixing. Its
declared center-valid threshold is `C(center)<=0.01`.

## Numerical Morse and merge-tree analysis

For `k=2`, compute the center second derivative with the centered finite
difference on the 49-point grid. Index one requires curvature `<-1e-4`, index
zero curvature `>1e-4`, and smaller magnitude is declared degenerate. The merge
tree is exact on the interval graph.

For `k=3`, use the complete triangulated barycentric lattice. Estimate the center
Hessian by a symmetric local quadratic fit and classify eigenvalues with tolerance
`1e-4`. Enumerate discrete minima, maxima, and saddles by lower-link connectivity;
do not infer the transition from the center alone. Compute sublevel connectivity
by union-find on the full lattice graph.

The task Morse front is the earliest real depth where:

1. the barycenter has index zero;
2. `E<=0.05` and `B<=0.05`;
3. the barycenter lies in the task-valid vertex component;
4. the condition persists at every later discrete sublayer endpoint.

The commutator front is the earliest real depth with center index zero and center
commutator at most `0.01`, persisting at later discrete endpoints.

## Controls

Evaluate two full intervention families:

- random orbit pairing, independently permuting the non-reference sheets across
  nuisance/phase orbits;
- deterministic phase-shuffled pairing, rolling non-reference sheets across the
  quotient-phase order while preserving their internal tensor structure.

A control is specific when it does not produce a mature task front within one
sublayer of the frozen exact-orbit causal front. The controls are not interpreted
as deck actions and need not preserve vertex task loss.

## Primary gates

Require every gate separately on composition and extrapolation in at least four
of five seeds for each degree:

1. **Early cover structure:** before the frozen causal front, a task-valid
   sublevel has at least `k` sheet-associated components or the barycenter has
   positive Morse index.
2. **Near-front Morse event:** a barycenter index change, discrete saddle merger,
   or merge-barrier crossing occurs within one real sublayer of the causal front.
3. **Mature barycenter basin:** after the front the exact barycenter is in the
   task-valid vertex component, has index zero, and has `E<=0.05`, `B<=0.05`.
4. **Control specificity:** neither control produces the same mature front within
   one sublayer of the exact causal front.
5. **Task/closure distinction:** if the frozen Reynolds result failed autonomous
   closure near its task front, the Morse task and commutator fronts differ by
   more than `0.5` sublayer; if it passed, they agree within one sublayer.

The full hypothesis is confirmed only if all five gates pass for both degrees.
A loss crossing without an index change or saddle/merge-tree event is explicitly
reported as metric calibration, not a Morse mechanism.

## Stage-two certification rule

Seed 7 is the sole `k=3` certification candidate. After numerical localization,
continue only if it has an isolated, stable critical event and fit an
outward-rounded Chebyshev surrogate on a small
`(alpha,w)` box. Certify only statements proved for the stored floating
polynomial. Transfer to the transformer requires a rigorous network-to-surrogate
remainder; sampled errors cannot satisfy that gate. If no isolated event exists,
record certification as not attempted for lack of a valid seed-7 candidate rather
than cherry-picking another seed.

## Boundaries

This study analyzes a finite simplex grid and a declared vertex-exact residual
homotopy. It cannot establish a universal Morse normal form, exact critical-point
census for the neural network, continuum connectedness between grid points, or
formal transformer certification without interval remainder bounds.

## Append-only artifacts

The confirmatory root is
`data/experiments/tinyllm_morse_cerf/20260806_d6_preregistered`.
Shakedowns and certification amendments use separate roots.
