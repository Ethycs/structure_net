# J-space relation propagation through an intact model

Status: prospective design; exact CPU algebra control only is implemented.  
Date: 2026-11-13 (America/Los_Angeles).  
Study ID: `JSPACE-RELATION-PROPAGATION-01`.

## Corrected objective

The primary question is how algebraic relations among J-derived concept
observables propagate through a model. Compression is one possible application.
The prior CAV reconstruction pilot tested a narrower, stronger sufficiency claim;
its failed reconstruction does not decide this question. Preserve its evidence
and interpretation without retroactively changing its hypotheses or gates.

Keep the original model and its full activations intact. For concept scores
`z_l = C_l.T @ h_l`, investigate polynomial relations `I_l` within a layer and
joint relations `R_l` involving `z_l` and `z_(l+1)`. Directions, linear coordinate
bases, ideals of relations, and Gröbner bases of those ideals are distinct objects.
A Gröbner basis is computed from specified polynomial relations; it is not
computed directly from a list of activation vectors without that intervening step.

For an exact polynomial relation model, candidate next-layer constraints follow
by eliminating the previous variables from `I_l + R_l`:

`I_pred_next = (I_l + R_l) intersect K[z_(l+1)]`.

This determines polynomial consequences of the specified relation system. It
need not describe every realizable real-valued state, especially for approximate
or incompletely specified neural relations. A deterministic reduced map `T_l`
is optional. Where one is justified, preservation of an output relation `g`
can be checked by reducing `g(T_l(z))` modulo an input Gröbner basis.

## Experiment sequence

1. **Exact positive and negative controls, CPU only.** Start with two Boolean
   concepts. Propagate them through AND/OR and then XOR/AND. Verify inferred
   output ideals against exhaustive states, and verify that a deliberately
   changed transition changes the propagated relation. This validates the
   elimination/normal-form implementation, not a neural-model claim.
2. **Observe intact neural transitions.** Use a small bank of named J-derived
   observables at adjacent layers, with explicit cross-layer concept alignment.
   Record source and destination scores, task labels, token positions, and
   normalization information. No activation reconstruction is required.
3. **Fit and validate relations.** Start with degree-one and degree-two
   candidates and enough independent training examples to avoid an automatic
   underdetermined polynomial fit. Use separate prompt families for validation
   and evaluation. Reject near-constant observables and trivial zero relations.
   Candidate discovery and coefficient/threshold choices use training and
   validation only. Compare within-layer relations, joint transition relations,
   and their held-out propagation residuals.
4. **Test causal transport.** Intervene on selected concept coordinates while
   retaining the rest of the state, then test predicted changes in relation
   residuals at later layers. Include matched random/PCA coordinates, shuffled
   source-destination pairings, and low-degree regression controls. Natural
   transitions and interventions must be reported separately.
5. **Explore applications conditionally.** Relations that persist or transform
   reliably can support mechanistic interpretation, intervention design,
   prediction, or consistency checks. A separate autonomous-closure and runtime
   experiment is required for a compression claim.

## Interpretation and execution rules

Track which relations are preserved, violated, created, or become dependent,
their first detectable layer, held-out residuals, and causal responsiveness.
Keep variable identity, scaling maps, coefficient field, and monomial order
explicit. A textual change in a basis can result from coordinates or ordering;
compare relation consequences as well as basis listings.

Real transformer transitions contain nonpolynomial operations and noisy
floating-point observations. Exact ideal identities apply only to the exact
control or an explicitly justified polynomial model. Approximate fitted
relations must remain approximate claims: naive rationalization can create an
inconsistent ideal, and finite-sample interpolation can invent relations.
No exact symbolic certificate will be presented as a proof about the original
transformer merely because a fitted surrogate reduces to zero.

Parent task accuracy remains a separate endpoint. It gates claims about correct
task reasoning, not observation of the computations a model actually performs.
The previous 48 evaluation outcomes are already inspected and cannot become a
new confirmatory held-out set. New neural captures require fresh frozen splits
and a bounded protocol before launch; the current document does not claim that
those measurements have run.

Use root Pixi with CUDA hidden for this exact CPU control. Any subsequent GPU
capture or intervention job must acquire and validate an experiment reservation
and release it afterward. Keep all prior campaign artifacts append-only.
