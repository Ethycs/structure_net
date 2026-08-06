# Depth-Graded TinyLLM Family

**Status:** EXPERIMENTAL AS-BUILT  
**Date:** 2026-08-05  
**Applies to:** `tinyllm_model.py`, `tinyllm_depth_graded_quotient.py`

## Purpose

The depth-graded API treats real depths as compatible slices of one maximum-depth transformer rather than as independently trained models. Integer depths are exact block prefixes. Between consecutive prefixes, the next pre-normalized residual block is continuously activated.

For block `k` and gate `α ∈ [0,1]`:

\[
\widetilde h=h+\alpha A_k(\operatorname{LN}_1(h)),
\qquad
B_{k,\alpha}(h)=\widetilde h+\alpha M_k(\operatorname{LN}_2(\widetilde h)).
\]

Thus `B(k,0)` is the identity and `B(k,1)` is exactly the ordinary TinyLLM block.

```text
embedding = depth 0
    │
    ├── exact blocks 0 … floor(s)-1
    │
    └── gate α = fractional(s) on the next block
                         │
                         v
                 one shared LM head
```

This realizes a concrete continuous residual-gate path over depth. It is a neural representative of a mapping-telescope cylinder, not a proof that transformer depth has a unique natural continuum.

## Public model contract

`TinyLLMBlock.forward_gated(value, gate)` supplies the partial residual block. Its endpoints dispatch to the identity and ordinary block paths exactly.

`TinyLLMModel.forward_at_depth(input_ids, depth)`:

- accepts any real depth from zero through `n_layer`;
- executes exact full prefixes plus at most one partial block;
- applies the existing final layer norm and tied LM head;
- matches ordinary `forward` exactly at maximum integer depth;
- rejects feedback graphs because their refinement execution has a separate depth semantics.

`TinyLLMModel.residual_at_depth(input_ids, depth)` executes the same exact-prefix/
partial-block family but returns the unnormalized residual stream before `ln_f`
and the tied language-model head. This is the supported observation boundary for
testing whether information is present internally rather than only visible to the
current decoder.

Tests compare every integer depth to a manually evaluated prefix, compare maximum depth to ordinary forward logits, and check continuity around a fractional gate.

## Training arms

The experiment uses one d8 architecture and shared decoder in three matched arms:

| Arm | Depths contributing gradients per minibatch |
| --- | --- |
| `standard_final` | full depth 8 only |
| `discrete_multi_exit` | depth 8, depth 1, two seeded random integer depths |
| `continuous_gate` | depth 8, depth 1, two seeded random real depths |

Each sampled loss is backpropagated separately and averaged before one clipped AdamW update. Sequential backward passes keep activation memory bounded on the 8 GB test GPU. Initialization, data, minibatches, optimizer, and task targets remain matched. The ordinary controls must reproduce the retained source state hashes.

## Depth–training diagram

At checkpoints `0,25,50,100,200,400,600`, the runner evaluates a continuous input lift across depths `0,0.25,…,8`. One compressed NPZ field archive is stored per arm/task cell, together with every final model checkpoint.

For phase, a depth cell records circular-posterior alignment, winding degree, minimum moment magnitude, and phase-grid resolution. A carrier front requires:

- alignment at least 0.9;
- winding degree `+1`;
- minimum moment magnitude at least 0.05;
- resolved adjacent angular increments.

For cosine, a quotient-proxy front requires:

- posterior cosine Pearson correlation at least 0.9;
- paired opposite-branch residual-distance ratio at most 0.5;
- cosine-reference distance Spearman correlation at least 0.6;
- nondegenerate residual spread.

The cosine branch-distance criterion is not conditional mutual information, and the paired metric criterion is not a Reeb graph or cosheaf.

## Conditional branch-depth protocol

`tinyllm_conditional_branch_depth_scan.py` adds the direct residual test that the
original depth diagram lacked. It freezes checkpoints and evaluates exact
cosine-matched phase pairs with:

- linear and two-hidden-layer nonlinear branch probes conditioned on cosine;
- a cross-validated cosine-only log-loss null;
- within-pair randomized-label controls;
- residual cosine regression, decoder cosine correlation, and posterior H1;
- overlapping and disjoint nuisance families;
- a finite-fiber MST merger-scale component proxy;
- pre-attention, post-attention, and post-MLP cuts around the detected residual front.

Three nested fronts keep distinct claims separate:

| Front | Required evidence |
| --- | --- |
| ID residual quotient | branch at chance, residual cosine retained, negligible conditional log-loss gain, one-component fiber proxy |
| ID decoder-supported quotient | ID residual quotient plus decoder cosine correlation at least 0.9 |
| nuisance-robust quotient | decoder-supported quotient plus shifted residual cosine retention, shifted branch chance, and shifted one-component proxy |

The strictest front is the preregistered headline. Failure of its nuisance gate
does not erase an ID residual result; it limits the result to the distribution on
which cosine itself remains recoverable.

## Depth defect charge

At every phase checkpoint, posterior moments form a sampled field on the phase/depth cylinder. `complex_defect_charge` compares the degree at depth zero and full depth with the signed cell charge between them.

The shared random initialization required targeted phase refinement at five nearly zero depths. Their winding degrees resolved without changing. Its original 1024-point defect-cell decomposition remains explicitly exploratory; trained checkpoint cylinders are the evidentiary charge set.

## Mathematical boundary

The implementation establishes a continuous family for one chosen residual gating rule. It does not establish:

- a neural-ODE or depth-refinement limit;
- state/posterior convergence as the depth discretization is refined;
- induced-map persistent cost;
- conditional branch mutual information;
- a Reeb cosheaf, Whitney stratification, or linked defect curves;
- interval-certified zero isolation;
- repeatability beyond the retained seed.

A well-posed ODE control remains a distinct experiment because its invertible finite-depth flow has different quotient-forming constraints.

## Verification

```bash
pixi run pytest -q \
  tests/structure_net/components/test_tinyllm_model.py \
  tests/structure_net/test_tinyllm_depth_graded_quotient.py \
  tests/structure_net/test_tinyllm_conditional_branch_depth_scan.py
```

The measured campaigns are recorded in
`../08 - Analysis/2026-08-05_tinyllm-depth-graded-quotient.md` and
`../08 - Analysis/2026-08-05_tinyllm-conditional-branch-depth-scan.md`.
