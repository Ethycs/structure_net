# TinyLLM A8 H2 gauge channelization preregistration

**Status:** REGISTERED — NOT RUN
**Date:** 2026-08-30
**Menu ID:** A8
**Hypothesis:** `tinyllm-h2-gauge-channelization-v1`
**Planned schema:** `nal.tinyllm-h2-gauge-channelization.v1`
**Evidence parent:** `tinyllm-causal-h2-attention-v1`

## Question

Can orthogonal gauge synchronization inside the already validated A7 causal H2
representation expose diagonal or fixed-width block channels whose deterministic
pruning preserves the frozen A7 approximation gates while reversing its measured
finite-size storage and operation disadvantage at 256 tokens?

This experiment tests structural gauge fixing inside fixed H2 subspaces. It does
not claim to reduce A5 TT ranks, change A7 H2 ranks, learn a token-space
descrambler, or establish an online subquadratic compiler.

## Frozen evidence

| Item | Frozen value |
| --- | --- |
| A7 campaign | `data/experiments/tinyllm_causal_h2_attention/20260830_registered/campaign_results.json` |
| A7 campaign SHA-256 | `4885696dd746a52cb015b51d34733901c2acd50baccfc59f2f76cfe176eeb9b2` |
| A7 implementation SHA-256 | `9e61430463ecc11ff5d99560625a9f85a6e5b189d39a43fdbcf26bb06e804dbc` |
| A6 campaign SHA-256 | `6f42a59b3a723eb4b80742e8fab8278be9b21d1db35131cc4ea81bd702e03c01` |
| checkpoint SHA-256 | `5a7491b4231c30feaaabda7babf0a831dd4d72fbb4e7f3e757114cadf098df09` |
| token stream SHA-256 | `f339655453b970ae6cc1cbdc7c78f8a5234c42437bc7bba8fb31a1dee5c9d765` |
| evaluation seeds | `101, 211, 307, 401, 503` |
| layers | `0..7` |
| heads | `0, 3, 7` |
| primary length | `256` |
| diagnostic lengths | `64, 128` |

The evaluation seeds retain A4–A7 semantics: deterministic validation-prefix
replicates against one frozen pretrained checkpoint, not independently trained
model seeds.

## Frozen A7 construction and gates

A8 must replay A7's strong-admissibility partition, leaf size `16`, separation
ratio `1.0`, build tolerance `0.0025`, logarithmic-squared rank cap, stabilized
unnormalized kernel construction, augmented `[V, 1, probes]` contraction, and all
causality, normalization, output, tail, and validity checks without modification.

At 256 tokens the required partition remains `33 ADMISSIBLE`, `31 DENSE`, and
`15 ZERO`, with canonical SHA-256
`1535f675b18b8493d5da1c4e1048eefb73c884e700953451de7f9b8b380688f2`.

The frozen representation gates are:

- at least `0.80` passing cells at length 256;
- at least `0.50` passing cells in every layer at length 256;
- kernel row-relative maximum `0.01`;
- denominator-relative maximum `0.01` with positive denominators;
- attention row-L1 maximum `0.025`;
- probe and value relative-Frobenius maxima `0.02`;
- token-output p99 global-RMS-normalized maximum `0.05`;
- exact contraction relative error at most `1e-10` and zero future leakage.

The frozen compression gates at length 256 are:

- storage ratio median at most `0.75` and p90 at most `1.0`;
- operation ratio median at most `0.75` and p90 at most `1.0`.

## Gauge action

For every nonempty query basis `U_I` and key basis `V_I`, A8 introduces
orthogonal matrices `G_I` and `H_I`. The exact gauge action is

```text
U_I' = U_I G_I
V_I' = V_I H_I
S_IJ' = G_I^T S_IJ H_J
E_query,c<-p' = G_c^T E_query,c<-p G_p
E_key,c<-p' = H_c^T E_key,c<-p H_p
```

Before pruning this must leave both explicit assembly and the prescribed H2
contraction invariant to relative error at most `1e-10`. Ranks, partition, near
blocks, and dense operator values are frozen.

## Frozen arms

Each cell and length evaluates the following arms in this order:

1. `exact_unpruned`: optimized gauges with no pruning; mandatory equivalence
   control and unchanged A7 accounting.
2. `identity_diagonal`, `identity_block2`, `identity_block4`: project the original
   SVD coordinates without optimization; fixed-coordinate controls.
3. `optimized_diagonal`, `optimized_block2`, `optimized_block4`: optimize gauges
   for the matching mask, then set every off-block transfer and coupling entry to
   exact zero.

For a rectangular matrix, an entry `(i,j)` is retained exactly when
`floor(i / b) == floor(j / b)`, with `b` equal to `1`, `2`, or `4`. The mask is
therefore known from matrix shapes and has no stored index cost.

## Frozen optimizer

The optimizer may inspect only the unpruned transfer and coupling matrices. It
must not inspect dense reconstruction errors, attention/output metrics,
acceptance gates, or campaign aggregates.

- dtype: float64 on the selected CUDA device;
- parameterization: a skew-symmetric parameter followed by a Cayley transform,
  right-multiplied by the restart initializer;
- objective: total squared Frobenius energy outside the arm's fixed block mask,
  divided by total squared Frobenius energy across all query transfers, key
  transfers, and admissible couplings;
- optimizer: Adam, `96` updates, learning rate `0.03`, no weight decay;
- restarts: exactly two, identity and deterministic local spectral covariance;
- selection: the restart with lower final factor-only objective;
- no early stopping, threshold sweep, rank change, or post-result retuning.

The spectral initializer at a node is the descending eigenbasis of the sum of
normalized incident left or right Gram matrices. Eigenvector signs are fixed by
making the largest-magnitude component positive. Degenerate empty or zero-energy
nodes use identity.

## Diagnostic length transfer

Lengths 64 and 128 are reported to determine whether the same frozen optimizer
produces a consistent block-energy trend before the held-out maximum length.
Optimizer hyperparameters and all masks are frozen before any 256-token result is
read. Because H2 basis ranks and cluster incidence change with length, numerical
gauge matrices are not asserted to be dimension-wise identical across lengths;
the transferable object tested here is the fixed local gauge-selection rule.
This is weaker than cross-checkpoint shared-gauge evidence and must be labeled as
such.

## Sparse accounting

Rotated leaf bases retain their original dense storage. Near-field storage and
work are unchanged. Transfer and coupling storage equals the number of entries
allowed by the public block masks, including retained numerical zeros. Their
multiply-add count is that structural scalar count times the unchanged augmented
channel count. Gauge-optimization and H2-construction work are compile-time
diagnostics and are not counted, matching A7's construction boundary; this
prevents an inference-speed claim.

## Primary decision order

1. `invalid_parent_or_gauge_contract` if any frozen hash, A7 replay condition,
   partition, finite-value, exact-gauge invariance, or contraction check fails.
2. `gauge_channelization_compression_pass` if at least one optimized arm passes
   every frozen representation, layer, storage, and operation gate at length 256.
3. `gauge_channelization_representation_only` if an optimized arm passes the
   representation and layer gates but no optimized arm passes compression.
4. `gauge_channelization_sparsity_accuracy_tradeoff` if an optimized arm passes
   compression accounting but every compression-passing optimized arm fails the
   representation or layer gate.
5. `gauge_channelization_no_structural_gain` otherwise.

All arms are reported. The descriptive preferred arm is chosen only after the
classification: among fully passing arms, smallest block width first; otherwise
among representation-passing arms, lowest storage median then lowest kernel-row
error median. This preference cannot rescue the primary classification.

## Controls and validity

- `exact_unpruned` must reproduce A7 assembly and contraction while preserving
  full storage and operation counts exactly.
- Every optimized arm is compared with its matching identity-projection control.
- Orthogonality residuals for all gauges must be at most `1e-10`.
- Structural nonzero counts must equal the analytically generated mask counts.
- Pruned contraction must match pruned explicit assembly to `1e-10`.
- All five evaluation seeds are required; partial completion is not evidence.

## Interpretation boundaries

A pass would show that much of A7's finite-size overhead is removable internal
channel mixing in an operator-specific H2 factorization. It would not show shared
gauges across independently trained models, reduce A5's paired-bit ranks, include
online construction cost, or establish realized sparse-kernel wall-clock speed.

A representation-only outcome would show coherent channels that are not sparse
enough under the frozen blocks. An accuracy/compression tradeoff would show that
the overhead can be deleted only by violating the frozen attention certificate.
A no-gain result would support irreducible internal mixing at this block scale.
