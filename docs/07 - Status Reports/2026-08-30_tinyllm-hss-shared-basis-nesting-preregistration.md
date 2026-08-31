# TinyLLM HSS shared-basis and nesting diagnostic preregistration

**Status:** REGISTERED — NOT RUN
**Date:** 2026-08-30
**Menu ID:** A6
**Hypothesis:** `tinyllm-hss-shared-basis-nesting-v1`
**Planned schema:** `nal.tinyllm-hss-shared-basis-nesting.v1`
**Parent:** `tinyllm-dynamic-ttno-rank-pilot-v1`

## Question

Why did the A4 chronological token-tree HSS boundary rank remain at median `33`
while the paired-bit TTNO rank reached median `90` at 256 tokens? Specifically,
are individually low-rank dyadic interactions compatible with one shared basis
per cluster, and are those shared bases nested across parent-child edges?

This study diagnoses necessary structure for constructive causal H2 attention.
It does not construct A7's simultaneous operator and cannot certify H2 output
error or cost.

## Frozen parent evidence

Use the same parent aggregate, checkpoint, token stream, 120 frozen cells, and
artifact hashes declared by A5. The required A4 aggregate SHA-256 is
`9d3fa8ec7332860785b3d62dff5805e4ec23c9f6fedc5178c6809b15cd05feba`.
Chronological token order is primary because A4 found it lower-rank than Q/K-PCA
order in all 120 cells at length 256.

Lengths `32, 64, 128, 256` are retained for scaling. The primary localization
is length 256. Use relative-Frobenius tolerance `1e-2` as primary and `1e-3` as
sensitivity.

## Complement peer partition

For every non-root dyadic token cluster `u`, partition its complement into the
disjoint siblings encountered on the path from `u` to the root. This produces a
deterministic set `Peers(u)` of at most `log2(n)` dyadic clusters whose union is
exactly `u^c`.

For the causal attention matrix `A`, measure both orientations:

```text
query-side block: A[u, v]
key-side block:   A[v, u]^T
```

for every `v` in `Peers(u)`.

## Independent and shared ranks

At each cluster define:

```text
r_ind_query(u) = max_v rank_epsilon(A[u,v])
r_ind_key(u)   = max_v rank_epsilon(A[v,u]^T)

M_query(u) = concat_v A[u,v]       = A[u,u^c] up to column order
M_key(u)   = concat_v A[v,u]^T     = A[u^c,u]^T up to column order

r_shared_query(u) = rank_epsilon(M_query(u))
r_shared_key(u)   = rank_epsilon(M_key(u))
```

The sharing-inflation ratio is `r_shared / max(r_ind,1)`. This distinguishes
small independent blocks from the larger basis needed to serve all partners at
once. Reconstructed concatenations must match the direct HSS boundary matrices
exactly.

## Parent-child nesting

Let `U_u` be the leading left singular subspace of `M_query(u)` at the declared
rank, and define the key-side basis analogously. For child `c` of parent `u`,
restrict the parent basis to the rows of `c` and measure

```text
d(c <- u) = ||(I - U_c U_c^T) restrict_c(U_u)||_F
            / max(||restrict_c(U_u)||_F, 1e-12).
```

Also record the numerical rank of `[U_c, restrict_c(U_u)]` and its ratio to
`max(r_shared(c),1)`. Query and key orientations are kept separate.

Because truncated singular subspaces can rotate at a degenerate cutoff, a cut
is `spectrally_stable` only when `s_r / max(s_(r+1), 1e-12*s_1) >= 2`. Nesting
defect gates use stable cuts; all cuts remain in descriptive summaries.

## Primary endpoints and gates

Pool query and key orientations only after retaining orientation-specific raw
records. Aggregate per frozen cell before aggregating across 120 cells.

| Gate | Rule at length 256 |
| --- | --- |
| validity | 120/120 cells reproduce A4 chronological HSS maxima and exact complement concatenation |
| shared-basis compactness | median sharing inflation `<= 2.0` and 90th percentile `<= 3.0` |
| nesting fidelity | on stable cuts, median defect `<= 0.10` and 90th percentile `<= 0.25` |
| nested-rank compactness | median augmented/shared rank ratio `<= 1.5` and 90th percentile `<= 2.0` |

Classification is:

- `invalid_parent_or_boundary_contract` if validity fails;
- `shared_and_nested_hierarchy_supported` if all three structural gates pass;
- `shared_basis_bottleneck` if only shared-basis compactness fails;
- `nesting_bottleneck` if shared-basis compactness passes and either nesting
  gate fails;
- `combined_sharing_and_nesting_bottleneck` if sharing and nesting fail.

If fewer than 25% of parent-child cuts are spectrally stable, nesting is
classified `indeterminate_degenerate_cutoffs`; it cannot pass or fail the
nesting gate, and the campaign classification becomes
`shared_basis_result_nesting_indeterminate` unless validity fails.

## Controls

- Run the same diagnostic on A4's causal-uniform, smooth-Fourier, and IID-Q/K
  controls.
- Apply one deterministic random simultaneous token permutation per evaluation
  seed as a tree-destruction control.
- Confirm that an exactly rank-one dense operator has independent rank, shared
  rank, and nested augmented rank one with zero nesting defect.
- Reconstruct every shared boundary from its peer blocks before calculating a
  basis.

## Expected artifacts

```text
data/experiments/tinyllm_hss_shared_basis_nesting/20260830_registered/
  campaign_results.json
  runs/seed_<seed>/result.json
```

## Interpretation boundaries

- Passing A6 establishes finite shared/nested subspace compatibility, not a
  simultaneous H2 approximation or its weighted row-relative error.
- SVD bases are diagnostic bases, not an implicit subquadratic core generator.
- HSS complement ranks are stronger than individual block ranks but do not fix
  an H2 admissibility rule for A7.
- One pretrained checkpoint does not establish cross-model repeatability.
- A6 cannot rescue A4's failed paired-bit TTNO envelope.
