# TinyLLM dynamic TTNO rank pilot preregistration

**Status:** PREREGISTERED EXPLORATORY PILOT
**Date:** 2026-08-29
**Hypothesis:** `tinyllm-dynamic-ttno-rank-pilot-v1`
**Schema:** `nal.tinyllm-dynamic-ttno-rank-pilot.v1`

## Question

Do causal attention operators from one frozen BabyLM-pretrained TinyLLM checkpoint
have numerical ranks compatible with a small input-conditioned hierarchical
operator, and specifically with a quantized tree tensor network operator (TTNO),
as context length grows from 32 to 256 tokens?

This pilot separates two objects that are easy to conflate:

1. a token-cluster tree for an `n x n` attention matrix, measured through
   HSS-style off-diagonal boundary ranks as a stronger diagnostic than isolated
   admissible H2 block ranks; and
2. a genuine TTNO of the linear map on `R^n`, obtained for `n = 2^L` by
   reshaping row and column indices into `L` binary input/output pairs and
   measuring matricization ranks on a balanced dimension tree over those
   paired modes.

The second construction is a quantized TTNO with `L = log2(n)` physical modes.
It is not a TTNO with one tensor-product physical site per token.

## Evidence role and prediction

This is an exploratory, single-checkpoint rank pilot. Evaluation stream seeds
replicate input selection, not model training. The result may be classified as
`polylog_compatible_pilot`, `mixed_rank_pilot`, or `bond_growth_observed_pilot`,
but cannot confirm an architecture-level or natural-language population claim.

The directional prediction is that the best of two declared trees will keep
the 1% relative-Frobenius numerical TTNO rank within the explicit
`ceil(log2(n)^2)` envelope in at least 80% of frozen attention cells, and that
the median rank divided by `log2(n)^2` will not increase from length 64 to 256.

## Frozen source and data

| Item | Fixed value |
| --- | --- |
| model | Structure Net `TinyLLMModel`, d8, eight layers and eight heads |
| checkpoint | `data/experiments/tinyllm_babylm_pretrain/20260812_d8_seed7/checkpoint_step12000.pt` |
| token stream | `data/corpora/babylm_10M_bpe16k.tokens.npy` |
| split | final 262,144 tokens, matching the pretraining validation suffix |
| evaluation seeds | `101, 211, 307, 401, 503` |
| lengths | `32, 64, 128, 256` nested prefixes from each sampled validation segment |
| layers | all eight |
| heads | `0, 3, 7`, fixed before outcomes to cover low, middle, and high head indices |
| arithmetic | attention matrices and rank calculations in float64 |

For each evaluation seed, one start offset is sampled uniformly from the
validation suffix with enough room for 256 tokens. The same segment prefix is
used at every declared length. No text or rank outcome is used to choose an
offset.

## Operators and exact identity check

For every selected layer and head, the runner captures the actual normalized
block input and applies the frozen learned Q/K projection. The causal operator is

```text
A = softmax((Q K^T) / sqrt(d) + causal_mask).
```

It independently reconstructs the same operator through the Gaussian log
factorization

```text
log G_ij = -||q_i-k_j||^2 / (2 sqrt(d))
log W_j  =  ||k_j||^2 / (2 sqrt(d))
A_gaussian = row_normalize(exp(log G + log W) * causal_mask).
```

The maximum absolute difference must be at most `1e-10`. This is an arithmetic
contract, not a compression result.

## Declared trees

Every operator is evaluated under both of these Q/K-only orderings:

- `chronological`: original causal token order;
- `qk_pca`: stable sort by the first principal-component score of centered
  `[Q, K]` features, with component sign fixed by its largest-magnitude loading.

After simultaneous row/column permutation, indices are binary tensorized in
most-significant-bit order. `best_declared_tree` is the lower rank of these two
orders for the same frozen cell. This optimistic existence diagnostic is fixed
in advance; no search over other permutations is allowed.

## Rank definitions

For a matrix or tensor matricization with singular values `s`, the relative
Frobenius numerical rank at tolerance `epsilon` is the smallest `r` satisfying

```text
sqrt(sum(s[r:]^2) / sum(s^2)) <= epsilon.
```

The runner records `epsilon = 1e-2` as primary and `1e-3` as a sensitivity
diagnostic.

### Token-tree boundary rank

A balanced binary tree recursively partitions the ordered token leaves. For
every non-root cluster `I`, the runner measures the numerical ranks of
`A[I, I^c]` and `A[I^c, I]`. Their maximum is the HSS-style boundary rank.
This is not claimed to be an H2 construction or its application cost.

### Quantized TTNO rank

For `n = 2^L`, reshape `A[i,j]` into an order-`L` tensor with paired modes
`(i_bit_l, j_bit_l)` of dimension four. For every non-root node of a balanced
dimension tree over these paired bit modes, matricize the node modes against
their complement. The maximum numerical rank is the reported quantized TTNO
bond-rank diagnostic.

Per-cut numerical ranks do not by themselves construct one simultaneous
epsilon-accurate TTNO; they measure the necessary rank profile and the standard
hierarchical-SVD compatibility diagnostic.

## Controls

At every length and evaluation seed, use the same head width and causal mask for:

- `causal_uniform`: zero Q/K, producing uniform attention over each causal prefix;
- `smooth_fourier`: deterministic low-frequency positional Q/K features;
- `iid_qk`: independent standard-normal Q/K features.

The controls use the same tree and rank code. The exact identity matrix is a
unit-test control and must have exact quantized TTNO rank one.

## Sparse exception diagnostic

At length 256, for each natural attention cell under each declared ordering,
retain the two largest allowed attention entries per row as `S` and set
`R = A - S`. Record:

- mass captured by `S`;
- exact quantized TTNO cut rank of `S` using tolerance `1e-12`;
- 1% numerical quantized TTNO rank of `R`;
- whether separating `S` lowers the remainder rank relative to `A`.

This row-wise top-two split is a declared diagnostic, not a learned sparse
router or an H2 near-field partition.

## Primary classification

The unit of a frozen attention cell is `(evaluation_seed, layer, head)`. For
each cell and length, use `best_declared_tree` at `epsilon = 1e-2`.

| Gate | Rule |
| --- | --- |
| Gaussian identity | all reconstructed operators have max absolute error `<= 1e-10` |
| polylog envelope | at least 80% of natural cells satisfy rank `<= ceil(log2(n)^2)` at every length |
| normalized growth | aggregate median `rank / log2(n)^2` at length 256 is no larger than at length 64 |
| random separation | secondary: natural median rank at length 256 is at most 75% of matched `iid_qk` median |

Classification is:

- `polylog_compatible_pilot` if the identity, envelope, and normalized-growth
  gates pass;
- `mixed_rank_pilot` if the identity gate passes but exactly one of the two rank
  gates fails;
- `bond_growth_observed_pilot` if the identity gate passes and both rank gates
  fail;
- `invalid_arithmetic_contract` if the Gaussian identity gate fails.

Random separation and sparse improvement are mechanistic diagnostics and cannot
rescue a failed primary rank gate.

## Artifacts and execution

Expected artifacts:

```text
data/experiments/tinyllm_dynamic_ttno_rank/20260829_d8_babylm_pilot/
  campaign_results.json
  runs/seed_<seed>/result.json
```

Planned command:

```bash
pixi run python -m experiments.structure_net.tinyllm_dynamic_ttno_rank \
  --device cpu \
  --output data/experiments/tinyllm_dynamic_ttno_rank/20260829_d8_babylm_pilot
```

The producing runner, checkpoint, and token stream are content-hashed because
the repository is not clean at launch. Result JSON is atomic and rejects NaN.

## Method boundaries

- One pretraining checkpoint does not test variation across learned models.
- Five evaluation seeds sample correlated prefixes from one validation stream;
  they are not five model seeds.
- Length 32--256 cannot establish an asymptotic big-O law; the gates test finite
  compatibility with one declared polylog envelope.
- PCA ordering is an explicit core-generation cost and is not evidence for an
  end-to-end subquadratic compiler.
- Dense attention is materialized for diagnosis, so this run does not benchmark
  or implement subquadratic attention.
- HSS boundary ranks, H2 admissible-block ranks, and quantized TTNO ranks are
  related compression diagnostics but are not interchangeable.
- Quantizing an `n x n` position-mixing operator into `log2(n)` bit modes differs
  from the many-body TTNO construction for a tensor-product state with `n`
  physical sites described by Ceruti, Kressner, and Sulz (2024).

## Pre-outcome implementation amendment: Gram spectra

The first full launch was interrupted on 2026-08-29 after seven minutes because
no evaluation-seed record had completed; it produced no `result.json` or
`campaign_results.json`. Timing-only profiling identified the direct LAPACK SVD
as the bottleneck. Before inspecting any completed seed or printed primary rank,
the implementation was changed to compute the declared 1% and 0.1% relative
Frobenius ranks from the eigenvalues of the smaller Gram matrix. These
eigenvalues are algebraically the squared singular values and therefore leave
the estimator and thresholds unchanged. The `1e-12` sparse exact-rank diagnostic
retains direct SVD to avoid squaring its numerical condition. Focused tests and
the shakedown are rerun under the amended implementation, whose content hash
must differ from the interrupted launch.

## References

- Ceruti, Kressner, and Sulz, *Low-rank Tree Tensor Network Operators for
  Long-Range Pairwise Interactions*, https://arxiv.org/abs/2405.09952
- The motivating dynamic sparse-TTNO proposal supplied with this experiment.
