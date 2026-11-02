# TinyLLM constructive causal H2 attention preregistration

**Status:** REGISTERED — NOT RUN  
**Date:** 2026-08-30  
**Menu ID:** A7  
**Hypothesis:** `tinyllm-causal-h2-attention-v1`  
**Planned schema:** `nal.tinyllm-causal-h2-attention.v1`  
**Evidence parent:** `tinyllm-dynamic-ttno-rank-pilot-v1`  
**Diagnostic predecessor:** `tinyllm-hss-shared-basis-nesting-v1`

## Question and primary lock

Does one simultaneous nested H2 approximation exist for the frozen causal
TinyLLM attention operators under a non-tunable strong-admissibility rule?

The primary construction is fixed as follows:

```yaml
tree: balanced chronological binary
leaf_size: 16
admissibility: gap_tokens >= max(query_block_size, key_block_size)
near_field: exact dense
rank_cap: ceil(log2(sequence_length)^2)
rank_selection: deterministic nested float64 SVD
build_tolerance: 0.0025
primary_kernel_error: weighted row-relative maximum <= 0.01
campaign_pass: at least 80% of cells at every primary length
```

There is no per-cell selection of tree, leaf size, admissibility, or rank cap;
no sparse top-k correction; no post-hoc clipping or row-sum repair; and no
rescue by choosing a sensitivity configuration. A6's result may explain A7 but
cannot tune this already-frozen construction.

## Frozen evidence and campaign scope

| Artifact | Required identity |
| --- | --- |
| A4 aggregate | `data/experiments/tinyllm_dynamic_ttno_rank/20260829_d8_babylm_pilot/campaign_results.json` |
| aggregate SHA-256 | `9d3fa8ec7332860785b3d62dff5805e4ec23c9f6fedc5178c6809b15cd05feba` |
| A4 implementation | `ffdf9bb77449a4dcad6c67f111b70a3543eae42495ab067044835d13bf65c8fb` |
| checkpoint SHA-256 | `5a7491b4231c30feaaabda7babf0a831dd4d72fbb4e7f3e757114cadf098df09` |
| token stream SHA-256 | `f339655453b970ae6cc1cbdc7c78f8a5234c42437bc7bba8fb31a1dee5c9d765` |

Reuse exactly the five validation-prefix seeds `101, 211, 307, 401, 503`, all
eight layers, and heads `0, 3, 7`: 120 frozen cells per length. These seeds are
input-selection replicates, not model seeds.

- Integrity length: `32`.
- Primary lengths: `64, 128, 256`.
- Conditional extension: `512`.

The frozen checkpoint has block size and learned positional embeddings of 256.
The 512 arm therefore requires a separately fingerprinted checkpoint trained
or explicitly validated at 512; it cannot change the primary verdict.

## Target operator and normalization pathway

For each cell, calculate float64 logits and the stabilized causal kernel:

```text
logit[i,j] = q[i]^T k[j] / sqrt(d_k)
m[i]       = max_{j <= i} logit[i,j]
K[i,j]     = 1[j <= i] exp(logit[i,j] - m[i])
d[i]       = sum_{j <= i} K[i,j]
A[i,j]     = K[i,j] / d[i]
```

Construct the H2 approximation to `K`, not to a pre-normalized surrogate. Then
calculate

```text
d_tilde = K_tilde @ 1
A_tilde = K_tilde / d_tilde[:,None]
```

and apply the same H2 matvec to the augmented values `[V, 1]`, producing the
attention numerator and denominator through one representation. Do not clip
negative entries or repair row sums.

Construct a direct H2 approximation of `A` under the identical tree and rank
budget as a diagnostic-only oracle. The kernel-normalized arm determines the
primary verdict.

## Chronological tree and block partition

Use one balanced binary tree for query and key token intervals, with separate
query and key bases. Split `[a,b]` at `floor((a+b)/2)` until each leaf contains
at most 16 tokens. Padding is excluded exactly from every scientific matrix and
metric.

For inclusive intervals `I=[a,b]` and `J=[c,d]`:

1. If `c > b`, classify the block as future and exactly zero.
2. If `d < a`, let `gap(I,J)=a-d-1`. Classify it admissible exactly when
   `gap(I,J) >= max(|I|,|J|)`.
3. If the nonzero block is inadmissible and both nodes are leaves, store the
   causally masked block densely and exactly.
4. Otherwise split every nonleaf member and visit the Cartesian product of its
   children in query-major, left-to-right order.

The partition is geometric; numerical rank may not determine admissibility.

### Partition integrity fingerprints

Canonical serialization uses zero-based inclusive intervals, depth-first
query-major traversal, and one line per terminal block:

```text
KIND:q_start-q_end:k_start-k_end
```

| Length | Admissible | Dense | Zero | SHA-256 of canonical partition |
| ---: | ---: | ---: | ---: | --- |
| 32 | 0 | 3 | 1 | `8e753f6334cd7d928b78b01fa4acc4310181ca053a81d42b72cd71c67b49d7e5` |
| 64 | 3 | 7 | 3 | `5a5a87fb6313c2dda8d7bdb1ce818ed8f81b931558cb9328f4b3aa3c83b77a76` |
| 128 | 12 | 15 | 7 | `97ac50d907bdfe6518af2e6d313bb62c6aebd284e2e8c7819955a8a763939600` |
| 256 | 33 | 31 | 15 | `1535f675b18b8493d5da1c4e1048eefb73c884e700953451de7f9b8b380688f2` |
| 512 | 78 | 63 | 31 | `79d9f9d12ff42883224ae0d0629933718503b800567203e04712f8765c3924e3` |

A mismatch invalidates the implementation before a scientific run. Length 32
has no far-field block and is an integrity check, not substantive evidence.

## Shared nested construction

For campaign length `n`, the rank cap is

```text
R(n) = ceil(log2(n)^2)
```

giving caps `25, 36, 49, 64, 81, 100` at lengths `32, 64, 128, 256,
512, 1024`. Node ranks must also obey node size and the sum of child ranks;
leaf ranks cannot exceed 16.

Every query node uses one basis shared over all admissible block rows it serves,
including inherited ancestor interactions. Every key node is treated
analogously. Separate per-block bases are forbidden.

Select bases from the row-mass-scaled operator `D^-1 K = A`, where
`D=diag(K @ 1)`, then rescale as required to represent `K`. Use deterministic
float64 SVD, not randomized sketching. For

```text
L_n = max(1, ceil(log2(n/16)))
epsilon_build = 0.0025
```

choose the smallest permissible node rank satisfying

```text
sum_{j>r} sigma_j^2 / sum_j sigma_j^2
    <= epsilon_build^2 / (2 L_n).
```

If the condition cannot be met, use the fixed cap and record
`rank_cap_hit=true`. Do not increase a rank after inspecting global error.
Couplings and transfer matrices must reconstruct a single simultaneous nested
operator and must support the prescribed upward, interaction, and downward
passes.

## Probe lock

The missing operational probe constant is frozen before implementation:

```yaml
probe_seed: 1707
bit_generator: numpy.random.PCG64
columns: 32
distribution: 2 * integers(0, 2) - 1
scope: reinitialize once per length and reuse the same matrix for every cell
```

Thus the smaller-length probe matrices are row prefixes of the larger matrices.
The probes are not used during basis construction.

## Numerical validity gates

Failure of any condition invalidates the record rather than counting as a
scientific failure. All construction and validation use float64.

| Integrity condition | Gate |
| --- | ---: |
| stabilized kernel attention versus model softmax | max absolute error `<= 1e-12` |
| block partition | exact count and fingerprint match |
| query/key basis orthogonality | spectral residual `<= 1e-10` |
| query/key parent-child nestedness | relative residual `<= 1e-10` |
| explicit `K_tilde @ x` versus H2 contraction | relative error `<= 1e-10` |
| future-token leakage | exactly zero or `< 1e-15` after assembly |
| NaN or nonfinite quantity | none |

For a parent basis and its two child bases, nestedness is the relative
Frobenius residual from the best transfer through their block-diagonal direct
sum. Query and key sides are checked separately.

## Cell-level scientific gates

A cell passes only when all six primary gates pass:

| Endpoint | Gate |
| --- | ---: |
| weighted row-relative kernel error `max_i sum_j |K-K_tilde| / sum_j K` | `<= 0.01` |
| denominator relative error `max_i |d-d_tilde|/d` | `<= 0.01` and `min(d_tilde)>0` |
| normalized attention row L1 error `max_i sum_j |A-A_tilde|` | `<= 0.025` |
| 32-column held-out Rademacher probe relative Frobenius error | `<= 0.02` |
| actual value-output relative Frobenius error | `<= 0.02` |
| p99 token output error normalized by global output RMS | `<= 0.05` |

Report but do not gate on maximum negative attention mass. Label a result
nearly positivity-preserving when this mass is at most `1e-3`.

## Campaign verdict

A primary length passes when at least 96 of its 120 cells pass all six gates.
All three primary lengths must pass. At length 256, every layer must have at
least 50% cell pass rate for a model-wide representation pass. The 80% rule is
a preregistered campaign threshold, not a population hypothesis test; report
prefix-clustered descriptive intervals.

Classification is applied in this order:

1. `invalid_h2_construction_contract` if a validity gate fails;
2. `h2_normalization_path_failed` if direct H2(A) passes its identical
   representation gates while kernel-normalized H2(K) fails;
3. `h2_representation_failed` if any primary length is below 80%;
4. `h2_layer_selective_only` if all length-level thresholds pass but any layer
   is below 50% at length 256;
5. `h2_representation_pass` otherwise.

The oracle arm only refines the cause of a kernel-arm failure and never rescues
the kernel-arm primary verdict.

## Finite-size compression gate

Count scalars exactly as

```text
C_H2 = C_near + C_leaf + C_transfer + C_coupling
C_dense = n(n+1)/2
```

Also count multiply-adds for the prescribed upward, interaction, and downward
passes on `d_v+1` channels. At the longest available length, compression passes
only when both storage and multiply-add ratios have median at most `0.75` and
90th percentile at most `1.0`.

- Representation plus compression: `h2_constructive_compression_pass`.
- Representation without compression: `h2_representation_pass_no_finite_size_compression`.

Wall-clock performance is explicitly deferred to A10.

## Locked sensitivity campaigns

Run only after freezing the primary classification, as complete separate
campaigns:

- leaf sizes `8` and `32`;
- separation ratios `0.5` and `2.0` times the larger block diameter;
- rank envelopes `0.5 R(n)` and `2 R(n)`;
- direct-A target versus kernel-normalized K target;
- a learned chronological contiguous tree only in a later campaign.

Sensitivity results explain but cannot change the primary verdict.

## Expected artifacts

```text
data/experiments/tinyllm_causal_h2_attention/20260830_registered/
  campaign_results.json
  runs/seed_<seed>/result.json
```

## Interpretation boundaries

- A7 tests existence of a simultaneous nested representation after dense
  materialization; it does not provide A9's implicit core compiler.
- Passing mathematical storage/operation counts does not imply measured speed.
- The single checkpoint cannot establish cross-model repeatability.
- Passing A7 supports an H2 endpoint, not automatically a strict TTNO.
- Failure under this fixed construction stops compilation of this chronological
  strong-admissibility candidate; sensitivities remain explanatory only.

## Background supplied with the proposal

- [Adaptive Sketching Based Construction of H2 Matrices on GPUs](https://arxiv.org/html/2506.16759)
- [Fast Multipole Attention for Transformer Neural Networks](https://arxiv.org/html/2310.11960v3)
