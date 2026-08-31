# TinyLLM TTNO cut-localization and parity-control preregistration

**Status:** REGISTERED — NOT RUN
**Date:** 2026-08-30
**Menu ID:** A5
**Hypothesis:** `tinyllm-ttno-cut-localization-parity-v1`
**Planned schema:** `nal.tinyllm-ttno-cut-localization-parity.v1`
**Parent:** `tinyllm-dynamic-ttno-rank-pilot-v1`

## Question

Was the measured 128-to-256 paired-bit TTNO rank cliff intrinsic to the frozen
attention operators, caused by the new balanced paired-bit cut introduced at
eight index bits, or avoidable under another fixed bit-mode topology?

This is a post-A4, preregistered diagnostic. It does not retest whether A4
passed. A4's `bond_growth_observed_pilot` classification remains fixed.

## Frozen parent evidence

| Artifact | Required identity |
| --- | --- |
| A4 aggregate | `data/experiments/tinyllm_dynamic_ttno_rank/20260829_d8_babylm_pilot/campaign_results.json` |
| aggregate SHA-256 | `9d3fa8ec7332860785b3d62dff5805e4ec23c9f6fedc5178c6809b15cd05feba` |
| A4 implementation | `ffdf9bb77449a4dcad6c67f111b70a3543eae42495ab067044835d13bf65c8fb` |
| model checkpoint SHA-256 | `5a7491b4231c30feaaabda7babf0a831dd4d72fbb4e7f3e757114cadf098df09` |
| token stream SHA-256 | `f339655453b970ae6cc1cbdc7c78f8a5234c42437bc7bba8fb31a1dee5c9d765` |

The study reuses exactly the five validation-prefix seeds `101, 211, 307,
401, 503`, all eight layers, heads `0, 3, 7`, and nested lengths `32, 64, 128,
256`. Evaluation seeds remain input-selection replicates, not model seeds.

## Rank estimator

Use A4's paired operator tensor and relative-Frobenius numerical rank at
`epsilon = 1e-2` (primary) and `1e-3` (sensitivity). For every non-root edge,
retain:

- the paired bit-mode subset;
- matricization shape;
- numerical rank;
- fraction of the maximum rank at that length;
- whether the edge is newly possible at 256 tokens.

The A4 MSB-first balanced tree must reproduce its stored per-edge ranks exactly
before any new arm is interpreted.

## Fixed topology arms

Every arm is a deterministic transformation of the same frozen operator. No
rank outcome may select a topology.

| Arm | Definition | Purpose |
| --- | --- | --- |
| `msb_balanced` | A4 paired modes in most-significant-bit order | exact replay baseline |
| `lsb_balanced` | reverse the paired bit-mode order before the same balanced recursion | move coarse/fine cuts |
| `odd_even_modes` | odd-index paired modes followed by even-index paired modes | break the A4 root partition |
| `gray_token_order` | simultaneous row/column binary-reflected Gray-code permutation, then MSB tree | parity-local token ordering |
| `zero_pad_128_to_256` | place A128 in the upper-left 128 block of a zero 256 operator | isolate tensor-shape/cut introduction |
| `duplicate_128_to_256` | direct sum `A128 + A128`, equivalently `I2 kron A128` in chronological order | introduce the new high bit while repeating the same operator |
| `natural_256` | actual A256 | measured cliff target |

Zero padding is not row stochastic and is used only as an operator-tensor rank
control. The duplicated arm is the stronger function-preserving parity control
on the two 128-position sectors.

## Primary endpoints

For each of the 120 frozen cells define:

```text
r128       = maximum primary rank of natural A128 under msb_balanced
r256       = maximum primary rank of natural A256 under msb_balanced
rzero      = maximum primary rank of zero_pad_128_to_256
rduplicate = maximum primary rank of duplicate_128_to_256
ralt       = minimum natural-A256 rank over the three non-baseline topologies
cliff      = max(r256 - r128, 1)
zero_artifact_fraction      = clip((rzero - r128) / cliff, 0, 1)
duplicate_artifact_fraction = clip((rduplicate - r128) / cliff, 0, 1)
topology_reduction          = (r256 - ralt) / r256
```

Also record the critical A128 and A256 edge, the rank added at every homologous
edge, and the fraction of the total cliff attributable to the new root split.

## Gates and classification

| Gate | Rule |
| --- | --- |
| validity | 120/120 cells reproduce A4 ranks and all transformed tensors round-trip exactly |
| parity artifact | median of the larger zero/duplicate artifact fraction is at least `0.75` |
| topology sensitivity | at least 80% of cells obtain topology reduction at least `0.25` |
| intrinsic persistence | median of the larger artifact fraction is below `0.25` and fewer than 20% of cells obtain 25% topology reduction |

Classification is fixed in this order:

1. `invalid_parent_or_tensor_contract` if validity fails;
2. `new_cut_artifact_dominant` if the parity-artifact gate passes;
3. `bit_topology_sensitive` if topology sensitivity passes;
4. `intrinsic_operator_rank_growth` if intrinsic persistence passes;
5. `mixed_cut_and_operator_effect` otherwise.

The 0.1% ranks, per-layer patterns, and which alternative topology wins are
secondary diagnostics and cannot change the classification.

## Conditional 512-token extension

The current BabyLM checkpoint has block size and learned positional embeddings
of length 256. Consequently, natural 256-to-512 attention is unavailable under
the frozen parent model. A later 512-token arm may be added only as a separately
fingerprinted extension using a checkpoint trained or explicitly validated at
that length. It cannot retroactively change this study's primary classification.

## Expected artifacts

```text
data/experiments/tinyllm_ttno_cut_localization_parity/20260830_registered/
  campaign_results.json
  runs/seed_<seed>/result.json
```

## Interpretation boundaries

- The experiment localizes the A4 paired-bit rank cliff; it does not construct
  a TTNO or H2 operator.
- Alternative bit-mode orders are fixed topology controls, not an exhaustive
  tree search.
- Zero padding and sector duplication diagnose representation effects but are
  not natural language attention distributions.
- A topology-sensitive result would motivate tree search; it would not establish
  a subquadratic input-conditioned tree compiler.
