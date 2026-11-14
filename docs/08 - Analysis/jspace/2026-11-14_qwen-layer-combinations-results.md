# Qwen recovered-layer compatibility search

Study **JSPACE-QWEN-LAYER-COMBINATIONS-01**. Classification: `additional_tested_compression_passes_development_gate`. **Confirmed: false.**

The most compressed **tested** model meeting the47/64 development gate uses **5/36 compact blocks** and scores **48/64 (75.00%)**. The original native control scored50/64. This layer count is the primary metric; it is not model accuracy or a claim of global optimality.

The search evaluated **786 unique masks** on the same64 previously inspected development cases. All36 compact blocks reproduce27/64; compact blocks0/1 with a native suffix reproduce49/64; restoring native12/15 reproduces24/64. No weights changed, no repair training occurred, and the untouched confirmation set was not opened.

Most-compressed passing mask: `m000001100010100000000010000000000000`. Compact layers: `[5, 6, 10, 12, 22]`. Conditional-choice CE: 4.053233; native decision agreement: 79.69%.

| Background outside six candidates | Compact blocks | Tested at count | Best /64 | Worst /64 | Best switched group | Worst switched group |
|---|---:|---:|---:|---:|---|---|
| outside_native | 0 | 1 | 50 | 50 | [] | [] |
| outside_native | 1 | 6 | 50 | 44 | [22] | [12] |
| outside_native | 2 | 15 | 52 | 42 | [6, 22] | [10, 12] |
| outside_native | 3 | 20 | 52 | 42 | [6, 8, 22] | [5, 8, 12] |
| outside_native | 4 | 15 | 48 | 43 | [5, 6, 10, 22] | [5, 6, 12, 22] |
| outside_native | 5 | 6 | 48 | 38 | [5, 6, 10, 12, 22] | [5, 6, 8, 10, 12] |
| outside_native | 6 | 1 | 43 | 43 | [5, 6, 8, 10, 12, 22] | [5, 6, 8, 10, 12, 22] |
| outside_compact | 36 | 1 | 27 | 27 | [] | [] |
| outside_compact | 35 | 6 | 29 | 23 | [22] | [10] |
| outside_compact | 34 | 15 | 31 | 26 | [12, 22] | [5, 8] |
| outside_compact | 33 | 20 | 32 | 24 | [5, 10, 12] | [5, 8, 12] |
| outside_compact | 32 | 15 | 31 | 26 | [6, 10, 12, 22] | [5, 6, 8, 12] |
| outside_compact | 31 | 6 | 33 | 28 | [5, 6, 8, 10, 22] | [5, 6, 8, 12, 22] |
| outside_compact | 30 | 1 | 34 | 34 | [5, 6, 8, 10, 12, 22] | [5, 6, 8, 10, 12, 22] |

In `outside_native`, switched candidate layers are compact; in `outside_compact`, switched layers are restored to native. Each best/worst comparison holds the number of compact blocks constant. Ties follow correct count, native agreement, lower conditional CE, then lexical layer indices.

The deterministic candidate set was `[5, 6, 8, 10, 12, 22]` (top_two_single_rescues_then_positive_pair_excess_rescue_then_single_fill). The selected three-layer repair target is **[5, 10, 12]**: restoring those layers to native in the otherwise compact model scores **32/64**, a change of **+5** from27/64. Paired changes: 7 gained and 2 lost correct answers. This conditional native-restoration result does not demonstrate that training the compact group can learn the rescue.

| Selected example | ARC /16 | BoolQ /16 | HellaSwag /16 | WinoGrande /16 | Gains vs native | Losses vs native |
|---|---:|---:|---:|---:|---:|---:|
| Most compressed passing | 14 | 12 | 10 | 12 | 5 | 7 |
| Chosen triple restored | 6 | 9 | 7 | 10 | 7 | 25 |

![All630 pair excess-rescue values](../../../data/experiments/jspace_layer_combinations/20261114_v1/pair_excess_rescue.png)

Every point uses the all-compact background. Excess rescue is correct(i,j)−correct(i)−correct(j)+27; it is a descriptive, background-dependent interaction count. The symmetric plot contains630 unique pairs, not independent replications or a causal decomposition.

Shared-prefix execution was checked against independent paths, all four controls passed exact predictions and numerical logit tolerance, and all resident parameters/buffers remained unchanged. Runtime timings and decoder-call savings describe search execution only; there is no deployment token-throughput claim. Repeated development selection does not supply independent confirmation.

Evidence: `data/experiments/jspace_layer_combinations/20261114_v1/layer_combinations_analysis.json`, SHA256 `25d7313caf77a1ea814162e5f71ba144ddefdedb1ae44a303c3d5915d0c8d008`. Pair data: `data/experiments/jspace_layer_combinations/20261114_v1/pair_excess_rescue.csv`, SHA256 `421506b27c0dc9cc9649ba6715cd2d012a11ea0587a4c31bfd27917b0fa69b6e`. Full successful search, mask completeness, deterministic selection, per-case scores and archived/live source bytes authenticated. Small inputs rehashed; large fixed weights retain producerSHA256 plus immutable file identities, and each selected state was hashed during bank loading.
