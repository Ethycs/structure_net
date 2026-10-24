# Native NCM packed and sequential history diagnostic

Hypothesis `ncm-native655-history-path-parity-v1`. Classification **two_TRAIN_packed_sequential_paths_aligned_at_identical_histories**.

The old native655 checkpoint was frozen. The first and longest original TRAIN cases were selected before execution; source-only DEV and sealed final sources were not used. With identical oracle histories, packed and sequential inference agree within1e-4 at every level and have zero argmax disagreements across both641-code sequences. The largest absolute logit difference is 5.34057617e-05.

| TRAIN case | Gold-history codes correct | Free-generated codes correct |
| --- | ---: | ---: |
| 0ef3ef09ae79 | 636/641 | 615/641 |
| f61dd06534ee | 635/641 | 543/641 |

Generated histories increase later-level code loss. Swapping the two source memories changes452and504packed code choices, respectively, so this trained head is sensitive to source conditioning on these probes. This supports investigating accumulated history errors; it does not establish their share of held-out semantic failures or prove source understanding. Two TRAIN cases cannot rule out all implementation bugs. These code-match counts are not whole-summary accuracy. The independent CPU audit replayed every metric from the retained raw logits and original TRAIN targets.
