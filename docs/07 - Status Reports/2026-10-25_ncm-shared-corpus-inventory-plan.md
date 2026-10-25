# Reuse the larger shared summarization corpus under current native limits

2026-10-25. Exploratory data inventory
`ncm-shared-summary-native128-inventory-v1`. The user suggested the large DVC
summarization dataset. The likely matching previously shared corpus is already
restored from `Ethycs/memory_condense`, Git
`fc25ea0d3cccc589346ddc13747218b8dc7a1221`, lakeFS
`82d974caec273e113dd864f9da352c003540e9225fbcce3766a81c103e005dab`.
Its existing manifest reports 204,967 TRAIN atomic source/summary pairs, of
which 45,136 fit the old 512-GPT2-input/48-output regime. The separately
partitioned 22,777 old test pairs are not new untouched benchmark data.

Audit the complete original `train.jsonl` only. Verify its manifest-pinned hash,
every row's original source and target hashes, original training split and
recorded application token counts. No resummarization, trimming or target
changes. Recount application tokens and tokenize original targets with the
current pinned BERT tokenizer. Count all exclusions independently, then retain
only inputs of 192–2,048 cl100k_base tokens and nonempty targets of at most 128
cl100k_base tokens that fit CLS + at most126 native lexical IDs + CLS, without
embedded native special tokens. Record excluded UNKs and length distributions.
Keep every rejected row represented in aggregate counts.

Preserve all existing clean evaluation reservations. Read only the audited
component index and selection metadata, plus the existing source-only metadata
of the held-out64 teacher/codec component cohort. Build protected source-family,
group, exact-source-hash and informative-turn sets from EVERY row in all
reserved components. Exclude matching training records. Do not open or hash
sealed final source files, and do not reconstruct their text. These are
metadata guards, not a new full-text near-duplicate audit or a final quality
qualification. A later training-admission study must audit overlaps against
open development sources and retain the final evaluation's independent guards.

This stage exports accepted TRAIN record metadata/line numbers for later
hydration, token histograms and all counts. It does not launch model training,
choose checkpoints, make judge calls, rewrite existing splits or qualify saved
summaries as semantically correct. The broader corpus has saved application
summaries, not independently human-verified references. It is a candidate
training source, not evidence for 95% task accuracy.

Use a fresh `20261025_shared_summary_native128_inventory_v1` root under
`data/experiments/ncm_paper_regime/`, CPU only, max RSS8GB. Preserve producer,
original manifest identity, all accepted record identities and exclusion
counts. Stop on any mismatch. Never modify the original data or prior runs.
