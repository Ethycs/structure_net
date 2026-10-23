# Wider public source pool prepared; no model quality measured

Date: 2026-10-23. Exploratory data preparation,
`wider_public_sources_prepared_quality_unmeasured`.

The full original training partition contains 204,967 rows. After length,
known-source and exact canonical-public-span checks, 14,638 rows from 3,860
unused groups remain. Availability by cl100k source-length band:

| Tokens per complete source | Eligible rows | Unique groups |
| --- | --- | --- |
| 512–767 | 13770 | 3668 |
| 768–1023 | 623 | 465 |
| 1024–1535 | 84 | 69 |
| 1536–2048 | 161 | 122 |

The rarest band has 69 groups, 18 shared with the longest band. Before any
pool model outputs or judgments, fixed quotas reserved 16 sealed-final, four
diagnostic and 28 training sources per band: **64 sealed, 16 diagnostic and
112 training sources**, from 192 distinct groups. Deterministic metadata replay
verified no ID/session/group/source-hash overlap or known-source reuse.

Complete selected sources contain 513–2048
cl100k tokens; native prompts contain 740–2276
tokens including unchanged native chat instructions. No truncation or short
native-context filter was used. Stored source lengths and exact public spans
were verified during preparation. Old saved targets were excluded from exports.
The finalizer checked export hashes and metadata without loading sealed texts.

This operational holdout comes from unused groups inside the original training
partition; it is not an untouched original benchmark partition. Both source
workers terminated successfully. Original sealed 256 cases and the new sealed
64 cases have no new outputs or labels from this preparation. The currently
running short-source NCM experiment remains unchanged and does not use this pool.

The pool is unlabeled and admits no new training run. It addresses source
availability, not 95% accuracy, batch/hierarchy quality, native memory fit,
compression or speed. Future labels, long-context lifecycle tests and complete
source-disjoint task evaluation need their own fixed protocols.

Aggregate `data/experiments/ncm_paper_regime/20261023_application_source_pool_results_v1/campaign_results.json`, SHA256 `6e3650aa71f2308447031176125597ba3226650bff24a594c14debeebcf5b848`. Confirmed=false.
