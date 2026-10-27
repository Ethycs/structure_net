# Counted supply in the 500 original summary stores

The complete core-store census finds **85,471 distinct non-verbatim source–summary pairs that fit the application length limits**, before source-disjoint splitting, native-codec capacity checks and label-quality review. This is a measured core-store count, not the total available across the broader archive, whose census is running.

| Population | Distinct pairs | Distinct inputs | Fits the applicable length limits |
|---|---:|---:|---:|
| Non-verbatim atomic source → summary | 81,819 | 76,218 | 81,819 |
| Ordered child summaries → stored parent summary | 111,245 | 111,245 | 3,652 |
| Exact source copies | 137,300 | — | Excluded from generated-summary count |

Atomic inputs use 192–2,048 `cl100k_base` tokens and targets at most 128 tokens. The qualifying merge inputs are at most 2,048 tokens and targets at most 128; no raw-fragment bypass threshold is imposed on merges. Another 107,593 recovered merge targets exceed 128 tokens. A structural child-summary input is reconstructed from stored children; that does not establish recovery of the exact original model prompt or prove a model generated the parent.

All 500 stores contain 541,010 section records: 270,805 atomic, 246,422 hierarchy and 23,783 parent records. The atomic scan verifies exact raw span bounds, hashes and attribution. There are 219,119 distinct raw input–target pairs after global text deduplication, including the 137,300 exact copies. Another 111,245 distinct pairs have recoverable ordered child inputs. The remaining 158,921 structural section records have no complete child input under this join contract and remain explicitly unresolved; they are not silently counted as usable teacher pairs.

An independent SQL reconciliation reproduces the old export's 270,805 physical atomic records and 227,744 pairs under its source-namespace key. Removing the source name from the content identity eliminates a further 8,625 duplicate pairs. Several targets for one raw input remain distinct pair records, explaining the 81,819-versus-76,218 difference.

These candidates include protected sources and unreviewed historical summaries. They have not been admitted to training, scored as semantic successes, or used to open the final evaluation. The 2,384 prepared application pairs are therefore not a measure of the total saved data supply. Four focused count/join tests pass, including global deduplication, rejection of corrupt spans, request-label ordering and exclusion of reader questions from merge requests.

Machine-readable evidence: `data/experiments/ncm_paper_regime/20261027_available_pair_census_fast_v1/core_pair_supply.json`, `core_count_reconciliation.json`, and `core_result.json`. The broader census restores all 798 published bundles, counts database and cache records, joins saved requests, and inspects files omitted by the sender's filename-based index. No total for that unfinished work is claimed here.
