# Whole public turns could not fill the clean NCM source quotas

Date: 2026-10-24. Original hypothesis `ncm-clean-full-size-source-partitions-v1`.
Classification **insufficient_component_disjoint_whole_turn_sources**. Campaign `data/experiments/ncm_paper_regime/20261024_clean_source_partition_failure_results_v1/campaign_results.json`;
SHA256 `733f237ccf38130e6fc3d735f96401ffd3781b76493f13d9c07444035d4f803f`.

| Source length band | Available components logged by worker | Required for development and final |
| --- | --- | --- |
| 0 | 11136 | 320 |
| 1 | 5513 | 32 |
| 2 | 571 | 32 |
| 3 | 16 | 32 |
| 4 | 16 | 32 |

Bands0–4 are192–511,512–767,768–1023,1024–1535 and1536–2048 cl100k tokens.
The actual public-data pilot passed. Full preparation built the76169-node
source graph, then exited1 at fixed-quota allocation. Each of the two longest
bands had16 eligible components, below the required32. No final or development
source exports were created and quotas were not reduced.

This posthoc audit binds the original producing source, immutable inputs,
complete log and actual subprocess receipts. Counts come from the frozen
worker's allocation exception; the graph was not independently recomputed by
this audit. The failure remains unchanged. A separately specified revision may
define exact bounded fragments from complete contiguous public exchanges before
model use, retaining the full128 development and320 final sizes, all input
bands and the95% semantic target. Generic short-turn hashes need special care
to avoid joining unrelated conversations through repeated greetings.

No model training, inference, labels or accuracy was measured. Both original
sealed source files remain closed. This result establishes a source-availability
failure for the whole-turn-only construction, not a student quality result.
