# Full NCM evaluation sources reserved from public exchanges

Date: 2026-10-24. Hypothesis `ncm-clean-exchange-source-partitions-v2`.
Classification **clean_exchange_sources_reserved_quality_unmeasured**. Campaign `data/experiments/ncm_paper_regime/20261024_clean_exchange_partitions_results_v2/campaign_results.json`;
SHA256 `96f602f2f687a589f573a6e5966f07d0acee9a3b43d52e9184ec186359e17538`.

| Source partition | Cases |
| --- | --- |
| final_short | 256 |
| final_long | 64 |
| development_short | 64 |
| development_long | 64 |

Candidates contain whole public turns or contiguous complete user-led exchanges.
All constituent turn text and literal spans are retained. Selection precedes
model outputs and labels and uses no quality scores. Source components join
families, known group aliases, normalized text and exact word5gram overlap
(at least32 shared and80% shorter containment). Word and gram IDs use exact
dictionary equality. All known sources and old final metadata are excluded.

Shared-turn-hash aliases apply only to full turns of at least192 cl100k tokens,
so generic bypass greetings do not join unrelated families. Complete source
text still undergoes normalization and overlap checks; all source-family IDs
and all original provenance spans remain. The original whole-turn-only quota
failure stays failed. Full128 development and320 final quotas are unchanged;
both long cohorts retain16 sources in each of four512–2048-token bands.

The preparation worker checked exact public spans and direct full-text guards
against known sources and across all448 selected cases before sealing. The CPU
auditor independently replayed metadata, complete exchange boundaries, quota
selection, development bytes and final creation/stat receipts. Final files were
not reopened. Their content hashes and text guards need independent checks
when final evaluation becomes eligible. Neither original sealed file was opened.

No student quality, labels or speed improvement is established here. Future
development needs61/64 same cases passing both qualified judges per cohort;
final needs244/256 short and61/64 long, all valid, after fixed-candidate
development and original frozen-body integration checks. The95% task goal
remains unmet. All prior failures and canonical first judgments remain intact.
