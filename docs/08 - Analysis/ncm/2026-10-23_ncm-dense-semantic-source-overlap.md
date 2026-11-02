# Source overlap invalidates the dense student generalization estimate

Date: 2026-10-23. Data audit `ncm-dense-semantic-source-overlap-v1`. Classification **development_split_contaminated**.
Campaign `data/experiments/ncm_paper_regime/20261023_dense_semantic_source_overlap_v1/campaign_results.json`; SHA256 `96fceebc76b6db036b35d06bff64103847d5cf3ebc3f5005d7a5e2d56171f700`.

| Original development cohort | Raw semantic passes | Cases with TRAIN source overlap | Passing outputs identical to TRAIN targets |
| --- | --- | --- | --- |
| short_development | 0/64 | 0/64 | 0 |
| long_diagnostic | 1/64 | 4/64 | 1 |

The original128-case study remains a recorded quality failure with all original
cases, predictions and first judgments preserved. Its metadata check established
distinct IDs, session IDs, group IDs and exact source hashes. It did not detect
related session-name variants, normalized duplicate passages or substantial text
containment. The recorded source-disjoint flag is therefore insufficient to
establish clean generalization. No cases are removed and no revised accuracy
is manufactured by changing the denominator.

The passing long-source prediction exactly reproduces a32-case TRAIN target.
Its development source `sharegpt_uD7CBng_2` differs from the training passage
`sharegpt_uD7CBng_4` by two characters; the shorter source is contained in the
longer. This is evidence of memorization on an overlapping passage.

This independent audit compares all690 approved TRAIN sources with all128
development sources. It joins observed sharegpt, hexadecimal and answer session
variants while preserving distinct ultrachat numeric IDs. It also tests shared
public-turn hashes, NFKC/case/word-normalized text equality and at least32 shared
word5grams covering80% of the shorter passage. The same checks audit pairs
inside development. These conservative checks do not prove all semantic
duplicates absent, but they catch the observed failure and must precede new
heldout selection. Conflict pairs and all raw cohort counts are retained.

Future student quality needs fresh development sources excluded by complete
source families, existing group IDs, public-turn identity and text overlap,
before outputs or judgments are produced. All690 approved TRAIN labels may
remain training data; they must not be called independent evaluation examples.
Both sealed final source files remain unopened and unhashed by this audit.
Their eligibility also needs metadata-based split checks before final use.
The frozen-body95% complete-summary goal remains unmet; no speed gain follows.
