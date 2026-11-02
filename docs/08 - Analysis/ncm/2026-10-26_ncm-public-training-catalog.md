# 162,353 public source-summary pairs are admitted for conditional pretraining

The complete data replay passed. Study `ncm-public-training-catalog-v1` is
classified `full_public_paired_catalog_admitted_for_pretraining`. It admits
162,353 unchanged XSum TRAIN pairs, roughly118times the previous1376-pair
reviewed training catalog, while preserving320 public diagnostic articles
outside TRAIN. No model was trained and no new task-accuracy result exists.

The audited aggregate is
`data/experiments/ncm_paper_regime/20261026_public_training_catalog_audit_v1/campaign_results.json`,
SHA256 `81ceb20ac7261e81706191fc3cde5e25742c1bdcd55ace8b77b6cf1db42e1985`.
The [prospective plan](../07%20-%20Status%20Reports/2026-10-26_ncm-public-training-catalog-plan.md)
fixed full-population retention, protection, deduplication and token rules
before this preparation pass.

| Population accounting | Rows |
| --- | ---: |
| Original pinned XSum TRAIN shard | 204,045 |
| Excluded by the previous token/source inspection | 40,994 |
| Entering the new catalog screen | 163,051 |
| Further exclusions | 698 |
| Retained paired examples | 162,353 |

The698 further exclusions consist of320 reserved public diagnostic articles,
one additional match to a protected/current-training target, and377 repeated
native target frames. The protection predicates overlap:321 rows match both
protected exact target text and native target frames;320 also match reserved
article IDs, source hashes and source text. These overlapping predicate counts
must not be added as separate excluded populations.

All320 public controls retain their previous cohort assignments:256
representative,32 longest-target stress and32 numerical stress. The catalog
also excludes exact/native-equivalent targets of the64 reserved application
codec cases and1376 existing reviewed training cases. Public source comparisons
use normalized equality or at least32 common distinct word5-grams with80%
shorter-text containment. No additional near-source conflict beyond the320
reserved articles was found among the previously eligible candidates.

Every retained pair preserves the original article and reference string, its
official dataset revision, article ID, original row number and exact hashes.
All candidate source and target token counts were recomputed twice, once during
preparation and once in a separate audit. Sources remain192–2048 cl100k tokens;
targets remain1–128 cl100k tokens and1–126 native lexical tokens. Targets use
the unchanged128-position native frame. There was no truncation or rewriting.
The final catalog contains88,780,546 source cl100k tokens and4,320,525 native
target lexical tokens.

The separate audit replayed all204,045 original source/target bindings, all
163,051 candidate tokenizations and new public-development source-overlap
checks, every exclusion and native-target deduplication decision, and all
162,353 UTF-8 record offsets. It checked that indexed records cover the payload
with no missing, duplicated or trailing rows. Preparation took230seconds and
peaked at624MB RSS; the replay took198seconds and peaked at675MB RSS. These are
data-preparation measurements, not model speed measurements.

The readable training JSONL is705,566,539bytes; its offset/provenance index is
119,231,722bytes. Original public-development pairs and exclusions are stored
separately. Access paths:

- `data/experiments/ncm_paper_regime/20261026_public_training_catalog_v1/training.jsonl`
- `data/experiments/ncm_paper_regime/20261026_public_training_catalog_v1/training_index.jsonl`
- `data/experiments/ncm_paper_regime/20261026_public_training_catalog_v1/public_development.jsonl`
- `data/experiments/ncm_paper_regime/20261026_public_training_catalog_v1/exclusions.jsonl`

The admission scope is public-reference conditional pretraining. XSum references
were not individually checked for factual correctness, and news article IDs do
not establish every related story family. The earlier inspection screened192
open application sources and512 protected components' metadata; sealed final
full texts remain unopened. Arbitrary semantic paraphrases and complete final
text disjointness have not been established. The320 controls are development
data whose earlier codec outputs are known, not a new sealed final test.

Thirty focused tests cover the catalog, index audit and registry boundaries,
including the previous data contracts. Original failures and all4179 canonical
semantic judgment firsts are unchanged. Registration remains conservative:
data admission does not confirm the95% complete-summary task hypothesis.

The next stage is full source-only prompt/alignment preparation followed by a
bounded feature pipeline and a larger conditional training run. Using cl100k
positions as a proxy, a whole-corpus FP32 feature cache would require about
1.09TB; actual GPT2-aligned positions are being measured separately. The full
corpus is retained rather than reduced to fit that eager cache. The latest
completed application quality result remains0/128 semantic passes per matched
arm; this data preparation does not change it.
