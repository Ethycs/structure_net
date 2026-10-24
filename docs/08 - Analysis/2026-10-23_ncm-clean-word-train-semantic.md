# Semantic diagnosis of saved TRAIN summaries

Date:2026-10-23. Hypothesis ncm-clean-word-TRAIN-semantic-v1.
Classification **TRAIN_semantic_diagnostic_measured**. Campaign data/experiments/ncm_paper_regime/20261023_clean_word_train_semantic_results_v1/campaign_results.json; SHA256 c88711fd3af4d65c887c8aac4aeb092c82adb117f6dcbf374b2cd42a47cfbcec.

| Saved model and history | Terra whole-summary passes | Original TRAIN16 | Additional TRAIN16 | Sonnet |
| --- | --- | --- | --- | --- |
| previous_word__correct_history | 1/32 (32 valid) | 1/16 | 0/16 | not admitted |
| previous_word__actual_history | 1/32 (32 valid) | 1/16 | 0/16 | not admitted |
| clean_word__correct_history | 2/32 (32 valid) | 1/16 | 1/16 | not admitted |
| clean_word__actual_history | 1/32 (32 valid) | 1/16 | 0/16 | not admitted |

Every row uses the same32 TRAIN sources, not unseen evaluation data. Correct
histories supply oracle target-derived coarse concepts. Actual histories use
the existing252-case forecaster. This study measures no new source-disjoint
accuracy and does not reclassify the token-fitting gate or admit a queued GPU
study. A high TRAIN or oracle semantic pass rate cannot satisfy the95% goal.

The saved192-position raw outputs were decoded without repair or clipping and
paired with complete public sources. No model generation or training occurred.
Judges saw one anonymous candidate and source only, using the unchanged qualified
128-token semantic rubric with claim-level evidence citations. Paraphrases are
allowed. Primary Sonnet was admitted separately for each condition only after
32valid secondary Terra judgments and at least31passes. All invalid judgments
stay in their fixed denominators. Semantic pass fractions are reported only
when every case in that condition is valid. Judges are models, not human gold.

All1572 prior first judgments and all pilot first judgments were
preserved exactly. Identical packets reused the first decision, including
failures. Conditional primary admissions, raw requests/responses, complete
source/output bindings, token budgets, subgroup and same-case joint counts
replayed. The independent semantic development result remains0/64short+0/64long
for the preceding candidate. Both sealed final cohorts remain unopened; this
TRAIN diagnostic establishes neither NCM success nor speed/compression gains.
