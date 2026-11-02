# Fixed native baselines evaluated with the application budget

Date: 2026-10-23. Exploratory `ncm-budget128-fixed-baselines-v1`.
Classification: `fixed_baseline_re_evaluation_complete_student_quality_separate`.
This compares the same frozen outputs under two explicit rubrics. No model
weights, prompts, generation settings or candidate texts changed.

| Baseline | Old-rubric whole pass | New-rubric whole pass | New valid | Faithful | Salient | Readable | Storage overflow |
| --- | --- | --- | --- | --- | --- | --- | --- |
| native_3b | 49/64 | 52/64 | 64/64 | 52/64 | 63/64 | 63/64 | 0 |
| native_1b | 50/64 | 47/64 | 64/64 | 49/64 | 61/64 | 64/64 | 0 |

The new qualified rubric removes an inherited approximately 32-word salience
target while retaining the 128-token storage cap, source-grounded faithfulness,
coverage and readability. Both judge aliases previously passed all 50 controls.
These are fallible LLM assessments, not independent human gold.
Paired whole-decision changes are `{'native_3b': {'old_fail_new_pass': 4, 'old_pass_new_fail': 1}, 'native_1b': {'old_fail_new_pass': 2, 'old_pass_new_fail': 5}}`. These fresh judgments
cannot isolate a rubric effect from judge variability and do not improve model
weights. No failed source is removed from either comparison.

All 64 original development sources per baseline were retained in order.
Over-cap, empty and poor-quality outputs remain in the denominator. First
new-profile pilot judgments, including negatives, were reused exactly; no
old-profile decision was passed off as a new-profile result. Raw request/parser
and first-valid decisions were replayed before finalization. Old aggregates
and caches remain unchanged. Wilson intervals and conditional independent
Sonnet outcomes are in the aggregate; Sonnet runs only after 64 valid Terra
judgments and at least 61 whole passes for that baseline.

These intact AR baselines are controls for the current frozen-body NCM study.
They cannot establish student accuracy, a working NCM replacement, compression
or speedup. The student must generate and pass its own source-disjoint summaries.
The final 256 cases remain sealed and the short-source prototype does not prove
the complete 2,048-token fragment / 6,300-token batch application range.

Aggregate `data/experiments/ncm_paper_regime/20261023_budget128_baselines_results_v1/campaign_results.json`, SHA256 `a4987be44401ee658de403c23c967a40dfaa6ed880e677e449f0d58cf2c4c8a4`. Registry confirmed=false.
