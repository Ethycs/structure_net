# Sol-reviewed TRAIN-label semantic qualification

Hypothesis `ncm-sol-evidence64-teacher-semantic-qualification-v1`. Classification **fixed64_sol_teacher_labels_semantically_qualified**.

Campaign `data/experiments/ncm_paper_regime/20261024_sol_teacher64_semantic_results_v1/campaign_results.json`, SHA256 `2d6ee5f2dbca777bfb0774569f2bd7e69ba426245dd0247f80a04d62f80d1b4b`.

All 64 first candidates were judged against complete sources with the unchanged 128-token semantic rubric. No exact-summary matching, filtering or semantic retries were used.

| Judge | Semantic label passes | Valid |
| --- | ---: | ---: |
| secondary (codex_sdk/gpt-5.6-terra) | 63/64 | 64/64 |
| primary (codex_sdk/gpt-5.6-sol) | 63/64 | 64/64 |

Same-case joint result: `{'cases': 64, 'joint_passed': 62, 'all_valid': True, 'qualified': True}`. All 3,222 preceding first judgments remain; current cache: 3350.

This measures teacher labels, not NCM student accuracy. Sol replaced a quota-exhausted reviewer after passing all 50 fixed controls and raw replay. The two automated judges share a provider and are not independent human raters. The original student failure remains reported; final sources remain closed and unhashed.
