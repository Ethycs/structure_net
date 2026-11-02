# Native BERT rendering of64 new teacher summaries

Hypothesis `ncm-sol64-native-text-roundtrip-semantic-v1`. Classification **fixed64_BERT_roundtrip_semantically_qualified**.

Campaign `data/experiments/ncm_paper_regime/20261024_sol64_bert_roundtrip_results_v2/campaign_results.json`, SHA256 `fc3219b54785da45db7ccdc43bf3d1810a8ba97d52e50fd0c29d44ac5fed2238`.

Every original first target was tokenized and rendered with the pinned native BERT tokenizer. No neural model, clipping, repair, candidate filtering or semantic reroll was used. The original raw summaries qualified62/64 jointly; their existing first judgments are reused.

| Judge | Semantic label passes | Valid |
| --- | ---: | ---: |
| secondary (codex_sdk/gpt-5.6-terra) | 63/64 | 64/64 |
| primary (codex_sdk/gpt-5.6-sol) | 62/64 | 64/64 |

Same-case joint result: `{'cases': 64, 'joint_passed': 62, 'all_valid': True, 'qualified': True}`. Paired transitions: `{'secondary': {'cases': 64, 'raw_passed': 63, 'rendered_passed': 63, 'retained_passes': 63, 'lost_passes': 0, 'gained_passes': 0, 'all_valid': True}, 'primary': {'cases': 64, 'raw_passed': 63, 'rendered_passed': 62, 'retained_passes': 62, 'lost_passes': 1, 'gained_passes': 0, 'all_valid': True}}`.

All3350 preceding first decisions remain unchanged; current canonical cache: 3478.

Judges assess complete-source faithfulness, salience and readability with the128 storage-token cap; character differences and exact-summary matches are not the semantic endpoint. These64 additional TRAIN-designated sources were gradient-disjoint from the original690 and128 DEV. This is a component rendering result, not source-only student accuracy or a ceiling on a source-conditioned decoder. The two automated judges share a provider; this is not independent human grading. No final source was opened or hashed; student task accuracy and speed remain unmeasured.
