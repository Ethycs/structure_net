# Frozen-codec semantics on64 new summaries

Hypothesis `ncm-sol64-frozen-codec-semantic-v1`. Classification **fixed64_codec_component_quality_failed**.

Campaign `data/experiments/ncm_paper_regime/20261024_sol64_codec_semantics_results_v1/campaign_results.json`, SHA256 `101e31047826cc73a3a27030cd973f9f007abc85985e6ed2ac030f4e10c35e19`.

Both codecs received the same oracle target codes from every original first summary; the two word decoders were frozen. Original sources, complete128-position outputs and the native renderer were retained without truncation or repair. The separate numeric workflow completed on GPU1 at834MB peak allocation and passed an independent CPU checkpoint-binding and rendering audit.

| Codec | Judge | Whole-summary passes | Valid |
| --- | --- | ---: | ---: |
| released | secondary | 52/64 | 64/64 |
| domain | secondary | 60/64 | 64/64 |

Same-case joint gates: `{'released': None, 'domain': None}`. Qualified codecs: `[]`.

Paired raw/BERT control transitions: `{'released': {'secondary': {'raw': {'cases': 64, 'raw_passed': 63, 'rendered_passed': 52, 'retained_passes': 52, 'lost_passes': 11, 'gained_passes': 0, 'all_valid': True}, 'bert': {'cases': 64, 'raw_passed': 63, 'rendered_passed': 52, 'retained_passes': 52, 'lost_passes': 11, 'gained_passes': 0, 'all_valid': True}}}, 'domain': {'secondary': {'raw': {'cases': 64, 'raw_passed': 63, 'rendered_passed': 60, 'retained_passes': 60, 'lost_passes': 3, 'gained_passes': 0, 'all_valid': True}, 'bert': {'cases': 64, 'raw_passed': 63, 'rendered_passed': 60, 'retained_passes': 60, 'lost_passes': 3, 'gained_passes': 0, 'all_valid': True}}}}`.

Canonical first judgments retained: 3523.

Semantic judges accept paraphrases and assess full-source faithfulness, salience, readability and the128 storage-token cap. High word-ID agreement does not establish this endpoint. These64 additional TRAIN-designated sources were excluded from previous gradients. This component result is not source-only summarization or a ceiling on a source-conditioned decoder. The two automated judges share a provider and are not independent human annotation. No sealed final source was opened or hashed; the95% student goal and speed remain unestablished.
