# Full summary semantic accuracy on fresh development sources

Date: 2026-10-24. Hypothesis `ncm-clean-full690-semantic-development-v1`.
Classification **clean_full690_semantic_development_quality_failed**. Campaign `data/experiments/ncm_paper_regime/20261024_clean_semantic_dev_results_v1/campaign_results.json`;
SHA256 `84114f770b11977fd6d64734d7562bc87c41c1f9efb61ab4fb50d10524a6992d`.

| Fresh development cohort | Judge | Complete summary passes | Valid | Task accuracy |
| --- | --- | --- | --- | --- |
| short_development | secondary | 0/64 | 64/64 | 0.00% |
| long_diagnostic | secondary | 0/64 | 64/64 | 0.00% |

The fixed final32768-update word student trained on all690 approved summaries.
All64 short and64 long sources come from fresh reserved components, with actual
TRAIN/family/text/turn exclusion replayed. No cases or first judgments were
removed. Judges permit paraphrases and assess faithfulness, salience, readability
and the128-token storage cap against the source. Exact reference-summary matching
is absent. Terra and conditionally admitted Sonnet retain their50/50 controls.

At least61 of the same64 cases must pass both judges in each cohort, with all
decisions valid. Reported rates and Wilson intervals describe model judgments;
they are not human ratings or a95% population confidence lower bound. The TRAIN
token-fit diagnostic does not determine semantic admission or task accuracy.

Original frozen Llama features, fixed source-only concept forecasts and one dense
word pass produce each complete192-position frame with global EOS. Raw TRAIN
constructor parity, fresh reload, retained inference pilots, unchanged models
and codec, memory bounds and exact anonymous public packets replayed. All1797
baseline first judgments and all new pilot decisions remain unchanged, with
1925 canonical firsts after this study. No negative was rerolled.

This fresh development result does not establish final95% accuracy. Sealed final
sources remain closed until a separately specified eligible final evaluation and
original frozen-body integration check. No inference speed gain is established.
