# Complete-summary accuracy with the training prompt date pinned

Date: 2026-10-24. Hypothesis `ncm-pinned-date-full690-semantic-development-v1`.
Classification **pinned_date_full690_semantic_development_quality_failed**. Campaign `data/experiments/ncm_paper_regime/20261024_pinned_date_development_results_v1/campaign_results.json`;
SHA256 `18c00793c40489c0f028d8dce895e5f00ed3f84ebe4e683dc997e9e810bfff47`.

| Fresh development cohort | Judge | Complete summary passes | Valid | Task accuracy |
| --- | --- | --- | --- | --- |
| short_development | secondary | 0/64 | 64/64 | 0.00% |
| long_diagnostic | secondary | 0/64 | 64/64 | 0.00% |

The fixed final32768-update word student trained on all690 approved summaries.
The SAME64 short and64 long sources previously failed with the24October prompt date.
This exploratory control changes only the date to explicit23October, with actual
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
and codec, memory bounds and exact anonymous public packets replayed. All1925
baseline first judgments and all new pilot decisions remain unchanged, with
2051 canonical firsts after this study. No negative was rerolled.

This matched-date development result does not establish final95% accuracy. Sealed final
sources remain closed until a separately specified eligible final evaluation and
original frozen-body integration check. No inference speed gain is established.
