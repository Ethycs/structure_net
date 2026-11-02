# Pretrained NCM source-only development result

Date: 2026-10-24. Hypothesis `ncm-lm1b-native-word-readout-source-only-DEV-v1`. Classification **full128_source_only_DEV_quality_failed**.
Campaign `data/experiments/ncm_paper_regime/20261024_lm1b_native_word_readout_results_v1/campaign_results.json`, SHA256 `fd7f7b96be952c19f723018baa074c0690a2f761bd500f3328e9bae263d71d84`.

The authors' released LM1B generator and task-adapted codec were frozen while
11,796,480 new source-attention parameters trained for4096fixed updates on690
examples. The independent CPU auditor replayed all65,536 sampled examples,
all neighborhood corruption and960source-only self-history microbatches.
Full precision source-only inference used every unchanged permanent DEV source,
without target words or codes. This is conditional task adaptation, not a paper replication.

| Cohort | Judge | Complete-summary passes | Valid |
| --- | --- | ---: | ---: |
| short_development | secondary | 0/64 | 64/64 |
| long_diagnostic | secondary | 0/64 | 64/64 |

Both-cohort same-case DEV qualified: **False**. Primary admitted: **False**. Canonical first judgments retained: 2728.

The semantic judges assess supported claims, salience and readability; exact
summary matches are not required. Nonempty output and128stored-token cap
remain mandatory. Invalids are retained. Every raw output, failure and prior
first judgment is preserved. Previous old-student0/128 and codec failures
remain separate evidence. The final320sources remain closed and unhashed;
final task accuracy and speed are unmeasured. This DEV result does not
complete the95%task objective.

This follow-up changes only word selection. The same fixed4096 model
reproduced all baseline concept IDs and unpenalized word predictions exactly.
The unchanged author pick_tokens function was extracted from the pinned
evaluation script: frequency11.5, deficit2, adjacency10/window4/decay0.5
and the exact released token-unigram allowance, computed in FP64. All
128positions of every logit frame were saved; CPU replay matched every
selected word ID, including both retained inference pilots.

Concept generation stays at the baseline deterministic policy and the
existing predicted-CLS renderer stays unchanged. This is a word-readout
control, not a full replication of author unconditional sampling. No
further weights were trained and no prior judgment was rerolled.
Final campaign SHA256: `c8164656fc0d434c0584a8a8e0f9c16a8de8f9d1e3b356433fbf9847bf0aa9b7`.
