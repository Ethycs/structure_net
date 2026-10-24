# Full-native NCM: sixteen-summary TRAIN learnability diagnosis

Hypothesis `ncm-lm1b-native16-TRAIN-semantic-v1`; campaign `data/experiments/ncm_paper_regime/20261024_lm1b_native16_semantic_results_v1/campaign_results.json`, SHA256 `ddb6ad51a9b0af01012121e139cdae6891fd3c52ffd95c4133e358b5a65df60a`.

The authors' pretrained generator, domain word decoder and source attention
were trained together:127,732,736parameters,256fixed updates,16TRAIN cases.
The native encoder/VQ and original Llama body stayed frozen. Both pilot and
full control reset the identical warm initializer; pilot weights were discarded.

Both calibrated judges assessed EVERY final source-only TRAIN summary against
the complete source, with paraphrases allowed and128stored-token cap. The
reference arm diagnoses target quality and is never shown alongside model output.
All16selected cases and all prior first judgments were retained.

| Arm | Terra | Sonnet | Same-case passes |
| --- | ---: | ---: | ---: |
| reference | 16/16 | 16/16 | 16/16 |
| native | 15/16 | 16/16 | 15/16 |

Descriptive word-ID fit: `1.0`; this is not semantic accuracy.
Final checkpoint: `a7941cb921662c04294ab5267aebeaad3ba1d17e8f140fb095e430316d542003`. Canonical cache: `2966` firsts.

These are seen TRAIN cases. They do not establish source-disjoint accuracy or
the95%goal. The last qualified source-disjoint student remains0/128.
Final320sources remain closed/unhashed. Speed has not been measured.
