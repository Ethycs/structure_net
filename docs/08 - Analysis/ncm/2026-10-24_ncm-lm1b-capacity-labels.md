# Native NCM capacity-compatible training label revisions

Date: 2026-10-24. Hypothesis `ncm-lm1b-capacity35-source-grounded-labels-v1`.
Classification **capacity35_revisions_quality_or_capacity_failed**. Campaign `data/experiments/ncm_paper_regime/20261024_lm1b_capacity_labels_results_v1/campaign_results.json`, SHA256 `f49329b87b2635c038a3b554b4c48b2ed1977e6f282e7c0081b319d74b0c745d`.

All 35 original TRAIN labels violating a native-frame or storage limit were
selected before generation. One schema-valid Sonnet candidate per full public
source was retained, without showing the previous target to the teacher. The
first four pilot candidates were reused exactly. Capacity/semantic misses never
triggered retries; bounded retries handled transport and schema only.

| Candidate wording | Judge | Passes | Valid |
| --- | --- | ---: | ---: |
| raw | secondary | 27/35 | 35/35 |
| raw | primary | 32/35 | 35/35 |
| bert_roundtrip | secondary | 25/35 | 35/35 |
| bert_roundtrip | primary | 30/35 | 35/35 |

Capacity-compatible candidates: 32/35. Dataset
admitted: **False**. Admission requires all 35 candidates to fit 126 BERT
lexical positions and 128 storage tokens before and after BERT rendering, and
all 35 raw and rendered summaries to pass BOTH qualified judges. Sonnet also
generated the candidates; Terra provides the second model's assessment. Both
judges use full exact public sources and accept paraphrases.

All 2284 earlier first decisions and all pilot firsts remain unchanged;
the resulting canonical cache contains 2424 decisions. Valid negatives
and exhausted invalids were not rerolled. If admitted, all 690 cases are retained,
655 previous targets stay unchanged, and stale word/concept target arrays are
removed before new native encoding. Previous labels and failures are preserved.

This is TRAIN-label preparation, not source-only model accuracy. The native
task-domain codec's separate held-out result remains 52/64; the previous
conditional student remains 0/128. No model trained here, no final source was
opened, and neither the full 95% task goal nor a speedup is established.
