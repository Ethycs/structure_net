# Native NCM capacity-compatible training label revisions

Date: 2026-10-24. Hypothesis `ncm-lm1b-capacity35-critique-revision-v1`.
Classification **capacity35_critique_revisions_quality_or_capacity_failed**. Campaign `data/experiments/ncm_paper_regime/20261024_lm1b_capacity_revision_results_v1/campaign_results.json`, SHA256 `eaeb64b16326fe4066d5740e3d4492ad8cfdee7a1114fe7cf2ca65fbb534512f`.

The original 35-case capacity population is retained. The 23 first candidates
that passed capacity and both judges in both wordings were kept exactly. The
other 12 received one genuine critique-guided revision from the full source,
previous candidate, BERT rendering and recorded issues. The new prompt aims
at 30-40 words and spells out essential units/numbers. The first four pilot
candidates were reused exactly. Capacity/semantic misses never triggered
schema retries. Unchanged failed text shares its original negative decision.

| Candidate wording | Judge | Passes | Valid |
| --- | --- | ---: | ---: |
| raw | secondary | 34/35 | 35/35 |
| raw | primary | 35/35 | 35/35 |
| bert_roundtrip | secondary | 33/35 | 35/35 |
| bert_roundtrip | primary | 35/35 | 35/35 |

Capacity-compatible candidates: 35/35. Dataset
admitted: **False**. Admission requires all 35 candidates to fit 126 BERT
lexical positions and 128 storage tokens before and after BERT rendering, and
all 35 raw and rendered summaries to pass BOTH qualified judges. Sonnet also
generated the candidates; Terra provides the second model's assessment. Both
judges use full exact public sources and accept paraphrases.

All 2424 earlier first decisions and all pilot firsts remain unchanged;
the resulting canonical cache contains 2472 decisions. Valid negatives
and exhausted invalids were not rerolled. If admitted, all 690 cases are retained,
655 previous targets stay unchanged, and stale word/concept target arrays are
removed before new native encoding. Previous labels and failures are preserved.

This is TRAIN-label preparation, not source-only model accuracy. The native
task-domain codec's separate held-out result remains 52/64; the previous
conditional student remains 0/128. No model trained here, no final source was
opened, and neither the full 95% task goal nor a speedup is established.
