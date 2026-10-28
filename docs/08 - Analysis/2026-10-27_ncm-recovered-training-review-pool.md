# Recovered task-data review pool

Of the census's 244,587 length-fitting raw-text summary pairs, **104,503 pairs**
pass the current original-source provenance, protected-source and native-token
capacity filters. They cover **55,701 distinct input texts** in **11,282 source
components**. These are saved-label review candidates; they are not yet admitted
training data and this count says nothing about NCM model accuracy.

The worker completed successfully and an independent readback checked every
export row against the disk-backed pair, source, proof, decision and outcome
tables. It also checked content hashes, source/target limits, unique pair IDs,
export hash and complete exclusion counts. The review pool is 43.8 times the
2,384-pair expanded training arm by pair count; multiple targets and overlapping
sources mean this is not 43.8 times as many independent examples.

| Outcome | Pair count |
|---|---:|
| Raw pairs examined | 244,587 |
| Retained for label review | **104,503** |
| Original-source provenance unresolved | 89,478 |
| Protected or previously blocked source component | 43,551 |
| Target does not fit one complete native frame | 24,009 |
| Native special or unknown token | 126 |

Exclusion categories overlap and must not be summed. The 15,049 fitting merge
pairs are deferred from this raw-text pass. Native capacity means 1–126 lexical
tokens plus the frame's boundary tokens, without truncation; it does not establish
successful codec reconstruction. The application's stored-target cap remains
128 `cl100k_base` tokens.

There are 69,458 retained pairs with 192–511-token sources and 35,045 with
512–2,048-token sources. Across pair rows, the pool contains 45,212,946 source
tokens and 4,240,770 target tokens in the stored tokenizer; repeated inputs count
repeatedly. The preparation's measured peak host RSS was 1.38 GB.

The source guard excludes whole connected groups touching the existing protected
cohorts, links original families/turns/groups, and checks exact or substantial
normalized text overlap against the 192 open held-out sources. It uses reserved
metadata for the sealed final cohort; full-text overlap against the sealed 320
cases has not been audited. Those files were not opened. Thus the current guards
must not be described as a completed final-test contamination audit.

Source proof is currently available for 85,160 of the census's 173,801 distinct
raw inputs. The 88,641 remaining inputs stay explicit. Most have native-source
occurrence identifiers; tracing those identifiers through original body stores
may recover more source provenance. They are not automatically fixtures or
invalid data, and remain outside this first review pool.

The inspectable JSONL is
`data/experiments/ncm_paper_regime/20261027_recovered_pair_training_pool_v1/review_candidates_train.jsonl`.
It contains source and target text, hashes, token counts, component and proof
references, plus explicit `training_admitted: false` and
`semantic_label_quality_reviewed: false` flags. It is 345,586,303 bytes with SHA-256
`4ea297064370d4b9cfa40c170a7b79db4d65c4a6991fc33f223852e399296003`.

The independent readback is in
`data/experiments/ncm_paper_regime/20261027_recovered_pair_training_pool_audit_v1`;
actual tool session `27138`, terminal `ad101b`, exit zero. The producer worker's
actual systemd terminal was invocation `f612c2b63af5454dbdda361712af7d86`, PID
1708260, exit zero without restarts, observed in tool chunk `8ac2bc`.

A fixed 128-case saved-label sample has now been selected before judging: 64 short
and 64 long sources, with one pair per source component. Its artifacts are in
`data/experiments/ncm_paper_regime/20261027_recovered_label_sample_v1`. This balanced
sample tests label quality and is not a pair-weighted corpus accuracy estimate.
It has not yet been semantically graded. Review must use the qualified 128-token
source-grounded rubric and preserve existing first judgments. No new teacher or
judge requests were made by preparation or readback.
