# Task-domain NCM decoder adaptation with 64 source sessions held out

Date: 2026-10-24. Hypothesis `ncm-lm1b-domain-decoder-heldout-semantics-v1`.
Classification **domain_decoder_heldout_semantics_quality_failed**. Campaign `data/experiments/ncm_paper_regime/20261024_lm1b_domain_decoder_results_v1/campaign_results.json`, SHA256 `2afebfffbc04110a87e18e4c5c23e89738c15c3d7267f90dcaab03387155cf9a`.

Only the released LM1B codec's 18,870,656 decoder parameters were fine-tuned for
1024 fixed updates. The encoder, books, EMA and normalization stayed unchanged;
all 690 evaluation code hierarchies equal their original released-codec values.
The original Llama body and native NCP generator were absent and unchanged.

The 16,384-frame corpus contains 3,276 author-TRAIN replay frames, 6,554 selected
summary windows and 6,554 selected raw-source windows from 626 task TRAIN cases.
The complete candidate pools contain 649 summary windows and 3,772 source windows.
All 64 semantic cases are excluded by case, session, group, source and turn hash.
The CPU auditor independently regenerated every frame from original text/seeds,
then replayed all 32,768 sampled frames across 1,024 updates and the full schedule. Strict
decoder/optimizer/RNG reload passed. This is reconstruction training; loss is
not task accuracy. No native validation or held-out task text entered gradients.

| Same 64 full public sources | Judge | Reconstructed-summary passes | Valid |
| --- | --- | ---: | ---: |
| uncompressed | secondary | 64/64 | 64/64 |
| uncompressed | primary | 64/64 | 64/64 |
| bert_roundtrip | secondary | 59/64 | 64/64 |
| PAD0_codec | secondary | 45/64 | 64/64 |
| repaired_codec | secondary | 52/64 | 64/64 |

Oracle codec qualified: **False**. Qualification requires 61 of the SAME 64
cases passing both qualified judges, including the uncompressed control. Meaning,
salience, readability, nonempty output and <=128 storage tokens determine passes;
paraphrases are accepted. The inherited 2266 decisions and pilot firsts remain
unchanged; the resulting cache has 2284 firsts. No valid failure was rerolled.

All 690 numeric targets remain, including 25 that need multiple native frames.
All-690 token agreement is 64971/65063; this combines
fit and held-out data. Fitting cases recover 59058/
59113 tokens; held-out cases recover
5913/5950 tokens.
The semantic 64 were inspected in prior studies: they are gradient-disjoint
component development, not a fresh independent test. Their outputs still use
gold target codes. A qualified codec would not establish source-only summarization.
The previous conditional student remains 0/128 on its separate source-disjoint
task development evaluation. No final source was opened or hashed. The full 95%
task goal and any speed improvement remain unestablished.
