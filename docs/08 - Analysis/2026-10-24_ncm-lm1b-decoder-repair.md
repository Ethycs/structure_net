# Released NCM decoder fine-tune

Date: 2026-10-24. Hypothesis `ncm-lm1b-fixed1024-decoder-repair-semantic-v1`.
Classification **decoder_repair_oracle_semantics_quality_failed**. Campaign `data/experiments/ncm_paper_regime/20261024_lm1b_decoder_repair_results_v1/campaign_results.json`, SHA256 `9fcefdba0ef580599b0b226d762b898b180601756ebfd57513ba92e7c7441075`.

The authors' exact LM1B 111M codec was loaded before updates. Only its 18,870,656
decoder parameters were fine-tuned for 1024 fixed updates. The encoder, codebooks,
EMA and normalization state stayed unchanged. The original Llama and native NCP
generator were absent. All 690 evaluation code hierarchies match the release.
The native TRAIN split supplied 16384 selected frames, 20% full128 and 80% variable
short reconstructions. Application summaries and native validation were not used
for gradients. Decoder/optimizer/RNG reload and all 1024 sampled updates were
audited. Training loss is not task accuracy.

| Same 64 full public sources | Judge | Reconstructed-summary passes | Valid |
| --- | --- | ---: | ---: |
| uncompressed | secondary | 64/64 | 64/64 |
| uncompressed | primary | 64/64 | 64/64 |
| bert_roundtrip | secondary | 59/64 | 64/64 |
| PAD0_codec | secondary | 45/64 | 64/64 |
| repaired_codec | secondary | 39/64 | 64/64 |

Oracle codec qualified: **False**. Qualification requires 61 of the SAME 64
cases passing both qualified judges, including the uncompressed control. Meaning,
salience, readability, nonempty output and <=128 storage tokens determine passes;
paraphrases are accepted. The inherited 2239 decisions and pilot firsts remain
unchanged; the resulting cache has 2266 firsts. No valid failure was rerolled.

All 690 numeric targets remain, including 25 that need multiple native frames.
Repaired token reconstruction is 64069/65063, versus
63872/65063 before fine-tuning. These are gold-code TRAIN reconstruction probes,
not a source-only summarizer or held-out task accuracy. The previous conditional
student remains 0/128 on its source-disjoint development evaluation. No final
source was opened. The 95% task goal and any speed improvement remain unestablished.
