# Native NCM padding and semantic reconstruction

Date: 2026-10-24. Hypothesis `ncm-lm1b-CLS-padding-semantic-control-v1`.
Classification **native_codec_semantic_training_control_quality_failed**. Campaign `data/experiments/ncm_paper_regime/20261024_lm1b_padding_semantic_results_v1/campaign_results.json`;
SHA256 `af2335b3a87c83509673f8e693aabf5682a71364f5350d5ee637a10ce9416337`. Exploratory target-aware TRAIN representation control.

| Same64 full public sources | Judge | Complete reconstructed-summary passes | Valid |
| --- | --- | ---: | ---: |
| uncompressed | secondary | 64/64 | 64/64 |
| uncompressed | primary | 64/64 | 64/64 |
| bert_roundtrip | secondary | 59/64 | 64/64 |
| PAD0_codec | secondary | 45/64 | 64/64 |
| CLS_codec | secondary | 45/64 | 64/64 |

Qualified codec arms: `[]`. At least61 of the SAME64 cases must pass
both qualified judges for the reference and codec. Paraphrases are accepted;
faithfulness, salience, readability, nonempty text and<=128 storage tokens decide
each pass. These are oracle target-code reconstructions, NOT source-only model
accuracy. The frozen student's separate source-disjoint result remains0/128.

The sample was fixed by hash before outputs:32 original targets and32 reviewed
targets. The uncompressed, BERT roundtrip, PAD0 codec and CLS codec arms isolate
label quality, normalization and neural reconstruction. All prior2051
first decisions and all pilot decisions are preserved; the final cache contains
2239. Identical packets shared their first decision. No valid negative
or exhausted invalid was rerolled.

All690 numerical targets remain, including25 overlength targets. The only new
input factor replaces tail PAD0 with CLS101. CLS reconstruction recovers
63873/65063 native tokens, versus63872/65063 for PAD0.
No-padding controls (4 cases) retained exact raw predictions.
The native model stayed unchanged, FP32/TF32 off on GPU2, with a repeated memory
pilot and fresh full-load parity. This is not a single-frame claim for overlength
references. Token agreement cannot replace semantic evaluation.

No conditional generator was trained. Original Llama weights and MLPs remain
frozen and absent. No final source was opened, and no95% task or speed claim is
established. The next training decision must address the failed semantic
components rather than select a model by token agreement alone.
