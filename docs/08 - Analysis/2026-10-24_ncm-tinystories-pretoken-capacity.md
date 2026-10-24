# TinyStories byte-tokenizer capacity control

Date: 2026-10-24. Hypothesis `ncm-tinystories-pretoken-capacity-control-v1`. Classification **lossless_text_encoding_but_unchanged128_frame_capacity_failed**.
Campaign `data/experiments/ncm_paper_regime/20261024_tinystories_pretoken_capacity_v1/audit/campaign_results.json`, SHA256 `99bd5b69b68b84c7d1c75b62f0e0141703e69bc3864f5260515a3e111364fe1a`.

The pinned author byte vocabulary and merge files were downloaded (203,399bytes),
SHA-verified, safely loaded as plain data and evaluated with the unchanged author
BPETokenizer. All690 current TRAIN targets remain unchanged and in order.
The independent CPU finalizer replayed every encoded sequence and boundary.

| Component measurement | Count |
| --- | ---: |
| TinyStories exact text roundtrips | 690/690 |
| BERT literal text roundtrips | 0/690 |
| Fits128 lexical positions, no boundaries | 181/690 |
| Fits127 lexical positions plus one boundary | 173/690 |
| Fits126 lexical positions plus two boundaries | 170/690 |

TinyStories uses 98683 lexical tokens versus 62115 BERT wordpieces; the longest TinyStories target has 239tokens. Case and Unicode fixtures roundtrip exactly.

Literal BERT text differences reflect normalization and formatting; this does
not imply zero semantic accuracy. Likewise, lossless byte encoding does not
establish neural codec reconstruction or source-only summarization. The byte
encoder is not a direct128-position replacement for these unchanged labels.
Testing it would require a new framing/root-policy study, shorter targets or
a larger native model; no claim about those untested variants is made.

No neural checkpoint, DEV labels or final sources were used by this control.
No teacher or judge requests were issued and the canonical semantic cache was
unchanged. The ongoing LM1B training and fixed source-only evaluation retain
their original producers and gates. The95%task goal and speed remain unproven.
