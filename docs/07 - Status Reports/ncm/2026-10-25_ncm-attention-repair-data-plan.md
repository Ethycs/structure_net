# Actual generator-error data for the attention repair experiment

Use two explicitly separate TRAIN sources: 4,949 audited historical core stored
summaries and 732 new engineering summaries. All original targets remain intact
and unreviewed. Historical core exposure is declared; this is not an untouched
whole-LongMemEval experiment. All 512 reserved components, original 128 DEV and
sealed 320 final sources remain excluded. The 156 engineering test sources are
additional held-out evidence and never contribute target codes or gradients.

First verify all 732 existing engineering target-code bindings by independently
re-encoding their complete word frames with the original frozen native codec.
Then encode all 4,949 historical TRAIN targets, preserving all 641 native code
positions. Check frozen codec state, every source/target binding, code ranges
and actual repeatability on first/longest target frames. Use GPU2 FP32, TF32 off;
parent and trainable generator are absent from this stage. Peak limit 3 GB.

In separate fresh roots, run the fixed final 3911eab7... NCM on all 4,949 core
and all 732 engineering TRAIN sources. Inputs to generation are complete frozen
FP32 source features and masks only. No oracle codes, target words or target
latent enters prediction. Bind source/prompt/alignment and feature identities,
preserve every raw generated code and word and its native rendered summary.
Validate against first/longest saved original TRAIN diagnostics, with exact
checkpoint/model state checks before and after. This is training-data preparation,
not semantic accuracy or model training. The unchanged predictions form the
baseline control for the later repair evaluation.

Use the existing bounded 2 GiB host feature loader, which verifies each complete
lossless payload at use. No clipping, dtype reduction, output repair, or semantic
selection. Snapshot each producer, metadata, code and model dependencies, preserve
partial per-case results on failure and require actual completion receipts.
New engineering features must finish before that partition is processed. The
model and original parent/MLPs remain frozen throughout. No new judge calls.

The subsequent fixed-budget repair training plan will use these actual mistakes,
source attention, exact clean-code targets in the loss only, and the completed
3,539-first semantic lineage. Whole-summary 95% remains the eventual goal; no
training code match, word match, partial output or component test can establish it.
