# Frozen codec reconstruction on candidate public training summaries

Exploratory hypothesis `ncm-public-xsum-codec-v1`: the unchanged repaired codec
can reconstruct the newly screened XSum training summaries well enough to
support a subsequent conditional-pretraining study. This is a target-aware
component diagnostic, with no training, semantic judge calls, source-only
generation, speed claim, or access to the sealed final320.

The complete public TRAIN inventory contains 163,051 examples passing the
current token and source screens. Their native target median is 26 tokens,
substantially shorter than the application training labels. They are not
verified factual gold, and the inventory does not establish comprehensive
source disjointness. This diagnostic does not admit them for model training.

Before any new reconstruction output, fix these samples from the unchanged
screened population. Rank article IDs by SHA-256 of the hypothesis ID, a null
separator, and article ID. Select the lowest 256 as the representative sample.
From the remainder select the 32 longest native summaries, breaking ties by the
same rank. From the remaining summaries containing a digit select the lowest
32 ranks. These separate long and numerical stress groups are not population
accuracy estimates. Record the full selection and preserve the original target
strings without truncation or rewriting. Add all 16 existing TRAIN codec
controls solely to check alignment with the previous round-trip experiment.

Compare two fixed arms on identical cases: the official released LM1B codec
and the repaired codec from the completed BROAD checkpoint
`74cdce0eb14bb99c801b6ec0a025c092f95249663becc05561af8dfbad6040ff`.
The repaired codec tensor hash must remain
`eb962706f10c693871f9c6df007e9fd3e23004c91c6258f435cdb09abf7fb366`.
Load a standalone codec, without source-attention adapters or a generator.
Both arms encode each reference into all 641 native codes and decode once in
FP32, with TF32 disabled and evaluation mode, on physical GPU1 RTX2060.
Require identical encoded codes between arms, unchanged tensors, finite logits,
and the full 128-position frame with original CLS/EOS framing and padding.

Use a two-public-case CUDA pilot: the first representative case and the longest
selected public target. Repeat both arms exactly. Also require all 16 TRAIN
controls in each arm to match the saved raw word IDs and codes from the earlier
codec-only measurement. The full run must reproduce these pilot outputs.
Use one GPU process, one case per forward, and a 4.9 GB allocation limit.

Report every case and, separately by group and arm, native-rendered exact
reconstruction, active-token sequence equality, content-token agreement without
PAD/CLS inflation, empty or invalid output, output length, and digit-sequence
preservation among references containing digits. Native rendering is uncased;
capitalization and tokenizer normalization are outside these measurements.
Digit equality does not measure numerical reasoning, units, or factual truth.

The descriptive gate for the representative sample requires at least 244/256
exact reconstructions, at least 99% content-token agreement, and at least 95%
digit-sequence preservation among its digit-bearing cases. All counts must be
complete and valid. Stress and TRAIN-control scores cannot rescue that gate.
A passing repaired codec supports proceeding to a separately specified training
study. Failure calls for decoder/domain diagnosis before assuming a larger
conditional-training pool solves the problem. The released comparison is a
fixed diagnostic, not an opportunity to silently select a new model.

Run focused CPU tests and CPU preparation before the CUDA pilot and full run.
Preserve failures, actual process receipts, model/data/source/environment hashes,
all code and output IDs, and a separate replayed aggregate. Use append-only roots
`data/experiments/ncm_paper_regime/20261026_public_xsum_codec_{prepare,pilot,eval,results}_v1`.
Run from the repository root with locked Pixi `ncm`; use an empty
`CUDA_VISIBLE_DEVICES` for preparation and physical GPU1 for inference.
The final endpoint remains at least 95% complete-summary semantic task accuracy
on appropriately source-disjoint application evaluation. This diagnostic cannot
meet that endpoint or reopen any failed study.
