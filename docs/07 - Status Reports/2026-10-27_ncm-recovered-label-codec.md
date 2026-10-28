# Frozen codec capacity on the recovered label sample

Run the existing repaired native codec on all 128 labels in the fixed recovered
sample, in the original order. This is a target-aware reconstruction diagnostic,
not source-only summarization or semantic task accuracy. Do not filter the sample
by the new saved-label grades. Use the same codec tensor identity as the current
public and paired task training, loaded strictly from the fixed BROAD checkpoint.

Build the established native128 frame: CLS/BOS, all lexical BERT tokens, CLS/EOS,
then PAD. Require 1–126 lexical tokens and no specials/unknowns; never truncate.
Save original text and tokenizer-normalized reference separately. Use raw argmax
over the full native vocabulary and the existing native renderer, with no repair.
Encode each target twice and require identical complete 641-code hierarchies.
Decode twice for the first case and the longest native target before processing
the remaining cases. Preserve original TRAIN control codes to detect a wrong
checkpoint or packing rule.

Use locked Pixi, physical GPU2 RTX2060SUPER, FP32, deterministic algorithms and
TF32 off, batch one, no optimizer. Require 3.5 GB free initially and own peak
allocation below 3.5 GB. Preserve both other training worker identities. Freeze
inputs and producer before inference in the fresh root
`20261027_recovered_label_codec_v1`, record the actual terminal and retain failures.

Report full-denominator exact rendered reconstruction, content token agreement,
boundary/active-frame agreement, invalid/empty outputs and digit-sequence retention,
overall and per fixed source-length stratum. Digit-sequence retention means the
ordered runs of digits in normalized reference and output; it does not prove
numeric reasoning or correct units/relations. Original-to-normalized digit changes
are reported separately. Save codes, raw IDs and decoded text for every case and
verify the frozen codec hash and lack of gradients at completion. Do not call this
a model-quality improvement or a speed benchmark.
