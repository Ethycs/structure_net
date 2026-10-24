# Correct CLS framing for the released LM1B codec

Date: 2026-10-24. Hypothesis `ncm-lm1b-framed-codec-compatibility-v1`.
Conformance: EXPLORATORY input-format follow-up, no training or source-only task
accuracy. The entire earlier native-codec control remains unchanged.

Known before this follow-up: the previous690-target control recovered62404/64323
native tokens, but its first-EOS test passed only4/690. It omitted an initial
CLS token and treated the decoder's initial CLS as the end, producing mostly
empty texts. The author BERT wrapper inserts boundary tokens. Inspection of the
SHA-verified released validation data further shows that its128-position frames
begin AND end with CLS101, with CLS between packed sentences. SEP102 is not the
frame terminator. This data is used for format/control inspection only, never
training or application model selection.

Two factors change prospectively: initial CLS framing and boundary-aware text
parsing. No weights, codec dimensions, levels, precision, output argmax, targets
or original sources change. Do not retroactively rewrite or regrade old outputs.

First run a fixed four-frame control directly on the first512 tokens of the
released native validation file, verifying repeated encode/decode and exact
final-latent/argmax parity between codec.decode and the author's idx_to_zhats
evaluation path. Record native token reconstruction without presenting four
frames as a benchmark replication or task accuracy.

For all690 existing TRAIN summaries, construct CLS101 + at most126 lexical
wordpieces + CLS101, padded to128 with PAD0. Overlength texts use multiple
complete windows and remain in every denominator. PAD0 on short summaries is
an application format adaptation, not asserted to occur in the native packed
validation. The codec remains globally native128; no tensor blocks or learned
weights are tiled. Report the resulting one-frame coverage explicitly.

Parse each raw frame by consuming a predicted leading CLS when present, then
stopping at the next predicted CLS. A missing initial CLS is recorded and no
lexical token is discarded. Missing terminal CLS retains the whole remaining
raw frame. Join predicted lexical IDs before BERT text decoding, preserving
wordpieces across diagnostic window boundaries. No correct target length enters
the parser. Preserve all raw codes and128-position argmax frames.

Reuse exactly the frozen released LM1B codec. FP32, TF32 off, eval/no gradients,
physical GPU2, one worker,4GB free and4.7GB peak limit. Require fresh first/longest
TRAIN pilot repeatability and retained-pilot parity in the full690 measurement.
Record native sequence/token fidelity, all-window boundary correctness and
active invalid IDs separately from meaning. The earlier generic field
first_EOS_correct is replaced in the completed result by all_window_EOS_correct.

Run fresh append-only data/native/pilot/reconstruction/workflow roots. Stop and
preserve any systems/provenance failure. Original Llama weights and MLPs remain
frozen and absent. All application final sources stay closed. A source-grounded
semantic control and a conditional generator remain separate future stages;
this target-aware reconstruction cannot establish the95% task goal or speed.
