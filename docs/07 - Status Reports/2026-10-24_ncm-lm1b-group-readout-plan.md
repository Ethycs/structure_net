# Group-invariant embedding readout before native NCM retraining

Date: 2026-10-24. Hypothesis `ncm-lm1b-group-invariant-readout-v1`.
Conformance: EXPLORATORY mechanistic reconstruction control, no training.
This plan precedes any native-embedding dictionary prediction on the fixed codes.

Known evidence: the full framed LM1B codec recovered63872/65063 native tokens on
all690 TRAIN summaries, with271 exact native sequences and684 correct window
boundaries. It still changes some punctuation and numbers. Its first four native
validation frames recover510/512 tokens. The failed frozen-body student remains
0/128 on the date-matched source-disjoint development evaluation.

The released LM1B encoder is a token-embedding lookup with no positional
embedding, transformer blocks or downsampling, followed by affine-free
GroupNorm4. Within a channel group, that normalization is a positive affine
transform shared across positions. Centering and L2-normalizing each token's
group removes this unknown transform. The same operation on the learned native
embedding dictionary therefore provides a token readout without the correct
sequence's group means or variances. Quantization can still destroy information.

Compare this one fixed nearest-dictionary rule with the retained learned-decoder
outputs from the SAME saved native codes, on ALL690 existing targets including
all25 overlength targets with their complete715 diagnostic windows. Use all30522
native tokenizer entries, including specials. Do not fit parameters, search
normalization variants, inspect target words during prediction, repair outputs,
or use true encoder normalization statistics. New prediction receives only the
dequantized code latents and the fixed normalized pretrained embeddings.

As a separate positive control, apply the identical readout to the continuous
encoder outputs and require exact recovery of every original128-position input,
including padding. Otherwise stop and preserve the failed invariance control.
This target-aware control never supplies predictions to the quantized arm.

Use the same first and longest two TRAIN targets for a repeated CUDA pilot, then
a fresh-model full690 pass retaining those exact pilot outputs. All native
weights remain unchanged; FP32, TF32 off, physical GPU2,4GB free and4.7GB peak
limit, one worker. Preserve every raw128-position prediction and saved-code
identity. Apply the existing predicted-CLS boundary parser without target length.

Report native token agreement, exact native sequences, EOS/BOS correctness and
continuous-control fidelity separately from meaning. Even perfect reconstruction
would not establish source-only summary accuracy. Original Llama and its MLPs
remain frozen and absent, all final source contents stay closed, and the latest
2051 semantic first judgments remain unchanged. No speed claim or95% task success
is admitted. Semantic component qualification and conditional generator training
are separate next stages under the unchanged full task objective.
