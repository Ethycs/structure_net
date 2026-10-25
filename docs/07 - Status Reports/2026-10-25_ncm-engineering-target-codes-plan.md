# Encode only engineering TRAIN targets with the original frozen codec

In parallel with source-only feature extraction on GPU0, encode the 732 unchanged
engineering TRAIN summaries on physical GPU2. Use the released native LM1B
encoder/VQ, FP32, TF32 off, the full native BERT 128-position frame and all 641
code positions across the original twelve levels. Verify every original target
hash, native word frame and output code range. Repeat the first and longest
lexical target to verify actual new-data geometry, and check the complete frozen
codec hash/absence of gradients before and after. Peak allocation limit: 3 GB.

Preserve all existing labels without repair. They remain unreviewed stored
summaries. Do not encode engineering-test targets or read sealed final sources.
The frozen body and generator are absent, and no trainable weight is updated.
Snapshot the producer and save ordered source/target/code bindings for the
separately planned fixed-budget attention-repair training experiment. This is
training-data preparation, not source-only task accuracy or a speed measurement.
