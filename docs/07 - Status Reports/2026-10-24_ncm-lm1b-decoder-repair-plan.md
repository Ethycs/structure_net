# Frozen-book decoder repair from released native NCM weights

Date: 2026-10-24. Hypothesis `ncm-lm1b-frozen-book-decoder-repair-v1`.
Conformance: EXPLORATORY staged training; quality remains unmeasured until a
separate fixed-checkpoint reconstruction and semantic evaluation.

Known before updates: the two padding variants each pass45/64 source-grounded
oracle reconstructions under Terra. Uncompressed targets pass64/64 both judges;
the BERT roundtrip passes59/64 Terra. All decisions remain valid and preserved.
The codec therefore adds failures even with correct target codes. A padding
change and fixed dictionary readout did not solve that problem. The original
source-only student remains0/128. No model or95% goal claim follows from token fit.

Fine-tune ONLY the released LM1B codec's decoder, including its four existing
blocks and output layer, from exact author weights. Keep its encoder, codebooks,
quantization modules, EMA statistics and global normalization frozen and in eval
mode. The original Llama body and its MLPs, and the NCM concept generator, remain
outside this process and unchanged. This is not full codec retraining or a
random restart. Preserving code indices keeps the released generator compatible.

Training uses ONLY the authors' TRAIN split at dataset revision
`6580ba41a4b0a976dde876fbe84d25327617b0e6`, with both binary and offsets SHA-verified.
Select16384 distinct native128 frames by fixed seed79142; reject only empty
lexical frames with explicit receipts. No application target or native validation
frame enters gradient updates. Use20% untouched native frames and80% variable
lexical-prefix reconstruction frames: CLS101,1..126 native lexical tokens,
CLS101 and PAD0 to128. These are unpaired text augmentations, not clipped
application inputs or outputs. Padding stays at the original PAD0 baseline to
isolate decoder adaptation after the failed matched-padding study.

Use full clean12-level codes only, with encoder and books evaluated without
gradients. Minimize native word CE ignoring PAD0. Decoder dropout stays at the
released0.1; no fine-level dropout, code corruption, label smoothing or target
latent optimization. FP32 with TF32 off on physical GPU2, one worker,4GB free,
peak at most4.7GB. AdamW(.9,.95),eps1e-8,weight decay.01,clip1; microbatch8,
accumulation4,1024 fixed updates. Warmup64 to2e-5, cosine to2e-6. Independent
sampler80142,augmentation81142,dropout82142 streams. No quality-based checkpoint
selection; save a step512 recovery file and select only the fixed final1024.

Before full training require a fresh two-update CUDA pilot with real backward,
nonzero decoder updates, no encoder/book gradients or state drift, strict
checkpoint/optimizer/RNG save/reload and exact probe output parity. Use the same
first/longest target-code probes only for read-only lifecycle checks. They never
contribute a training loss. The full run starts again from the release and must
reproduce the pilot's INITIAL outputs, not continue pilot weights.

Preserve contiguous traces with every sampled native row and augmentation,
source hashes, actual worker exits and frozen-state hashes. Stop and preserve
any systems, gradient, numerical or provenance failure. Training loss cannot
establish semantic accuracy. A separate fresh producer must measure the fixed
decoder on all690 unchanged targets and apply source-grounded semantic controls,
retaining the latest2239 first decisions. Baseline BERT judgment differences may
include target ambiguity/judge variability as well as normalization effects;
they are not an information-loss bound. No old negative is rerolled.

All final application sources remain closed. Conditional NCM attachment training
must await appropriate representation and target-quality checks. The unchanged
objective remains95% complete-summary accuracy on the full source-disjoint task,
followed by a matched speed measurement only after quality qualifies.
