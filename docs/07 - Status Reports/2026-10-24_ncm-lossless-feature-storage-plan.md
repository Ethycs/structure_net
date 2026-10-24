# Lossless FP32source-feature storage control

2026-10-24. Hypothesis `ncm-lossless-FP32-source-feature-storage-control-v1`.
The larger source corpus may exceed available disk as raw FP32features. Before
adding a new feature quantization factor, measure lossless cache compression.
This concerns training-data storage; it does not shrink model parameters or
establish neural task accuracy or model token throughput.

Choose16of the655unchanged original TRAIN sources by fixed hypothesis/ID hash:
4short and3in each long bin. No teacher/model-output selection. Read the original
cached FP32features without editing them. Preserve every source position and
all3072channels. Compare two zlib-level1layouts: original interleaved FP32bytes
and a reversible permutation grouping each of the four byte positions together.
No floating-point arithmetic, rounding, residual approximation or quantization.

Require reconstructed bytes to match the original exactly in EVERYcase and both
layouts, with raw SHA, full shape, dtype, source identity, input/payload/header
hashes and exact compressed byte accounting. Decode the stored artifact, not
only an in-memory buffer. Reject incomplete/trailing/corrupted payloads or wrong
headers. Unit controls include low mantissa bits, signed zero and IEEE special
bit patterns; approximate numeric equality is insufficient.

Freeze source/codec/plan/tests/auditor before producing artifacts. Record actual
tool completion, then independently decode all32saved payloads against all16
original source tensors and replay storage aggregates. Report per-band and total
ratios. The sample does not guarantee the compression ratio of a different
corpus. CPU serialization timings are diagnostics only, not model-speed evidence.
Future full-corpus extraction needs a separate frozen protocol and actual disk
checks. This study never initializes a GPU, reads selected final sources, trains
weights, changes canonical judgments or alters the running native655/DEV jobs.
