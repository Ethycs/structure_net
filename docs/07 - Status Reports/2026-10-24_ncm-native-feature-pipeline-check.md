# Native655 failure: source feature pipeline systems check

2026-10-24. Read-only systems diagnostic, not a new quality experiment or a
rescue of the failed native655 hypothesis. The completed registered native655
development result is already known: Terra accepted 0/64 short and 0/64 long
summaries, although the fixed TRAIN diagnostic outputs are fluent.

Before assuming a larger training set fixes this, check a possible mismatch
between TRAIN and DEV source features. Require the actual completed, registered
and independently read-back failed DEV workflow. Preserve all checkpoints,
raw outputs, judgments and source features; final320 sources remain unopened
and unhashed. This diagnostic requests no new labels or semantic judgments.

CPU: independently rebuild the original source-only prompt, explicit23Oct2026
date, native/GPT2tokenization and complete character alignment for all655TRAIN
and128DEV cases. Compare every geometry field exactly. Verify cached feature
hashes, source/prompt/alignment bindings and full FP32shape. Record source-level
RMS, centeredRMS, positionL2 and duplicate-adjacent-position fraction, reporting
cohorts separately. Strip historical target columns before rebuilding metadata.

GPU: after actualCPUexit0, use physicalGPU0/nativeBF16 with TF32disabled and the
original unchanged28layer Llama body. Recompute the first and longest distinct
source in EACH of TRAIN, shortDEV and longDEV (six fixed probes). Keep every
source position and original FP32character alignment. Compare allFP32bytes
with original cached features, retaining recomputed tensors and numerical
differences even if parity fails. Require unchanged body hash and peak allocated
memory<=8.6GB. No model training and no checkpoint selection.

Freeze code, this plan and tests before launch. Use the fresh append-only root
`data/experiments/ncm_paper_regime/20261024_native655_feature_pipeline_check_v1`.
Record actual tool exits; CPU replay of all six stored comparisons is required
before interpretation. Exact parity only rules out a pipeline mismatch on the
six probes; it does not prove absence of every integration error. Statistics
are descriptive, not semantic task accuracy. This systems check does not
establish95%quality, paper replication, model compression or throughput.
