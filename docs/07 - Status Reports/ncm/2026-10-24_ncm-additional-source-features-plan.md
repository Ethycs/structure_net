# Original frozen-body FP32features: larger TRAIN corpus

2026-10-24. Hypothesis `ncm-additional10000-frozen-body-lossless-FP32-features-v1`.
This protocol is prepared while native655trains; it does not run concurrently
on GPU0. Admit extraction ONLYafter that fixed native655model's fullDEVworkflow
actually completes, is registered and independently backend verified, and fails
the unchanged realDEVgate. If DEV succeeds, final qualification takes priority;
do not start this larger GPU job. Final320sources remain closed/unhashed here.

Require completed independently rebuilt10000source metadata and audited exact
FP32cache codec. Fixed source index72db66dd94fb8b062ebe9cce830abba0b3b8a50e3297a3d4fca406396f409f16
has7,777,015source positions, maximum3777GPT2positions/2275native prompt tokens.
Do not read new teacher labels. Historical TRAIN geometry/feature references
include target columns, but only their source prompt/alignment/features are used.

Use the SAME original28-layer Llama body and MLPweights, unchanged BF16forward,
native chat system and explicit23Oct2026date, complete context/source positions
and original FP32character-overlap alignment. No training, source clipping,
feature quantization, layer removal, target/code input or model-weight change.
Verify original body hash before/after and absence of gradients. Keep TF32off,
physicalGPU0/nativeBF16 and the original8.6GBallocated-memory limit.

Before each pilot/full worker, recompute the original first/longest TRAIN feature
references and require EXACTFP32byte parity with their immutable cached tensors.
A fresh feature pilot uses the first new source, longest native prompt and
longest GPT2source, deduplicated to2or3cases. Repeated body extraction and stored
payload decoding must reproduce every FP32byte. Reuse ALLpilot entries exactly
in the full10000cache; no output/case selection by quality or byte similarity.

Store FP32features with audited byte-shuffled zlib1compression. This is reversible
byte permutation plus compression, not BF16feature storage or approximate
tensorization. Headers bind source ID/hash, prompt/alignment hashes, original
body fingerprint, dtype, all3072channels, full shape and raw reconstructed SHA.
Manifest entries separately hash header and payload. The loader validates all
bindings, finite FP32shape and payload bytes before returning features. Preserve
all source positions. The sample storage control achieved31.8%of raw size, but
full-corpus footprint must be measured rather than assumed.

Freeze source/plan/tests/auditor before extraction. Write each finished feature
and entry independently; aggregate the final manifest once, avoiding repeated
whole-corpus manifest rewrites. Keep an8GiBfree-disk floor; on failure preserve
the partial producer rather than changing precision or dropping sources. Record
actual child/whole-tool exits and independently decode/bind ALL10000source caches
on CPU, retaining raw/stored byte totals, peak memory and actual original-body
state. This does not establish student accuracy or model speed. Canonical first
judgments and all existing feature/checkpoint files stay unchanged.
