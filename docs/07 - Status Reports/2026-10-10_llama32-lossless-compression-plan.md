# Lossless component compression: exploratory plan

Date: 2026-10-10. Hypothesis `llama32-3b-lossless-component-compression-v1`.
Profile: EXPLORATORY. Parent: completed exact residual study, commit `3fdc7b9`.

Question: which existing components can be stored in fewer bytes while
recovering every tensor bit and all checkpoint metadata? The directional
expectation is that reversible bit rearrangement followed by entropy coding
will reduce storage, but factors plus exact residuals may remain larger than
compressing the original weights directly. This is one frozen model, not
independent training replicates or an accuracy experiment.

The original FP16 checkpoint, rank-215 FP16 factors, and all 197 existing FP32
residual shards (including repair indices and values) are immutable inputs.
Their SHA-256 identities must match the parent artifacts before compression.
Normalization vectors and model/configuration metadata count toward storage.
Tied embedding/output weights are represented once, as in the source.

This is explicitly a CPU-only, single-process byte-codec experiment. No
training, decomposition, GPU work, or new inference-quality evaluation is
needed. The parent failed accuracy gate remains unchanged. CPU timings are
descriptive implementation costs, not isolated inference benchmarks.

Compare Zstandard level 3 after four exactly invertible layouts: unchanged
bytes, byte shuffle, bit shuffle, and adjacent-value XOR followed by bit
shuffle. Screen the first, middle and final 65,536 elements of each matrix/core
from embedding and layers 0, 14 and 27, plus all normalization vectors and all
repair arrays. These screen slices are selected before inspecting sizes.
Per component group (original, factors, residuals), choose the layout with
the lowest total screened payload bytes; break ties by listed order.
The screen selects a codec, not a held-out generalization claim. Full-model
totals must come from actual encoded data, never sample extrapolation.

Apply each selected layout to its complete group in chunks of at most 1 MiB.
Store a raw chunk whenever compression would expand it. Retain actual archives
and complete typed metadata, including shapes, dtypes, dictionary keys and
tensor references. Count payload plus manifest bytes; report filesystem block
allocation separately if needed. Hash and decode every persisted chunk, compare
it directly with the source bytes, and record per-tensor hashes and totals.
Any mismatch is a hard stop; no lossy fallback. Empty tensors, signed zero,
infinities and NaN payload bits must survive the codec tests. An archive decoder
must work from the archive alone without the original checkpoint.

Primary outcomes: all-tensor bitwise round trip, actual total archive bytes,
and ratio to uncompressed tensor bytes for each group. Secondary: family
breakdown (embedding, attention, MLP, norms, factors, residual/repairs), bit
occupancy on screen slices, encode/decode wall time, and the total of compressed
factors plus compressed residuals versus compressed original weights.
Classification: `lossless_storage_reduction` if every required tensor passes
and at least one complete representation is smaller than the original raw
FP16 model; `lossless_no_storage_reduction` if exact but not smaller;
`lossless_roundtrip_failed` if any reconstruction fails. Confirmation remains
false because this is exploratory and one checkpoint.

Use a new root under
`data/experiments/llama_lossless_compression/20261010_v1/`, with separate
shakedown artifacts. Freeze source/config/protocol hashes before the screen.
Do not reuse partial outputs or modify completed parent evidence. Record
strict JSON, all input/output hashes, 199 archive results (one original,
one factor checkpoint, 197 residual shards), an aggregate, report and registry
read-back. Commit the durable source/tests/docs. Stop on provenance mismatch,
codec failure, insufficient disk space, or producer changes.

Storage compression does not itself reduce dense GEMM work, guarantee faster
inference, or establish compressed GPU execution. Decoding restores the original
weights and their accuracy. A storage gain cannot rescue the prior compute or
accuracy gates.
