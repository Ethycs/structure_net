# Lossless tensorization with an exact residual network

Date: 2026-10-10. Hypothesis: `llama32-3b-lossless-residual-cotenancy-v1`.
Exploratory fixed assay, replacing the lossy screen at the user's explicit request.
The user selected **exact correction; preserve losslessness**. The stopped parent
campaign and all its completed evidence remain intact. No claim is made that its
four-configuration screen was completed.

## Contract and controls

Use the same complete Meta Llama-3.2-3B-Instruct checkpoint and 28-layer graph,
with FP16 inference and the existing frozen rank-215 SVD factors. No training,
truncation of the residual, quantized residual, or learned approximate correction
is permitted in the lossless arm. The approximate factors alone are a diagnostic.

Require every reconstructed weight to equal the original FP16 weight **bit for
bit**, including signed zero. Those FP16 values exactly represent the original
BF16 checkpoint values. Require all measured logits and generated token IDs to
equal the original FP16 implementation bit for bit. Mathematically exact
factorizations are not automatically bitwise-exact floating-point executions.

Two constructions are distinguished explicitly:

1. **Stored matrix residual**: predict P in FP32 from the two FP16 factor cores,
   store R = W - P in FP32, and reconstruct (P + R) in FP16. Store exact original
   values for any exceptional entries that do not reconstruct bit for bit.
   Verify all matrices, preserve residual shards and hashes, and count every
   residual, repair, factor and normalization byte. Estimate capacity from this
   hard storage lower bound before attempting any impossible GPU allocation.
   Decoding then executing the identical dense operator preserves the original
   arithmetic, but does not remove the dense execution cost.
2. **Exact residual-network execution control**: retain the original dense
   weights as the lossless correction oracle. For each projection, embedding
   and output head, compute the tensorized prediction and the original dense
   result, form their FP32 difference and add it back. Apply bitwise output repair
   against that dense result if floating-point addition fails to reproduce it.
   This is an explicitly expensive exact control, not a compressed network or
   evidence that a smaller exact residual has been discovered. Count its dense
   weight storage, extra factor storage and all executed work.

Both constructions represent a correction network without any approximation to
the required correction. Do not claim that keeping a dense correction oracle is
a speedup. A direct factor-output-plus-stored-matrix-residual sum is only a
diagnostic unless it independently passes bitwise output equality: floating-point
reassociation may change its outputs despite exact weight reconstruction.

## Data, accuracy and execution

Use the already hashed corrected fixtures `inputs_v2/fixtures.json` (SHA-256
`7390e72e11379437ee2c22aff2b984190ff15eaa76312c51f5dca2723b3c9e9c`): four fresh
validation examples with a fixed worked example, followed by ten sealed final
examples. No corrected-fixture model result has been observed at registration.
The original invalid Q4 assay remains invalid and does not define the reference
for this losslessness test. The reference is the unmodified FP16 network.

The FP16 reference must first satisfy >=90% fragment coverage and >=95% supported
output lines on validation. If it fails, stop final summary evaluation as an
invalid task assay. Otherwise evaluate both the original and exact residual
network on the ten final examples, with the same 96-token greedy policy and
source-fragment metrics. Report exact counts and paired generated token IDs.
This small synthetic assay does not establish production accuracy.

CUDA numerical controls use 128/512-token prefills and cached decoding, the same
weights, token IDs and attention backend. Timing: one warmup, five alternating
samples, synchronized GPU/wall timing, 128/512-token prefills and 32 decode steps
after a 128-token prefill. Measure both complete-model arms wherever storage
permits. Count 1024-token KV capacity per physical model. Check 1/2/4-model
capacity using exact byte lower bounds; report unattempted configurations as
storage-infeasible, not measured OOMs. Use independent physical model-axis slices
for a two-replica query-projection packing diagnostic.

Success requires bitwise reconstruction, bitwise measured outputs, no loss of
summary accuracy and >=1.10x throughput or lower resident memory than the matched
reference. Lossless-but-more-expensive is a valid completed negative efficiency
result. Original Q4 memory/timing are a secondary deployment comparison from the
preserved parent campaign. Never infer a general impossibility theorem about all
tensor networks from these two exact constructions.

Preserve source snapshots, producer/model/fixture fingerprints, strict JSON,
residual/checkpoint hashes, full generations, raw timings, a measured report and
conservative registry read-back. Use new `20261010_lossless_*` artifact roots.
