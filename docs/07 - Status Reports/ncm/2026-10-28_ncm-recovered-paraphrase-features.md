# Frozen source features for recovered conversational summaries

Extract all40,890 complete role/date-conditioned sources after the full metadata
pass completes. Reuse the existing tested packed-feature producer and its disk
auditor through process-local bindings; preserve the old source and evidence.
This is feature preparation for a subsequent training plan, not an amendment to
the running tree/dense comparison. No target field may reach the body.

The exact original28-layer Llama body runs in BF16 on physicalGPU0 RTX3060.
Character-aligned source features remain FP32 and are compressed losslessly into
256MiB disk shards. Reuse the qualified four compression threads and at most
eight pending matrices. Require8.6GB maximum own GPU allocation,100GiB free disk
headroom under `/data2`, exact original first/longest TRAIN feature controls,
unchanged model hash, no gradients, and complete byte-preserving round trips.

Pilot selection includes the seven metadata controls and the four largest
source-token and native-context cases, deduplicated by source ID. Extract each
twice and require exact FP32 bytes. Then independently load every pilot shard and
verify source/prompt/body bindings, offsets, payload hashes, shapes and finiteness.
Full extraction requires actual successful pilot and disk-audit receipts. It
re-extracts every source, reproduces all pilot hashes, and is followed by the same
complete independent disk replay. No silent precision, length, batch or hardware
change is permitted after a failed contract.

Roots `20261028_recovered_paraphrase_features_{pilot,full}_v1`; binaries beneath
`/data2/structure_net/ncm_recovered_paraphrase_20261028_v1/features/`. The reused
engine's `public_feature_store_*` status strings identify its existing lifecycle
schema; the new hypothesis and input hashes identify this conversational corpus.
Freeze the adapter, underlying producer, plans, tests, tokenizers, metadata and
body weights before launch. Preserve all failures and closed evidence. Existing
tree training and singleton target encoding continue on their own GPUs. This
preparation alone does not admit training, measure accuracy, or establish a speedup.
