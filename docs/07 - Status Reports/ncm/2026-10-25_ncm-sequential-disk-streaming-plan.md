# Reuse the bounded NCM disk loader

User-requested systems variant of `ncm-sequential-full732-v2`, after the user
asked to stream from disk and reuse Structure Net's existing setup. Reuse
`ncm_streaming_source_features.FeatureCache` and its lossless FP32 decoder.
Change only `host_feature_cache_bytes` from 24GiB to 256MiB. All sources remain
complete; evicted examples are reloaded from their existing compressed files.
The existing loader checks source identity, file hashes and reconstructed raw
FP32 bytes. No new loader, data conversion, quantization or sampling change.

Keep the original training source and active v2 worker unchanged. Its checkpoint
is written only after the fixed training loop and final diagnostics, so there is
no supported checkpoint from which to replace its in-memory cache mid-run. Do
not terminate it or repeat the full 2,048 updates just to change data placement.
The small-cache launcher is the configuration for future NCM training launches.
The current trained checkpoint remains the candidate for source-disjoint DEV.

Use the original full732 scientific plan, optimizer placement, seeds, objectives,
precision, parity limits, hardware choice and quality gates unchanged. The
launcher applies a process-local configuration override and preserves all
existing producer files. New roots are
`20261025_sequential_full732_disk_{pilot,train}_v1`; their producer uses the v2
schema with the explicit `existing_lossless_disk_LRU_256MiB` configuration
variant and this protocol. A full train launch requires the matching disk pilot.

Run the existing focused disk-loader tests. Replay all 732 actual TRAIN payloads
through the existing loader with a 256MiB limit, verifying the original raw-byte
checksums, full shapes and the resident-feature budget. Check the first and
longest examples again after eviction to establish repeatability. Record load
time and peak process RSS; these are loader observations, not application speed.
The 256MiB limit applies to retained feature tensors, not codec temporaries,
optimizer state, metadata, runtime overhead or the OS's reclaimable page cache.

Then run the unchanged two-update RTX2060 lifecycle/parity/checkpoint pilot with
the smaller cache. It may share the card with the active worker only if the
existing GPU preflight finds at least 2.4GB free; the current available 3.3GB
exceeds the previously measured 2.35GB allocator reservation plus context.
Do not use overlapping runs for throughput comparisons. Preserve actual terminal
receipts and compare hard outputs/checkpoint tensors with the completed original
pilot as a diagnostic. The exact source bytes and objective remain the validity
criteria. Do not claim held-out accuracy or inference speed from this systems test.
