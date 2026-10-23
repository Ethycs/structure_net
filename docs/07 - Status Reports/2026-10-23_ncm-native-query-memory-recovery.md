# Keep validation cache copies off the training GPU

Date: 2026-10-23. EXPLORATORY systems recovery v2. The original full28-layer
native-query pilot passed its initialization oracle: relativeL2=0.01966365,
top1agreement=0.9921875. It then exited1 with CUDA OOM in `loss.backward`, asking
for126MiB with51.44MiBfree. Its completed optimizer-update count was not saved;
do not infer it. Preserve the entire original root and source unchanged.

Restart a fresh systems pilot from seed42 in
`20261023_native_query_cache_pilot_v2`; do not resume a partial optimizer.
Move ONLY redundant immutable-cache validation snapshots to CPU and compare
them on CPU. The actual source/query caches and all forward/backward computation
remain on GPU0. Keep the same model, rank8attention adapters, batch2,128-target
frame, two updates, optimizer, BF16 precision and all numerical/frozen-weight/
gradient/reload gates. Set and record `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
to reduce allocator fragmentation. No batch, sequence, data or tolerance changes.

Hash the original failed producer, saved oracle and failure receipt as inputs.
CPU cache validation copies have no model/optimizer influence. This recovery
changes resource placement, not the tested conditional-query method. Stop this
version if OOM or any gate recurs; a further change requires new evidence and a
fresh plan/root. No task accuracy, NCM-training or speed claim follows.
