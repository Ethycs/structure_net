# Fix the cache-validation call without changing the study

Date: 2026-10-23. The first isolated-quality source-cache pilot exited with
`ValueError: native query dimensions differ` after saving the longest source
cache. Its validation incorrectly passed a rank-two native-token ID array to
`NativeQueryBackbone.query_logits`, whose documented local contract requires
rank-three input embeddings. No optimization updates or quality evaluation ran.

Keep the failed v1 root, producer and source intact, with its observed exit-one
receipt and failure record. The successful 252-case metadata preparation remains
valid and unchanged: it did not execute the broken cache-validation branch.

The v2 cache worker explicitly builds predecessor embeddings before both query
calls. Add a CPU regression test using an actual tiny Llama backbone, source
KV serialization and the real query method, then rerun the same short/longest
pilot under `20261023_isolated_quality_cache_pilot_v2`. Only after terminal
success prepare the full cache under `20261023_isolated_quality_cache_v2`.
Recompute the longest cache; do not reuse the failed v1 file. All source/model,
precision, 252-label, frame, training and quality-gate settings remain fixed.
Record this amendment and v2 producer sources. Future training must select the
v2 cache, never infer success from the partially written v1 manifest.
